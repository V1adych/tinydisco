"""Meta-trains an update rule (nn.MetaNetwork) on a population of agents spread over many environments, after LPG
(Oh et al. 2020): python src/disco.py [--envs Chain-short Grid-dense ...] [--track]

Every meta step, each agent is updated num_updates times by the meta-network and then judged by A2C on a fresh rollout;
the meta-gradient flows back through those updates. Agents live on between meta steps (truncated backprop) until their
lifetime ends. The agent architecture and hyperparameters per environment are LPG's table 3, see SPECS.
"""

import time
from dataclasses import dataclass
from functools import partial
from typing import Any, NamedTuple

import jax
import optax
import tyro
from flax import nnx
from jax import numpy as jnp

from checkpoint import save_checkpoint
from envs import make
from log import Logger
from nn import ActorCritic, MetaNetwork, num_params


class Spec(NamedTuple):
    """An environment's agent: FeedForward sizes, adam learning rate and lifetime in env steps."""

    channels: tuple[int, ...]
    hidden: tuple[int, ...]
    lr: float
    lifetime: int = 30_000_000
    kl_cost: float = 0.5  # alpha_y, the weight of KL(y || y_hat) in the agent's loss


# LPG trained its tables by SGD with lr 40, which diverges here: a batch mean over 320+ samples visits the chain's first state far more than the others
TABULAR = Spec((), (), lr=0.005, lifetime=1_000_000)
SMALL_CONV = Spec((16,), (32,), lr=0.001)
SPECS = {
    "Chain-short": TABULAR,
    "Chain-short_noisy": TABULAR,
    "Chain-long": TABULAR,
    "Chain-long_noisy": TABULAR,
    "Chain-distraction": Spec((), (16,), lr=0.005, lifetime=2_000_000),
    "Grid-dense": SMALL_CONV,
    "Grid-long_horizon": SMALL_CONV,
    "Grid-small": Spec((), (32,), lr=0.001),
    "Grid-small_sparse": Spec((), (32,), lr=0.001),
    "Grid-very_dense": Spec((32, 16, 16), (256,), lr=0.001),
    # not in LPG; the grid world agent fits MinAtar's 10x10 boards
    "Asterix-MinAtar": SMALL_CONV,
    "Freeway-MinAtar": SMALL_CONV,
    "SpaceInvaders-MinAtar": SMALL_CONV,
}


@dataclass(frozen=True)
class Config:
    envs: tuple[str, ...] = tuple(SPECS)
    agents_per_env: int = 16
    num_envs: int = 64  # per agent
    num_steps: int = 20  # rollout length, one rollout per agent update
    num_updates: int = 5  # agent updates per meta step, all backpropagated through
    meta_steps: int = 20_000
    seed: int = 0
    gamma: float = 0.99
    gae_lambda: float = 0.95  # for the critic's targets and the outer advantage
    # meta-network
    y_dim: int = 30
    meta_hidden: int = 256
    meta_lr: float = 1e-4
    max_grad_norm: float = 1.0
    # outer loss regularisers, LPG eq. 4
    ent_coef: float = 0.01
    y_ent_coef: float = 1e-3
    pi_hat_l2: float = 1e-3
    y_hat_l2: float = 1e-3
    # logging
    log_interval: int = 10  # in meta steps
    save_interval: int = 1000  # in meta steps, 0 disables checkpoints
    checkpoint_dir: str = "checkpoints"
    track: bool = False
    wandb_project: str = "meta-rl"
    run_name: str | None = None


class Block(NamedTuple):
    """An environment and what all of its agents share."""

    name: str
    env: Any
    spec: Spec
    new_model: Any  # new_model(rngs=...) -> ActorCritic
    graphdef: Any
    optimizer: optax.GradientTransformation


class Agent(NamedTuple):
    """One learner: its networks, optimiser, envs and age in env steps. In a block these carry a leading axis over agents."""

    params: Any
    opt_state: Any
    env_params: Any
    env_state: Any
    obs: jax.Array
    ep_return: jax.Array
    ep_length: jax.Array
    age: jax.Array


class Transition(NamedTuple):
    obs: jax.Array
    action: jax.Array
    reward: jax.Array
    terminated: jax.Array
    boundary: jax.Array  # terminated or truncated: the next stored obs starts a new episode
    y_next: jax.Array  # the agent's y at the true next state, before any reset
    value_next: jax.Array


def make_block(name: str, cfg: Config) -> Block:
    env, params = make(name)
    spec = SPECS[name]
    new_model = partial(ActorCritic, env.observation_space(params).shape, env.num_actions, cfg.y_dim, spec.channels, spec.hidden)
    graphdef, _ = nnx.split(new_model(rngs=nnx.Rngs(0)))
    # eps_root keeps the meta-gradient finite through adam's sqrt for parameters whose gradient is exactly zero
    return Block(name, env, spec, new_model, graphdef, optax.adam(spec.lr, eps_root=1e-8))


def init_agent(block: Block, cfg: Config, key: jax.Array) -> Agent:
    key_model, key_params, key_env, key_age = jax.random.split(key, 4)
    _, params = nnx.split(block.new_model(rngs=nnx.Rngs(key_model)))
    env_params = block.env.sample_params(key_params) if hasattr(block.env, "sample_params") else block.env.default_params
    obs, env_state = jax.vmap(block.env.reset, in_axes=(0, None))(jax.random.split(key_env, cfg.num_envs), env_params)
    zeros = jnp.zeros(cfg.num_envs)
    # lifetimes start staggered so the population always spans young and old agents
    age = jax.random.randint(key_age, (), 0, block.spec.lifetime)
    return Agent(params, block.optimizer.init(params), env_params, env_state, obs, zeros, zeros, age)


def rollout(block: Block, cfg: Config, agent: Agent, key: jax.Array) -> tuple[Transition, dict, Agent]:
    """num_steps steps in every env of one agent: transitions (T, N, ...), episode stats summed over the rollout, the agent with its envs advanced."""
    model = nnx.merge(block.graphdef, agent.params)

    def step(carry, key):
        env_state, obs, ep_return, ep_length = carry
        key_action, key_step = jax.random.split(key)
        logits, _, _ = model(obs)
        action = jax.random.categorical(key_action, logits)
        next_obs, env_state, reward, done, info = jax.vmap(block.env.step, in_axes=(0, 0, 0, None))(jax.random.split(key_step, cfg.num_envs), env_state, action, agent.env_params)
        _, y_next, value_next = model(info["final_observation"])
        ep_return, ep_length = ep_return + reward, ep_length + 1
        stats = {"episodes": done.sum(), "return": (ep_return * done).sum(), "length": (ep_length * done).sum()}
        transition = Transition(obs, action, reward, info["terminated"], done, jax.nn.softmax(y_next), value_next)
        return (env_state, next_obs, ep_return * (1 - done), ep_length * (1 - done)), (transition, stats)

    carry = (agent.env_state, agent.obs, agent.ep_return, agent.ep_length)
    (env_state, obs, ep_return, ep_length), (transitions, stats) = jax.lax.scan(step, carry, jax.random.split(key, cfg.num_steps))
    return transitions, jax.tree.map(jnp.sum, stats), agent._replace(env_state=env_state, obs=obs, ep_return=ep_return, ep_length=ep_length)


def lambda_returns(reward, terminated, boundary, value_next, cfg: Config) -> jax.Array:
    """TD(lambda) targets over (T, N) rollouts. A terminal step gets no bootstrap; a truncated one, like the last step
    of the rollout, bootstraps from the value of the true next state."""

    def body(next_return, t):
        reward, terminated, boundary, value_next = t
        target = jnp.where(boundary, value_next, (1 - cfg.gae_lambda) * value_next + cfg.gae_lambda * next_return)
        ret = reward + cfg.gamma * (1 - terminated) * target
        return ret, ret

    _, returns = jax.lax.scan(body, value_next[-1], (reward, terminated, boundary, value_next), reverse=True)
    return returns


def log_prob_of(logits, action):
    return jnp.take_along_axis(jax.nn.log_softmax(logits), action[..., None], axis=-1)[..., 0]


def inner_update(block: Block, cfg: Config, meta_graphdef, meta_params, agent: Agent, key: jax.Array):
    """One agent update by the meta-network, LPG eq. 2; also returns the squared targets for the outer regularisers."""
    traj, stats, agent = rollout(block, cfg, agent, key)
    logits, y_logits, _ = nnx.merge(block.graphdef, agent.params)(traj.obs)
    gamma = jnp.full(traj.reward.shape, cfg.gamma)
    # targets: constants to the agent's gradient below, functions of the meta-parameters to the meta-gradient
    pi_hat, y_hat = nnx.merge(meta_graphdef, meta_params)(traj.reward, traj.terminated, traj.boundary, gamma, jnp.exp(log_prob_of(logits, traj.action)), jax.nn.softmax(y_logits), traj.y_next)
    returns = lambda_returns(traj.reward, traj.terminated, traj.boundary, traj.value_next, cfg)

    def loss_fn(params):
        logits, y_logits, value = nnx.merge(block.graphdef, params)(traj.obs)
        log_y = jax.nn.log_softmax(y_logits)
        kl = (jnp.exp(log_y) * (log_y - jax.nn.log_softmax(y_hat))).sum(-1)
        # the critic's value loss trains it for the outer advantage only; the meta-network never sees values
        return -(log_prob_of(logits, traj.action) * pi_hat).mean() + block.spec.kl_cost * kl.mean() + 0.5 * ((value - returns) ** 2).mean()

    updates, opt_state = block.optimizer.update(jax.grad(loss_fn)(agent.params), agent.opt_state, agent.params)
    agent = agent._replace(params=optax.apply_updates(agent.params, updates), opt_state=opt_state)
    return agent, (stats, (pi_hat**2).mean(), (y_hat**2).mean())


def outer_loss(block: Block, cfg: Config, meta_graphdef, meta_params, agent: Agent, key: jax.Array):
    """LPG eq. 4 for one agent: num_updates updates by the meta-network, then A2C of the updated agent on a fresh rollout."""
    keys = jax.random.split(key, cfg.num_updates + 1)
    update = jax.checkpoint(partial(inner_update, block, cfg, meta_graphdef, meta_params))
    agent, (stats, pi_hat_sq, y_hat_sq) = jax.lax.scan(update, agent, keys[:-1])

    traj, val_stats, agent = rollout(block, cfg, agent, keys[-1])
    logits, y_logits, value = nnx.merge(block.graphdef, agent.params)(traj.obs)
    log_probs, log_y = jax.nn.log_softmax(logits), jax.nn.log_softmax(y_logits)
    advantage = jax.lax.stop_gradient(lambda_returns(traj.reward, traj.terminated, traj.boundary, traj.value_next, cfg) - value)
    # normalised per agent, so that environments with large rewards don't dominate the meta-gradient
    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
    entropy = -(jnp.exp(log_probs) * log_probs).sum(-1).mean()
    y_entropy = -(jnp.exp(log_y) * log_y).sum(-1).mean()
    loss = -(log_prob_of(logits, traj.action) * advantage).mean() - cfg.ent_coef * entropy - cfg.y_ent_coef * y_entropy + cfg.pi_hat_l2 * pi_hat_sq.mean() + cfg.y_hat_l2 * y_hat_sq.mean()

    stats = jax.tree.map(jnp.add, jax.tree.map(jnp.sum, stats), val_stats)
    metrics = {"outer_loss": loss, "entropy": entropy, "y_entropy": y_entropy, "pi_hat_rms": jnp.sqrt(pi_hat_sq.mean()), "y_hat_rms": jnp.sqrt(y_hat_sq.mean())}
    return loss, (agent, stats, metrics)


def end_of_lifetime(block: Block, cfg: Config, agent: Agent, entropy, key: jax.Array) -> tuple[Agent, jax.Array]:
    """Ages the agent by the meta step and starts a new lifetime when its own is over or its policy collapsed (LPG's guard against early divergence)."""
    age = agent.age + (cfg.num_updates + 1) * cfg.num_steps * cfg.num_envs
    reset = (age >= block.spec.lifetime) | (entropy < 1e-3)
    fresh = init_agent(block, cfg, key)._replace(age=jnp.int32(0))
    return jax.tree.map(lambda f, a: jnp.where(reset, f, a), fresh, agent._replace(age=age)), reset


def train(cfg: Config):
    run_name = cfg.run_name or f"disco_s{cfg.seed}_{time.strftime('%Y%m%d-%H%M%S')}"
    blocks = [make_block(name, cfg) for name in cfg.envs]
    meta_graphdef, meta_params = nnx.split(MetaNetwork(cfg.y_dim, cfg.meta_hidden, nnx.Rngs(cfg.seed)))
    meta_tx = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(cfg.meta_lr))
    meta_opt_state = meta_tx.init(meta_params)
    key = jax.random.key(cfg.seed)
    agents = {block.name: jax.vmap(partial(init_agent, block, cfg))(jax.random.split(jax.random.fold_in(key, i), cfg.agents_per_env)) for i, block in enumerate(blocks)}

    @partial(jax.jit, donate_argnums=(0, 1, 2))
    def meta_step(meta_params, meta_opt_state, agents, key):
        grads, new_agents, stats, metrics = [], {}, {}, {}
        for i, block in enumerate(blocks):
            key_outer, key_reset = jax.random.split(jax.random.fold_in(key, i))
            grad_fn = jax.vmap(jax.grad(partial(outer_loss, block, cfg, meta_graphdef), has_aux=True), in_axes=(None, 0, 0))
            grad, (agent, block_stats, block_metrics) = grad_fn(meta_params, agents[block.name], jax.random.split(key_outer, cfg.agents_per_env))
            agent, reset = jax.vmap(partial(end_of_lifetime, block, cfg))(agent, block_metrics["entropy"], jax.random.split(key_reset, cfg.agents_per_env))
            grads.append(jax.tree.map(lambda g: g.mean(0), grad))
            new_agents[block.name] = agent
            stats[block.name] = jax.tree.map(jnp.sum, block_stats) | {"resets": reset.sum()}
            metrics[block.name] = jax.tree.map(jnp.mean, block_metrics)
        grad = jax.tree.map(lambda *g: sum(g) / len(g), *grads)  # every environment weighs the same
        updates, meta_opt_state = meta_tx.update(grad, meta_opt_state, meta_params)
        metrics["meta_grad_norm"] = optax.tree.norm(grad)
        return optax.apply_updates(meta_params, updates), meta_opt_state, new_agents, stats, metrics

    add = jax.jit(lambda a, b: jax.tree.map(jnp.add, a, b))
    logger = Logger(cfg, cfg.wandb_project, run_name, cfg.track)
    steps_per_meta_step = len(blocks) * cfg.agents_per_env * cfg.num_envs * cfg.num_steps * (cfg.num_updates + 1)
    print(f"{run_name}: meta-network {num_params(nnx.merge(meta_graphdef, meta_params)):,} params, {len(blocks)} envs x {cfg.agents_per_env} agents, {steps_per_meta_step:,} env steps per meta step")

    episode_stats, last_time, last_step = None, time.perf_counter(), 0
    for i in range(1, cfg.meta_steps + 1):
        meta_params, meta_opt_state, agents, stats, metrics = meta_step(meta_params, meta_opt_state, agents, jax.random.fold_in(key, i))
        episode_stats = stats if episode_stats is None else add(episode_stats, stats)

        if i % cfg.log_interval == 0 or i == cfg.meta_steps:
            metrics, episode_stats = jax.device_get((metrics, episode_stats))
            now = time.perf_counter()
            log = {"sps": (i - last_step) * steps_per_meta_step / max(1e-6, now - last_time), "meta_grad_norm": metrics["meta_grad_norm"]}
            for block in blocks:
                s = episode_stats[block.name]
                log |= {f"{block.name}/{k}": v for k, v in metrics[block.name].items()}
                log |= {f"{block.name}/return": s["return"] / max(s["episodes"], 1), f"{block.name}/length": s["length"] / max(s["episodes"], 1), f"{block.name}/resets": s["resets"]}
            logger.log(i, log, console=["sps", "meta_grad_norm", *(f"{block.name}/return" for block in blocks)])
            episode_stats, last_time, last_step = None, now, i

        if cfg.save_interval and (i % cfg.save_interval == 0 or i == cfg.meta_steps):
            path = save_checkpoint(f"{cfg.checkpoint_dir}/{run_name}", i, nnx.merge(meta_graphdef, meta_params))
            logger.upload_checkpoint(path, i)

    logger.close()
    return nnx.merge(meta_graphdef, meta_params)


if __name__ == "__main__":
    train(tyro.cli(Config))
