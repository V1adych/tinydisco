"""Meta-trains an update rule (nn.MetaNetwork) after DiscoRL (Oh et al. 2025) on a population of agents playing Atari
games: python src/disco.py [--games breakout pong ...] [--track]

Every meta step each agent first acts, with its current parameters: num_updates training rollouts and one validation
rollout twice as long. Then it learns from the training rollouts, one update each, towards the meta-network's targets;
later updates learn from an older policy's data, which the rule sees through the behaviour policy mu. The meta-objective
is a policy gradient of the updated agent on the validation rollout, with V-trace advantages from a separate critic,
and the meta-gradient flows back through the updates. Agents live on between meta steps until their lifetime ends.
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

from checkpoint import fetch_checkpoint, load_checkpoint, save_checkpoint
from envs import make
from envs.atari import TRAIN_GAMES
from log import Logger
from nn import FeedForward, MetaNetwork, RuleInputs, agent_heads, normalize, num_params, take


@dataclass(frozen=True)
class Config:
    games: tuple[str, ...] = TRAIN_GAMES
    agents_per_game: int = 4
    num_envs: int = 32  # per agent
    num_steps: int = 20  # T of a training rollout; the validation rollout has 2T
    num_updates: int = 2  # agent updates per meta step, all backpropagated through
    lifetime: int = 5_000_000  # env steps of an agent before it starts over
    meta_steps: int = 20_000
    seed: int = 0
    gamma: float = 0.99
    lam: float = 0.95  # trace decay of Retrace and V-trace
    # agent
    d: int = 64  # bins of y and z (DiscoRL: 600)
    lr: float = 3e-4
    max_abs_update: float = 1.0  # cap on each parameter's adam step before the learning rate, as DiscoRL
    q_cost: float = 0.5
    critic_lr: float = 1e-3
    # meta-network
    meta_hidden: int = 256
    meta_lr: float = 1e-4
    max_grad_norm: float = 1.0
    # meta-objective regularisers, as DiscoRL's meta-training colab
    ent_coef: float = 1e-2
    yz_ent_coef: float = 1e-3  # entropy of y and z on the validation rollout
    target_mean_coef: float = 1e-3  # squared mean of each target's logits, against drift
    pi_hat_kl_coef: float = 1e-2  # KL(pi || pi_hat): keeps the policy targets near the current policy
    # logging
    log_interval: int = 10  # in meta steps
    save_interval: int = 1000  # in meta steps, 0 disables checkpoints
    checkpoint_dir: str = "checkpoints"
    resume: str | None = None  # continue from a checkpoint directory or wandb artifact at its step; optimizer state and agents start fresh
    track: bool = False
    wandb_project: str = "meta-rl"
    run_name: str | None = None


class Block(NamedTuple):
    """A game and what all of its agents share."""

    name: str
    env: Any
    num_actions: int
    new_agent: Any  # new_agent(rngs=...) -> FeedForward with the outputs of nn.agent_heads
    new_critic: Any
    graphdef: Any
    critic_graphdef: Any
    agent_tx: optax.GradientTransformation
    critic_tx: optax.GradientTransformation


class Agent(NamedTuple):
    """One learner: its network, critic, optimisers, envs and age in env steps. In a block these carry a leading axis over agents."""

    params: Any
    opt_state: Any
    critic: Any
    critic_opt_state: Any
    env_state: Any
    obs: jax.Array
    age: jax.Array


class Rollout(NamedTuple):
    obs: jax.Array  # (T + 1, N, ...): the last one is where the next rollout starts
    action: jax.Array  # (T, N)
    reward: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    mu: jax.Array  # (T, N, A): the policy that acted
    critic_final: jax.Array  # (T, N): the critic's value of the observation before any reset, the bootstrap of a truncated episode


def make_block(game: str, cfg: Config) -> Block:
    env, _ = make(f"Atari-{game}")
    num_actions = env.num_actions
    new_agent = partial(FeedForward, num_actions + cfg.d + num_actions * cfg.d + num_actions)
    new_critic = partial(FeedForward, 1)
    graphdef, _ = nnx.split(new_agent(rngs=nnx.Rngs(0)))
    critic_graphdef, _ = nnx.split(new_critic(rngs=nnx.Rngs(0)))
    # eps_root keeps the meta-gradient finite through adam's sqrt for parameters whose gradient is exactly zero
    agent_tx = optax.chain(optax.scale_by_adam(eps_root=1e-8), optax.clip(cfg.max_abs_update), optax.scale(-cfg.lr))
    return Block(game, env, num_actions, new_agent, new_critic, graphdef, critic_graphdef, agent_tx, optax.adam(cfg.critic_lr))


def init_agent(block: Block, cfg: Config, key: jax.Array) -> Agent:
    key_agent, key_critic, key_env, key_age = jax.random.split(key, 4)
    _, params = nnx.split(block.new_agent(rngs=nnx.Rngs(key_agent)))
    _, critic = nnx.split(block.new_critic(rngs=nnx.Rngs(key_critic)))
    obs, env_state = jax.vmap(block.env.reset, in_axes=(0, None))(jax.random.split(key_env, cfg.num_envs), None)
    # lifetimes start staggered so the population always spans young and old agents
    age = jax.random.randint(key_age, (), 0, cfg.lifetime)
    return Agent(params, block.agent_tx.init(params), critic, block.critic_tx.init(critic), env_state, obs, age)


def act(block: Block, cfg: Config, agent: Agent, key: jax.Array, length: int) -> tuple[Rollout, dict, Agent]:
    """length steps in every env of one agent, with its current policy: the rollout, episode stats summed over it, the agent with its envs advanced."""
    model = nnx.merge(block.graphdef, agent.params)
    critic = nnx.merge(block.critic_graphdef, agent.critic)

    def step(carry, key):
        env_state, obs = carry
        key_action, key_step = jax.random.split(key)
        logits = agent_heads(model(obs), block.num_actions, cfg.d)[0]
        action = jax.random.categorical(key_action, logits)
        next_obs, env_state, reward, done, info = jax.vmap(block.env.step, in_axes=(0, 0, 0, None))(
            jax.random.split(key_step, cfg.num_envs), env_state, action, None
        )
        # the critic's value also bootstraps the agent's Q targets of truncated episodes, which saves an agent pass
        critic_final = critic(info["final_observation"]).squeeze(-1)
        transition = (obs, action, reward, info["terminated"], done & ~info["terminated"], jax.nn.softmax(logits), critic_final)
        stats = {"episodes": done.sum(), "score": info["score"].sum()}
        return (env_state, next_obs), (transition, stats)

    (env_state, obs), (transitions, stats) = jax.lax.scan(step, (agent.env_state, agent.obs), jax.random.split(key, length))
    rollout = Rollout(jnp.concatenate([transitions[0], obs[None]]), *transitions[1:])
    return rollout, jax.tree.map(jnp.sum, stats), agent._replace(env_state=env_state, obs=obs)


def collect(block: Block, cfg: Config, agent: Agent, key: jax.Array) -> tuple[Rollout, Rollout, dict, Agent]:
    """num_updates training rollouts (stacked) and a validation rollout of 2T, all by the agent's current policy."""
    keys = jax.random.split(key, cfg.num_updates + 1)

    def one(agent, key):
        rollout, stats, agent = act(block, cfg, agent, key, cfg.num_steps)
        return agent, (rollout, stats)

    agent, (trajs, stats) = jax.lax.scan(one, agent, keys[:-1])
    valid, valid_stats, agent = act(block, cfg, agent, keys[-1], 2 * cfg.num_steps)
    return trajs, valid, jax.tree.map(lambda a, b: a.sum() + b, stats, valid_stats), agent


def retrace(q_a, v_next, reward, discount, boundary, c) -> jax.Array:
    """Retrace targets for Q(s_t, a_t) over (T, N): G_t = r_t + discount_t (v_{t+1} + c_{t+1} (G_{t+1} - Q(s_{t+1}, a_{t+1}))),
    with c = lambda min(1, pi/mu) and the trace cut where an episode ends and at the end of the rollout."""

    def body(trace, t):
        q_a, v_next, reward, discount, boundary, c = t
        ret = reward + discount * (v_next + jnp.where(boundary, 0.0, trace))
        return c * (ret - q_a), ret

    _, returns = jax.lax.scan(body, jnp.zeros_like(reward[0]), (q_a, v_next, reward, discount, boundary, c), reverse=True)
    return returns


def vtrace(v, v_next, reward, discount, boundary, rho, lam) -> tuple[jax.Array, jax.Array]:
    """V-trace (Espeholt et al. 2018) over (T, N): value targets and policy-gradient advantages for rho = pi/mu."""
    rho = jnp.minimum(rho, 1.0)
    delta = rho * (reward + discount * v_next - v)

    def body(acc, t):
        delta, discount, boundary, c = t
        acc = delta + discount * c * jnp.where(boundary, 0.0, acc)
        return acc, acc

    _, acc = jax.lax.scan(body, jnp.zeros_like(v[0]), (delta, discount, boundary, lam * rho), reverse=True)
    targets = v + acc
    targets_next = jnp.where(boundary, v_next, jnp.concatenate([targets[1:], v_next[-1:]]))
    return targets, rho * (reward + discount * targets_next - v)


def kl(p_logits, q_logits) -> jax.Array:
    """KL(p || q) over the last axis."""
    log_p = jax.nn.log_softmax(p_logits)
    return (jnp.exp(log_p) * (log_p - jax.nn.log_softmax(q_logits))).sum(-1)


def entropy(logits) -> jax.Array:
    log_p = jax.nn.log_softmax(logits)
    return -(jnp.exp(log_p) * log_p).sum(-1)


def rule_inputs(cfg: Config, out, traj: Rollout) -> tuple[RuleInputs, jax.Array]:
    """The rule's view of a rollout from the agent's outputs on its T + 1 observations, and the agent's Retrace targets for Q."""
    logits, y, z, q = out
    pi = jax.nn.softmax(logits)
    v = (pi * q).sum(-1)
    v_next = jnp.where(traj.truncated, traj.critic_final, v[1:])
    alive = (~traj.terminated).astype(jnp.float32)
    boundary = traj.terminated | traj.truncated
    rho = take(pi[:-1], traj.action) / take(traj.mu, traj.action)
    returns = retrace(take(q[:-1], traj.action), v_next, traj.reward, cfg.gamma * alive, boundary, cfg.lam * jnp.minimum(rho, 1.0))
    inputs = RuleInputs(
        traj.action, traj.reward, alive, boundary, pi, traj.mu, jax.nn.softmax(y), jax.nn.softmax(z), q[:-1], v[:-1], v_next, returns - v[:-1]
    )
    return inputs, returns


def inner_update(block: Block, cfg: Config, meta_graphdef, meta_params, agent: Agent, traj: Rollout):
    """One agent update towards the meta-network's targets, plus one critic update; also returns the meta-objective's
    regularisers on the targets."""
    meta_network = nnx.merge(meta_graphdef, meta_params)

    def loss_fn(params):
        out = agent_heads(nnx.merge(block.graphdef, params)(traj.obs), block.num_actions, cfg.d)
        inputs, returns = rule_inputs(cfg, jax.lax.stop_gradient(out), traj)
        # targets: constants to this gradient, functions of the meta-parameters to the meta-gradient
        pi_hat, y_hat, z_hat = meta_network(inputs)
        logits, y, z, q = jax.tree.map(lambda x: x[:-1], out)
        loss = kl(pi_hat, logits) + kl(y_hat, y) + kl(z_hat, take(z, traj.action)) + cfg.q_cost * (returns - take(q, traj.action)) ** 2
        return loss.mean(), (jax.lax.stop_gradient(logits), pi_hat, y_hat, z_hat)

    grads, (logits, pi_hat, y_hat, z_hat) = jax.grad(loss_fn, has_aux=True)(agent.params)
    updates, opt_state = block.agent_tx.update(grads, agent.opt_state, agent.params)
    params = optax.apply_updates(agent.params, updates)

    # the critic learns the values of the policy that was updated, by V-trace; it never carries the meta-gradient
    rho = take(jax.nn.softmax(logits), traj.action) / take(traj.mu, traj.action)
    alive = (~traj.terminated).astype(jnp.float32)

    def critic_loss_fn(critic_params):
        values = nnx.merge(block.critic_graphdef, critic_params)(traj.obs).squeeze(-1)
        v_next = jnp.where(traj.truncated, traj.critic_final, values[1:])
        targets, _ = vtrace(
            *jax.lax.stop_gradient((values[:-1], v_next)), traj.reward, cfg.gamma * alive, traj.terminated | traj.truncated, rho, cfg.lam
        )
        return 0.5 * ((targets - values[:-1]) ** 2).mean()

    critic_updates, critic_opt_state = block.critic_tx.update(jax.grad(critic_loss_fn)(agent.critic), agent.critic_opt_state, agent.critic)
    agent = agent._replace(
        params=params, opt_state=opt_state, critic=optax.apply_updates(agent.critic, critic_updates), critic_opt_state=critic_opt_state
    )

    reg = cfg.target_mean_coef * (pi_hat.mean() ** 2 + y_hat.mean() ** 2 + z_hat.mean() ** 2) + cfg.pi_hat_kl_coef * kl(logits, pi_hat).mean()
    metrics = {"pi_hat_entropy": entropy(pi_hat).mean(), "rho": jnp.minimum(rho, 1.0).mean()}
    return agent, (reg, metrics)


def outer_loss(block: Block, cfg: Config, meta_graphdef, meta_params, agent: Agent, trajs: Rollout, valid: Rollout):
    """The meta-objective for one agent: num_updates updates by the meta-network, then a V-trace policy gradient of the
    updated agent on the validation rollout, with DiscoRL's regularisers."""
    update = jax.checkpoint(partial(inner_update, block, cfg, meta_graphdef, meta_params))
    agent, (reg, metrics) = jax.lax.scan(update, agent, trajs)

    logits, y, z, _ = jax.tree.map(lambda x: x[:-1], agent_heads(nnx.merge(block.graphdef, agent.params)(valid.obs), block.num_actions, cfg.d))
    values = nnx.merge(block.critic_graphdef, agent.critic)(valid.obs).squeeze(-1)
    v_next = jnp.where(valid.truncated, valid.critic_final, values[1:])
    rho = jax.lax.stop_gradient(take(jax.nn.softmax(logits), valid.action) / take(valid.mu, valid.action))
    alive = (~valid.terminated).astype(jnp.float32)
    _, advantage = vtrace(values[:-1], v_next, valid.reward, cfg.gamma * alive, valid.terminated | valid.truncated, rho, cfg.lam)
    # normalised per agent, so that games with large values don't dominate the meta-gradient
    advantage = jax.lax.stop_gradient(normalize(advantage))
    policy_entropy = entropy(logits).mean()
    yz_entropy = (entropy(y) + entropy(take(z, valid.action))).mean()
    loss = (
        -(take(jax.nn.log_softmax(logits), valid.action) * advantage).mean()
        - cfg.ent_coef * policy_entropy
        - cfg.yz_ent_coef * yz_entropy
        + reg.mean()
    )

    metrics = jax.tree.map(jnp.mean, metrics) | {"meta_loss": loss, "entropy": policy_entropy}
    return loss, (agent, metrics)


def end_of_lifetime(block: Block, cfg: Config, agent: Agent, policy_entropy, key: jax.Array) -> tuple[Agent, jax.Array]:
    """Ages the agent by the meta step and starts a new lifetime when its own is over or its policy collapsed."""
    age = agent.age + (cfg.num_updates + 2) * cfg.num_steps * cfg.num_envs
    reset = (age >= cfg.lifetime) | (policy_entropy < 1e-3)
    fresh = init_agent(block, cfg, key)._replace(age=jnp.int32(0))
    return jax.tree.map(lambda f, a: jnp.where(reset, f, a), fresh, agent._replace(age=age)), reset


def train(cfg: Config):
    run_name = cfg.run_name or f"disco_s{cfg.seed}_{time.strftime('%Y%m%d-%H%M%S')}"
    logger = Logger(cfg, cfg.wandb_project, run_name, cfg.track)  # first, so that wandb records the console of a crash during setup
    blocks = [make_block(game, cfg) for game in cfg.games]
    meta_network, start = MetaNetwork(cfg.d, cfg.meta_hidden, nnx.Rngs(cfg.seed)), 0
    if cfg.resume:
        path, start = fetch_checkpoint(cfg.resume)
        load_checkpoint(path, meta_network)
    meta_graphdef, meta_params = nnx.split(meta_network)
    meta_tx = optax.adam(cfg.meta_lr)
    clip = optax.clip_by_global_norm(cfg.max_grad_norm)
    clip_state = clip.init(meta_params)  # stateless, so one init serves every game
    meta_opt_state = meta_tx.init(meta_params)
    key = jax.random.key(cfg.seed)
    # under jit, so that leaves that are the same array (atari's obs and its state's frame stack) get buffers of their own for donation
    agents = {
        block.name: jax.jit(jax.vmap(partial(init_agent, block, cfg)))(jax.random.split(jax.random.fold_in(key, i), cfg.agents_per_game))
        for i, block in enumerate(blocks)
    }

    @partial(jax.jit, donate_argnums=(0, 1, 2))
    def meta_step(meta_params, meta_opt_state, agents, key):
        grad_sum, new_agents, stats, metrics = jax.tree.map(jnp.zeros_like, meta_params), {}, {}, {}
        for i, block in enumerate(blocks):
            key_act, key_reset = jax.random.split(jax.random.fold_in(key, i))
            trajs, valid, block_stats, agent = jax.vmap(partial(collect, block, cfg))(
                agents[block.name], jax.random.split(key_act, cfg.agents_per_game)
            )
            grad_fn = jax.vmap(jax.grad(partial(outer_loss, block, cfg, meta_graphdef), has_aux=True), in_axes=(None, 0, 0, 0))
            grad, (agent, block_metrics) = grad_fn(meta_params, agent, trajs, valid)
            agent, reset = jax.vmap(partial(end_of_lifetime, block, cfg))(
                agent, block_metrics["entropy"], jax.random.split(key_reset, cfg.agents_per_game)
            )
            grad = jax.tree.map(lambda g: g.mean(0), grad)
            grad_sum = jax.tree.map(jnp.add, grad_sum, clip.update(grad, clip_state)[0])
            new_agents[block.name] = agent
            stats[block.name] = jax.tree.map(jnp.sum, block_stats) | {"resets": reset.sum()}
            metrics[block.name] = jax.tree.map(jnp.mean, block_metrics) | {"meta_grad_norm": optax.tree.norm(grad)}
        grad = jax.tree.map(lambda g: g / len(blocks), grad_sum)  # every game weighs the same
        updates, meta_opt_state = meta_tx.update(grad, meta_opt_state, meta_params)
        metrics["meta_grad_norm"] = optax.tree.norm(grad)
        return optax.apply_updates(meta_params, updates), meta_opt_state, new_agents, stats, metrics

    add = jax.jit(lambda a, b: jax.tree.map(jnp.add, a, b))
    steps_per_meta_step = len(blocks) * cfg.agents_per_game * cfg.num_envs * cfg.num_steps * (cfg.num_updates + 2)
    print(
        f"{run_name}: meta-network {num_params(nnx.merge(meta_graphdef, meta_params)):,} params, {len(blocks)} games x {cfg.agents_per_game} agents, {steps_per_meta_step:,} env steps per meta step"
    )

    episode_stats, last_time, last_step = None, time.perf_counter(), start
    for i in range(start + 1, cfg.meta_steps + 1):
        meta_params, meta_opt_state, agents, stats, metrics = meta_step(meta_params, meta_opt_state, agents, jax.random.fold_in(key, i))
        episode_stats = stats if episode_stats is None else add(episode_stats, stats)

        if i % cfg.log_interval == 0 or i == cfg.meta_steps:
            metrics, episode_stats = jax.device_get((metrics, episode_stats))
            now = time.perf_counter()
            log = {"sps": (i - last_step) * steps_per_meta_step / max(1e-6, now - last_time), "meta_grad_norm": metrics["meta_grad_norm"]}
            for block in blocks:
                s = episode_stats[block.name]
                log |= {f"{block.name}/{k}": v for k, v in metrics[block.name].items()}
                log |= {
                    f"{block.name}/score": s["score"] / max(s["episodes"], 1),
                    f"{block.name}/episodes": s["episodes"],
                    f"{block.name}/resets": s["resets"],
                }
            logger.log(i, log, console=["sps", "meta_grad_norm", *(f"{block.name}/score" for block in blocks)])
            episode_stats, last_time, last_step = None, now, i

        if cfg.save_interval and (i % cfg.save_interval == 0 or i == cfg.meta_steps):
            path = save_checkpoint(f"{cfg.checkpoint_dir}/{run_name}", i, nnx.merge(meta_graphdef, meta_params))
            logger.upload_checkpoint(path, i)

    logger.close()
    return nnx.merge(meta_graphdef, meta_params)


if __name__ == "__main__":
    train(tyro.cli(Config))
