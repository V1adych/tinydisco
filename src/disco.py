"""Meta-trains an update rule (nn.MetaNetwork) after DiscoRL (Oh et al. 2025) on a population of agents playing Atari
games, at nanodisco's scale: python src/disco.py [--games breakout pong ...] [--track]

Every agent update is as in DiscoRL's evaluation: the agent collects num_envs x num_steps new steps into its replay
buffer, then learns towards the meta-network's targets on a batch of trajectories, a replay share of them from the
buffer. A meta step backpropagates through num_updates such updates; the inner gradient's backward pass is a
Hessian-vector product by forward-over-reverse (utils.mixflow_grad), so each update stores only its parameters, targets
and batch. The updated agent then collects validation rollouts and is scored by a policy gradient with V-trace
advantages from a separate critic. Each agent's meta-gradient goes through its own clipped Adam and the steps are
averaged. Agents live on between meta steps until their lifetime, drawn with probability proportional to 1 / lifetime,
ends.

Left out of DiscoRL, as nanodisco does: the target network and its outputs among the rule's inputs, the agent's
next-policy prediction (its action model), the per-lifetime meta-RNN, running statistics for normalisation (batch
statistics instead) and the categorical Q head (a scalar one with a squared loss). An episode truncated by its time
limit (108k frames) ends like a terminated one, without a bootstrap, as in DiscoRL's environments.
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
from utils import entropy, kl, mixflow_grad, retrace, scale_by_adam_sg_denom, vtrace


@dataclass(frozen=True)
class Config:
    games: tuple[str, ...] = TRAIN_GAMES
    agents_per_game: int = 1
    num_envs: int = 8  # per agent
    num_steps: int = 29  # T of a rollout
    num_updates: int = 10  # agent updates per meta step, all backpropagated through
    batch: int = 96  # trajectories per agent update
    replay: float = 0.9  # share of the batch from the replay buffer, the rest from the new rollout
    # shortcut: whole frame stacks are stored (28 KB per step), so the buffer is small; store single frames if it must grow
    buffer_size: int = 10_000  # transitions per agent
    valid_rollouts: int = 6  # rollouts that score the updated agent: 6 x 8 = 48 trajectories, as DiscoRL
    lifetimes: tuple[int, ...] = (1_000_000, 2_500_000, 5_000_000, 10_000_000)  # env steps; DiscoRL's ratios at 1/20 scale
    meta_steps: int = 20_000
    seed: int = 0
    gamma: float = 0.997  # DiscoRL's on Atari
    lam: float = 0.95  # trace decay of Retrace and V-trace
    # agent
    d: int = 64  # bins of y and z (DiscoRL: 600)
    lr: float = 3e-4
    max_abs_update: float = 1.0  # cap on each parameter's adam step before the learning rate, as DiscoRL
    q_cost: float = 0.5
    critic_lr: float = 1e-3
    # meta-network
    meta_hidden: int = 256
    meta_lr: float = 1e-3
    max_grad_norm: float = 1.0  # per agent, before its own Adam
    # meta-objective regularisers, as DiscoRL's meta-training colab
    ent_coef: float = 1e-2
    yz_ent_coef: float = 1e-3  # entropy of y and z on the validation rollouts
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
    inner_grad: Any  # inner_grad(params, targets, batch): the agent's gradient, with a MixFlow-MG backward pass


class Rollout(NamedTuple):
    """T steps of N envs, time-major; the replay buffer stores trajectories in the same layout, along axis 1."""

    obs: jax.Array  # (T + 1, N, ...): the last one is where the next rollout starts
    action: jax.Array  # (T, N)
    reward: jax.Array  # (T, N)
    done: jax.Array  # (T, N): the episode ended at t (terminated or truncated), so obs t + 1 starts another one
    mu: jax.Array  # (T, N, A): the policy that acted


class Buffer(NamedTuple):
    data: Rollout  # with capacity in place of N
    idx: jax.Array  # where the next trajectories go
    size: jax.Array  # trajectories stored


class Agent(NamedTuple):
    """One learner: its network, critic, optimisers, envs, replay buffer, age and lifetime in env steps. In a block these
    carry a leading axis over agents."""

    params: Any
    opt_state: Any
    critic: Any
    critic_opt_state: Any
    env_state: Any
    obs: jax.Array
    buffer: Buffer
    age: jax.Array
    lifetime: jax.Array


def inner_loss(graphdef, num_actions: int, cfg: Config, params, targets, batch: Rollout) -> jax.Array:
    """The agent's loss on a batch: KL to the rule's pi, y and z targets, and a Retrace TD loss for its Q head."""
    pi_hat, y_hat, z_hat, returns = targets
    logits, y, z, q = agent_heads(nnx.merge(graphdef, params)(batch.obs[:-1]), num_actions, cfg.d)
    loss = kl(pi_hat, logits) + kl(y_hat, y) + kl(z_hat, take(z, batch.action)) + cfg.q_cost * (returns - take(q, batch.action)) ** 2
    return loss.mean()


def make_block(game: str, cfg: Config) -> Block:
    env, _ = make(f"Atari-{game}")
    num_actions = env.num_actions
    new_agent = partial(FeedForward, num_actions + cfg.d + num_actions * cfg.d + num_actions)
    new_critic = partial(FeedForward, 1)
    graphdef, _ = nnx.split(new_agent(rngs=nnx.Rngs(0)))
    critic_graphdef, _ = nnx.split(new_critic(rngs=nnx.Rngs(0)))
    agent_tx = optax.chain(scale_by_adam_sg_denom(), optax.clip(cfg.max_abs_update), optax.scale(-cfg.lr))
    inner_grad = mixflow_grad(partial(inner_loss, graphdef, num_actions, cfg))
    return Block(game, env, num_actions, new_agent, new_critic, graphdef, critic_graphdef, agent_tx, optax.adam(cfg.critic_lr), inner_grad)


def init_buffer(block: Block, cfg: Config) -> Buffer:
    n, T = cfg.buffer_size // cfg.num_steps, cfg.num_steps
    space = block.env.observation_space(None)
    data = Rollout(
        obs=jnp.zeros((T + 1, n, *space.shape), space.dtype),
        action=jnp.zeros((T, n), jnp.int32),
        reward=jnp.zeros((T, n)),
        done=jnp.zeros((T, n), bool),
        mu=jnp.zeros((T, n, block.num_actions)),
    )
    return Buffer(data, jnp.int32(0), jnp.int32(0))


def draw_lifetime(cfg: Config, key: jax.Array) -> jax.Array:
    """With probability proportional to 1 / lifetime, so that every lifetime gets the same share of experience."""
    lifetimes = jnp.array(cfg.lifetimes)
    return jax.random.choice(key, lifetimes, p=(1 / lifetimes) / (1 / lifetimes).sum())


def init_agent(block: Block, cfg: Config, key: jax.Array) -> Agent:
    key_agent, key_critic, key_env, key_lifetime, key_age = jax.random.split(key, 5)
    _, params = nnx.split(block.new_agent(rngs=nnx.Rngs(key_agent)))
    _, critic = nnx.split(block.new_critic(rngs=nnx.Rngs(key_critic)))
    obs, env_state = jax.vmap(block.env.reset, in_axes=(0, None))(jax.random.split(key_env, cfg.num_envs), None)
    lifetime = draw_lifetime(cfg, key_lifetime)
    # lifetimes start staggered so the population always spans young and old agents
    age = jax.random.randint(key_age, (), 0, lifetime)
    return Agent(params, block.agent_tx.init(params), critic, block.critic_tx.init(critic), env_state, obs, init_buffer(block, cfg), age, lifetime)


def act(block: Block, cfg: Config, agent: Agent, key: jax.Array) -> tuple[Rollout, dict, Agent]:
    """T steps in every env of one agent, with its current policy: the rollout, episode stats summed over it, the agent with its envs advanced."""
    model = nnx.merge(block.graphdef, jax.lax.stop_gradient(agent.params))

    def step(carry, key):
        env_state, obs = carry
        key_action, key_step = jax.random.split(key)
        logits = agent_heads(model(obs), block.num_actions, cfg.d)[0]
        action = jax.random.categorical(key_action, logits)
        next_obs, env_state, reward, done, info = jax.vmap(block.env.step, in_axes=(0, 0, 0, None))(
            jax.random.split(key_step, cfg.num_envs), env_state, action, None
        )
        stats = {"episodes": done.sum(), "score": info["score"].sum()}
        return (env_state, next_obs), ((obs, action, reward, done, jax.nn.softmax(logits)), stats)

    (env_state, obs), (transitions, stats) = jax.lax.scan(step, (agent.env_state, agent.obs), jax.random.split(key, cfg.num_steps))
    rollout = Rollout(jnp.concatenate([transitions[0], obs[None]]), *transitions[1:])
    return rollout, jax.tree.map(jnp.sum, stats), agent._replace(env_state=env_state, obs=obs)


def store(buffer: Buffer, rollout: Rollout) -> Buffer:
    """The rollout's trajectories overwrite the oldest ones."""
    capacity, n = buffer.data.action.shape[1], rollout.action.shape[1]
    i = (buffer.idx + jnp.arange(n)) % capacity
    data = jax.tree.map(lambda b, r: b.at[:, i].set(r), buffer.data, rollout)
    return Buffer(data, (buffer.idx + n) % capacity, jnp.minimum(capacity, buffer.size + n))


def sample(cfg: Config, buffer: Buffer, key: jax.Array) -> Rollout:
    """A batch of stored trajectories: a replay share of them from anywhere in the buffer, the rest from the num_envs
    stored last, the new rollout."""
    key_old, key_new, key_mix = jax.random.split(key, 3)
    capacity = buffer.data.action.shape[1]
    old = jax.random.randint(key_old, (cfg.batch,), 0, buffer.size)
    new = (buffer.idx - jax.random.randint(key_new, (cfg.batch,), 1, cfg.num_envs + 1)) % capacity
    i = jnp.where(jax.random.bernoulli(key_mix, cfg.replay, (cfg.batch,)), old, new)
    return jax.tree.map(lambda x: x[:, i], buffer.data)


def rule_inputs(cfg: Config, out, traj: Rollout) -> tuple[RuleInputs, jax.Array]:
    """The rule's view of a rollout from the agent's outputs on its T + 1 observations, and the agent's Retrace targets
    for Q. As in DiscoRL, only y and z carry the meta-gradient; the rest is constant."""
    logits, y, z, q = out
    pi, q = jax.lax.stop_gradient((jax.nn.softmax(logits), q))
    v = (pi * q).sum(-1)
    alive = (~traj.done).astype(jnp.float32)
    rho = take(pi[:-1], traj.action) / take(traj.mu, traj.action)
    returns = retrace(take(q[:-1], traj.action), v, traj.reward, cfg.gamma * alive, cfg.lam * jnp.minimum(rho, 1.0))
    inputs = RuleInputs(
        traj.action, traj.reward, alive, pi, traj.mu, jax.nn.softmax(y), jax.nn.softmax(z), q[:-1], v[:-1], v[1:], returns - v[:-1]
    )
    return inputs, returns


def agent_update(block: Block, cfg: Config, meta_graphdef, meta_params, agent: Agent, key: jax.Array):
    """Collects a rollout into the buffer, then, once the buffer holds a batch, one agent update towards the meta-network's
    targets on a sampled batch and one critic update. Also returns the meta-objective's regularisers on the targets."""
    key_act, key_sample = jax.random.split(key)
    rollout, stats, agent = act(block, cfg, agent, key_act)
    buffer = store(agent.buffer, rollout)
    batch = sample(cfg, buffer, key_sample)

    # recomputed in the meta-backward instead of stored: y and z carry the meta-gradient back to the params, through this pass
    forward = jax.checkpoint(lambda params: agent_heads(nnx.merge(block.graphdef, params)(batch.obs), block.num_actions, cfg.d))
    out = forward(agent.params)
    inputs, returns = rule_inputs(cfg, out, batch)
    pi_hat, y_hat, z_hat = nnx.merge(meta_graphdef, meta_params)(inputs)
    grads = block.inner_grad(agent.params, (pi_hat, y_hat, z_hat, returns), batch)
    updates, opt_state = block.agent_tx.update(grads, agent.opt_state, agent.params)
    params = optax.apply_updates(agent.params, updates)

    # the critic learns the values of the policy before the update, by V-trace; it never carries the meta-gradient
    logits = jax.lax.stop_gradient(out[0][:-1])
    rho = take(jax.nn.softmax(logits), batch.action) / take(batch.mu, batch.action)

    def critic_loss_fn(critic_params):
        values = nnx.merge(block.critic_graphdef, critic_params)(batch.obs).squeeze(-1)
        targets, _ = vtrace(jax.lax.stop_gradient(values), batch.reward, cfg.gamma * ~batch.done, rho, cfg.lam)
        return 0.5 * ((targets - values[:-1]) ** 2).mean()

    critic = jax.lax.stop_gradient(agent.critic)
    critic_updates, critic_opt_state = block.critic_tx.update(jax.grad(critic_loss_fn)(critic), agent.critic_opt_state, critic)

    # learning starts once the buffer holds one batch, as in DiscoRL's evaluation
    ready = buffer.size >= cfg.batch
    new = (params, opt_state, optax.apply_updates(critic, critic_updates), critic_opt_state)
    old = (agent.params, agent.opt_state, agent.critic, agent.critic_opt_state)
    params, opt_state, critic, critic_opt_state = jax.tree.map(lambda n, o: jnp.where(ready, n, o), new, old)
    agent = agent._replace(params=params, opt_state=opt_state, critic=critic, critic_opt_state=critic_opt_state, buffer=buffer)

    reg = cfg.target_mean_coef * (pi_hat.mean() ** 2 + y_hat.mean() ** 2 + z_hat.mean() ** 2) + cfg.pi_hat_kl_coef * kl(logits, pi_hat).mean()
    metrics = {"pi_hat_entropy": entropy(pi_hat).mean(), "rho": jnp.minimum(rho, 1.0).mean(), "ready": ready}
    return agent, (ready * reg, metrics, stats)


def outer_loss(block: Block, cfg: Config, meta_graphdef, meta_params, agent: Agent, key: jax.Array):
    """The meta-objective for one agent: num_updates updates by the meta-network, then a policy gradient of the updated
    agent on its own validation rollouts, with V-trace advantages and DiscoRL's regularisers."""
    key_train, key_valid = jax.random.split(key)
    update = partial(agent_update, block, cfg, meta_graphdef, meta_params)
    agent, (reg, metrics, stats) = jax.lax.scan(update, agent, jax.random.split(key_train, cfg.num_updates))

    def one(agent, key):
        rollout, stats, agent = act(block, cfg, agent, key)
        return agent, (rollout, stats)

    agent, (valid, valid_stats) = jax.lax.scan(one, agent, jax.random.split(key_valid, cfg.valid_rollouts))
    # (rollouts, T, N, ...) -> (T, rollouts * N, ...)
    valid = jax.tree.map(lambda x: jnp.moveaxis(x, 0, 1).reshape(x.shape[1], -1, *x.shape[3:]), valid)

    logits, y, z, _ = jax.tree.map(lambda x: x[:-1], agent_heads(nnx.merge(block.graphdef, agent.params)(valid.obs), block.num_actions, cfg.d))
    values = nnx.merge(block.critic_graphdef, agent.critic)(valid.obs).squeeze(-1)
    rho = jax.lax.stop_gradient(take(jax.nn.softmax(logits), valid.action) / take(valid.mu, valid.action))
    _, advantage = vtrace(values, valid.reward, cfg.gamma * ~valid.done, rho, cfg.lam)
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
    stats = jax.tree.map(lambda a, b: a.sum() + b.sum(), stats, valid_stats)
    return loss, (agent, metrics, stats)


def end_of_lifetime(block: Block, cfg: Config, agent: Agent, key: jax.Array) -> tuple[Agent, jax.Array]:
    """Ages the agent by the meta step and starts a new lifetime when its own is over."""
    age = agent.age + (cfg.num_updates + cfg.valid_rollouts) * cfg.num_steps * cfg.num_envs
    reset = age >= agent.lifetime
    # the new agent empties the old buffer rather than allocating one: its stale trajectories are never sampled
    fresh = init_agent(block, cfg, key)._replace(age=jnp.int32(0), buffer=agent.buffer._replace(idx=jnp.int32(0), size=jnp.int32(0)))
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
    # one clipped Adam per agent, whose steps are averaged, so that agents with large meta-gradients don't dominate (DiscoRL)
    meta_tx = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(cfg.meta_lr))
    meta_opt_state = {block.name: jax.vmap(lambda _: meta_tx.init(meta_params))(jnp.arange(cfg.agents_per_game)) for block in blocks}
    key = jax.random.key(cfg.seed)
    # under jit, so that leaves that are the same array (atari's obs and its state's frame stack) get buffers of their own for donation
    agents = {block.name: jax.jit(jax.vmap(partial(init_agent, block, cfg)))(jax.random.split(jax.random.fold_in(key, i), cfg.agents_per_game)) for i, block in enumerate(blocks)}

    @partial(jax.jit, donate_argnums=(0, 1, 2))
    def meta_step(meta_params, meta_opt_state, agents, key):
        update_sum, new_opt_state, new_agents, stats, metrics = jax.tree.map(jnp.zeros_like, meta_params), {}, {}, {}, {}
        for i, block in enumerate(blocks):
            key_meta, key_reset = jax.random.split(jax.random.fold_in(key, i))
            grad_fn = jax.vmap(jax.grad(partial(outer_loss, block, cfg, meta_graphdef), has_aux=True), in_axes=(None, 0, 0))
            grad, (agent, block_metrics, block_stats) = grad_fn(meta_params, agents[block.name], jax.random.split(key_meta, cfg.agents_per_game))
            agent, reset = jax.vmap(partial(end_of_lifetime, block, cfg))(agent, jax.random.split(key_reset, cfg.agents_per_game))
            updates, new_opt_state[block.name] = jax.vmap(meta_tx.update, in_axes=(0, 0, None))(grad, meta_opt_state[block.name], meta_params)
            update_sum = jax.tree.map(lambda s, u: s + u.sum(0), update_sum, updates)
            new_agents[block.name] = agent
            stats[block.name] = jax.tree.map(jnp.sum, block_stats) | {"resets": reset.sum()}
            metrics[block.name] = jax.tree.map(jnp.mean, block_metrics) | {"meta_grad_norm": jax.vmap(optax.tree.norm)(grad).mean()}
        updates = jax.tree.map(lambda u: u / (len(blocks) * cfg.agents_per_game), update_sum)
        return optax.apply_updates(meta_params, updates), new_opt_state, new_agents, stats, metrics

    add = jax.jit(lambda a, b: jax.tree.map(jnp.add, a, b))
    steps_per_meta_step = len(blocks) * cfg.agents_per_game * cfg.num_envs * cfg.num_steps * (cfg.num_updates + cfg.valid_rollouts)
    print(f"{run_name}: meta-network {num_params(nnx.merge(meta_graphdef, meta_params)):,} params, {len(blocks)} games x {cfg.agents_per_game} agents, {steps_per_meta_step:,} env steps per meta step")

    episode_stats, last_time, last_step = None, time.perf_counter(), start
    for i in range(start + 1, cfg.meta_steps + 1):
        meta_params, meta_opt_state, agents, stats, metrics = meta_step(meta_params, meta_opt_state, agents, jax.random.fold_in(key, i))
        episode_stats = stats if episode_stats is None else add(episode_stats, stats)

        if i % cfg.log_interval == 0 or i == cfg.meta_steps:
            metrics, episode_stats = jax.device_get((metrics, episode_stats))
            now = time.perf_counter()
            log = {"sps": (i - last_step) * steps_per_meta_step / max(1e-6, now - last_time)}
            for block in blocks:
                s = episode_stats[block.name]
                log |= {f"{block.name}/{k}": v for k, v in metrics[block.name].items()}
                log |= {f"{block.name}/score": s["score"] / max(s["episodes"], 1), f"{block.name}/episodes": s["episodes"], f"{block.name}/resets": s["resets"]}
            logger.log(i, log, console=["sps", *(f"{block.name}/score" for block in blocks)])
            episode_stats, last_time, last_step = None, now, i

        if cfg.save_interval and (i % cfg.save_interval == 0 or i == cfg.meta_steps):
            path = save_checkpoint(f"{cfg.checkpoint_dir}/{run_name}", i, nnx.merge(meta_graphdef, meta_params))
            logger.upload_checkpoint(path, i)

    logger.close()
    return nnx.merge(meta_graphdef, meta_params)


if __name__ == "__main__":
    train(tyro.cli(Config))
