import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import Any, NamedTuple

import jax
import numpy as np
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
    # memory against speed, with the same results: how much is computed at once
    agent_chunk: int = 4  # agents whose meta-gradients are computed at once: memory holds this many backward passes
    time_chunk: int = 10  # time steps of a rollout per pass of a network over it; >= num_steps + 1 is all at once
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


class Nets(NamedTuple):
    """What all agents share: one network for every game, its action axis padded to the most actions any game has."""

    num_actions: int
    obs_space: Any
    new_agent: Any  # new_agent(rngs=...) -> FeedForward with the outputs of nn.agent_heads
    new_critic: Any
    graphdef: Any
    critic_graphdef: Any
    agent_tx: optax.GradientTransformation
    critic_tx: optax.GradientTransformation
    inner_grad: Any  # inner_grad(params, targets, (batch, valid)): the agent's gradient, with a MixFlow-MG backward pass


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
    """One learner: its network, critic, optimisers, its game's legal actions, age and lifetime in env steps. Its envs
    (Envs) and replay buffer (Buffer) are kept apart: the envs' state differs between games, and the buffer must not
    pass through the meta-gradient's map, which would copy it. In train these carry a leading axis over agents."""

    params: Any
    opt_state: Any
    critic: Any
    critic_opt_state: Any
    valid: jax.Array  # (A,)
    age: jax.Array
    lifetime: jax.Array


class Envs(NamedTuple):
    state: Any
    obs: jax.Array


def apply(graphdef, params, obs: jax.Array, cfg: Config) -> jax.Array:
    """A network on time-major (T, N, ...) observations, time_chunk steps at a time (nanodisco's CNN_STEPS). Each chunk is
    recomputed in the backward pass instead of stored, so only one chunk's activations are alive at once."""
    net = jax.checkpoint(lambda params, obs: nnx.merge(graphdef, params)(obs))
    return jax.lax.map(lambda obs: net(params, obs), obs, batch_size=cfg.time_chunk)


def inner_loss(graphdef, num_actions: int, cfg: Config, params, targets, data: tuple[Rollout, jax.Array]) -> jax.Array:
    """The agent's loss on a batch: KL to the rule's pi, y and z targets, and a Retrace TD loss for its Q head."""
    pi_hat, y_hat, z_hat, returns = targets
    batch, valid = data
    logits, y, z, q = agent_heads(apply(graphdef, params, batch.obs[:-1], cfg), num_actions, cfg.d, valid)
    loss = kl(pi_hat, logits) + kl(y_hat, y) + kl(z_hat, take(z, batch.action)) + cfg.q_cost * (returns - take(q, batch.action)) ** 2
    return loss.mean()


def make_nets(cfg: Config, envs: list) -> Nets:
    num_actions = max(env.num_actions for env in envs)
    obs_space = envs[0].observation_space(None)
    assert all(env.observation_space(None).shape == obs_space.shape for env in envs), "the games must share an observation shape"
    new_agent = partial(FeedForward, num_actions + cfg.d + num_actions * cfg.d + num_actions)
    new_critic = partial(FeedForward, 1)
    graphdef, _ = nnx.split(new_agent(rngs=nnx.Rngs(0)))
    critic_graphdef, _ = nnx.split(new_critic(rngs=nnx.Rngs(0)))
    agent_tx = optax.chain(scale_by_adam_sg_denom(), optax.clip(cfg.max_abs_update), optax.scale(-cfg.lr))
    inner_grad = mixflow_grad(partial(inner_loss, graphdef, num_actions, cfg))
    return Nets(num_actions, obs_space, new_agent, new_critic, graphdef, critic_graphdef, agent_tx, optax.adam(cfg.critic_lr), inner_grad)


def zero_rollouts(nets: Nets, cfg: Config, lead: tuple[int, ...], n: int) -> Rollout:
    """Rollouts of n trajectories with leading axes lead: (*lead, T, n, ...)."""
    T = cfg.num_steps
    return Rollout(
        obs=jnp.zeros((*lead, T + 1, n, *nets.obs_space.shape), nets.obs_space.dtype),
        action=jnp.zeros((*lead, T, n), jnp.int32),
        reward=jnp.zeros((*lead, T, n)),
        done=jnp.zeros((*lead, T, n), bool),
        mu=jnp.zeros((*lead, T, n, nets.num_actions)),
    )


def init_buffer(nets: Nets, cfg: Config) -> Buffer:
    return Buffer(zero_rollouts(nets, cfg, (), cfg.buffer_size // cfg.num_steps), jnp.int32(0), jnp.int32(0))


def draw_lifetime(cfg: Config, key: jax.Array) -> jax.Array:
    """With probability proportional to 1 / lifetime, so that every lifetime gets the same share of experience."""
    lifetimes = jnp.array(cfg.lifetimes)
    return jax.random.choice(key, lifetimes, p=(1 / lifetimes) / (1 / lifetimes).sum())


def init_agent(nets: Nets, cfg: Config, valid: jax.Array, key: jax.Array) -> Agent:
    key_agent, key_critic, key_lifetime, key_age = jax.random.split(key, 4)
    _, params = nnx.split(nets.new_agent(rngs=nnx.Rngs(key_agent)))
    _, critic = nnx.split(nets.new_critic(rngs=nnx.Rngs(key_critic)))
    lifetime = draw_lifetime(cfg, key_lifetime)
    # lifetimes start staggered so the population always spans young and old agents
    age = jax.random.randint(key_age, (), 0, lifetime)
    return Agent(params, nets.agent_tx.init(params), critic, nets.critic_tx.init(critic), valid, age, lifetime)


def init_envs(env, cfg: Config, key: jax.Array) -> Envs:
    obs, state = jax.vmap(env.reset, in_axes=(0, None))(jax.random.split(key, cfg.num_envs), None)
    return Envs(state, obs)


def act(nets: Nets, cfg: Config, env, params, valid: jax.Array, envs: Envs, key: jax.Array) -> tuple[Rollout, dict, Envs]:
    """T steps in every env of one agent: the rollout, episode stats summed over it, and the envs advanced."""
    model = nnx.merge(nets.graphdef, params)

    def step(envs, key):
        key_action, key_step = jax.random.split(key)
        logits = agent_heads(model(envs.obs), nets.num_actions, cfg.d, valid)[0]
        action = jax.random.categorical(key_action, logits)
        next_obs, state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
            jax.random.split(key_step, cfg.num_envs), envs.state, action, None
        )
        stats = {"episodes": done.sum(), "score": info["score"].sum()}
        return Envs(state, next_obs), ((envs.obs, action, reward, done, jax.nn.softmax(logits)), stats)

    envs, (transitions, stats) = jax.lax.scan(step, envs, jax.random.split(key, cfg.num_steps))
    rollout = Rollout(jnp.concatenate([transitions[0], envs.obs[None]]), *transitions[1:])
    return rollout, jax.tree.map(jnp.sum, stats), envs


def collect(nets: Nets, cfg: Config, env, params, valid: jax.Array, envs: Envs, reset: jax.Array, key: jax.Array):
    """All rollouts of one agent's meta step, num_updates for its updates then valid_rollouts, with its parameters at
    the start of the meta step. The envs restart first if the agent did (reset)."""
    key_reset, key = jax.random.split(key)
    envs = jax.tree.map(lambda f, e: jnp.where(reset, f, e), init_envs(env, cfg, key_reset), envs)

    def one(envs, key):
        rollout, stats, envs = act(nets, cfg, env, params, valid, envs, key)
        return envs, (rollout, stats)

    envs, (rollouts, stats) = jax.lax.scan(one, envs, jax.random.split(key, cfg.num_updates + cfg.valid_rollouts))
    return rollouts, jax.tree.map(jnp.sum, stats), envs


def store(buffer: Buffer, rollout: Rollout) -> Buffer:
    """The rollout's trajectories overwrite the oldest ones."""
    capacity, n = buffer.data.action.shape[1], rollout.action.shape[1]
    i = (buffer.idx + jnp.arange(n)) % capacity
    data = jax.tree.map(lambda b, r: b.at[:, i].set(r), buffer.data, rollout)
    return Buffer(data, (buffer.idx + n) % capacity, jnp.minimum(capacity, buffer.size + n))


def sample(cfg: Config, buffer: Buffer, k: jax.Array, key: jax.Array) -> tuple[Rollout, jax.Array]:
    """Update k's batch from a buffer that already holds the meta step's num_updates rollouts: a replay share from the
    trajectories stored up to rollout k, the rest from rollout k; and whether those are at least a batch."""
    key_old, key_new, key_mix = jax.random.split(key, 3)
    capacity = buffer.data.action.shape[1]
    later = (cfg.num_updates - 1 - k) * cfg.num_envs  # trajectories of the rollouts after k, stored last
    # shortcut: when the buffer is full, the oldest `later` trajectories are left out though they were there at update k
    stored = buffer.size - later
    old = buffer.idx - later - 1 - jax.random.randint(key_old, (cfg.batch,), 0, stored)
    new = buffer.idx - later - jax.random.randint(key_new, (cfg.batch,), 1, cfg.num_envs + 1)
    i = jnp.where(jax.random.bernoulli(key_mix, cfg.replay, (cfg.batch,)), old, new) % capacity
    return jax.tree.map(lambda x: x[:, i], buffer.data), stored >= cfg.batch


def rule_inputs(cfg: Config, out, traj: Rollout, valid: jax.Array) -> tuple[RuleInputs, jax.Array]:
    """The rule's view of a rollout from the agent's outputs on its T + 1 observations, and the agent's Retrace targets
    for Q. As in DiscoRL, only y and z carry the meta-gradient; the rest is constant."""
    logits, y, z, q = out
    pi, q = jax.lax.stop_gradient((jax.nn.softmax(logits), q))
    v = (pi * q).sum(-1)
    alive = (~traj.done).astype(jnp.float32)
    rho = take(pi[:-1], traj.action) / take(traj.mu, traj.action)
    returns = retrace(take(q[:-1], traj.action), v, traj.reward, cfg.gamma * alive, cfg.lam * jnp.minimum(rho, 1.0))
    inputs = RuleInputs(
        traj.action, traj.reward, alive, pi, traj.mu, jax.nn.softmax(y), jax.nn.softmax(z), q[:-1], v[:-1], v[1:], returns - v[:-1], valid
    )
    return inputs, returns


def learn(nets: Nets, cfg: Config, meta_graphdef, meta_params, params, opt_state, batch: Rollout, valid: jax.Array):
    """One agent update towards the meta-network's targets on a batch. Also returns the policy before it (constant) and
    the meta-objective's regulariser on the targets."""
    # y and z carry the meta-gradient back to the params, through this pass
    out = agent_heads(apply(nets.graphdef, params, batch.obs, cfg), nets.num_actions, cfg.d, valid)
    inputs, returns = rule_inputs(cfg, out, batch, valid)
    pi_hat, y_hat, z_hat = nnx.merge(meta_graphdef, meta_params)(inputs)
    grads = nets.inner_grad(params, (pi_hat, y_hat, z_hat, returns), (batch, valid))
    updates, opt_state = nets.agent_tx.update(grads, opt_state, params)

    logits = jax.lax.stop_gradient(out[0][:-1])
    pi_hat_mean = pi_hat.mean(where=jnp.broadcast_to(valid, pi_hat.shape))  # the illegal actions' logits are a constant
    reg = cfg.target_mean_coef * (pi_hat_mean**2 + y_hat.mean() ** 2 + z_hat.mean() ** 2) + cfg.pi_hat_kl_coef * kl(logits, pi_hat).mean()
    return optax.apply_updates(params, updates), opt_state, logits, reg, entropy(pi_hat).mean()


def agent_update(nets: Nets, cfg: Config, meta_graphdef, meta_params, buffer: Buffer, agent: Agent, xs: tuple[jax.Array, jax.Array]):
    """Update k: once the buffer holds a batch, one agent update on a sampled batch and one critic update."""
    k, key = xs
    batch, ready = sample(cfg, buffer, k, key)
    params, opt_state, logits, reg, pi_hat_entropy = jax.checkpoint(partial(learn, nets, cfg, meta_graphdef))(
        meta_params, agent.params, agent.opt_state, batch, agent.valid
    )

    # the critic learns the values of the policy before the update, by V-trace; it never carries the meta-gradient
    rho = take(jax.nn.softmax(logits), batch.action) / take(batch.mu, batch.action)

    def critic_loss_fn(critic_params):
        values = apply(nets.critic_graphdef, critic_params, batch.obs, cfg).squeeze(-1)
        targets, _ = vtrace(jax.lax.stop_gradient(values), batch.reward, cfg.gamma * ~batch.done, rho, cfg.lam)
        return 0.5 * ((targets - values[:-1]) ** 2).mean()

    critic = jax.lax.stop_gradient(agent.critic)
    critic_updates, critic_opt_state = nets.critic_tx.update(jax.grad(critic_loss_fn)(critic), agent.critic_opt_state, critic)

    # learning starts once the buffer holds one batch, as in DiscoRL's evaluation
    new = (params, opt_state, optax.apply_updates(critic, critic_updates), critic_opt_state)
    old = (agent.params, agent.opt_state, agent.critic, agent.critic_opt_state)
    params, opt_state, critic, critic_opt_state = jax.tree.map(lambda n, o: jnp.where(ready, n, o), new, old)
    agent = agent._replace(params=params, opt_state=opt_state, critic=critic, critic_opt_state=critic_opt_state)
    metrics = {"pi_hat_entropy": pi_hat_entropy, "rho": jnp.minimum(rho, 1.0).mean(), "ready": ready}
    return agent, (ready * reg, metrics)


def outer_loss(nets: Nets, cfg: Config, meta_graphdef, meta_params, agent: Agent, buffer: Buffer, rollouts: Rollout, key: jax.Array):
    """The meta-objective for one agent: num_updates updates by the meta-network on its first rollouts (stored in the
    buffer already), then a policy gradient of the updated agent on the rest, with V-trace advantages and DiscoRL's
    regularisers."""
    update = partial(agent_update, nets, cfg, meta_graphdef, meta_params, buffer)
    agent, (reg, metrics) = jax.lax.scan(update, agent, (jnp.arange(cfg.num_updates), jax.random.split(key, cfg.num_updates)))

    # (rollouts, T, N, ...) -> (T, rollouts * N, ...)
    valid = jax.tree.map(lambda x: jnp.moveaxis(x[cfg.num_updates :], 0, 1).reshape(x.shape[1], -1, *x.shape[3:]), rollouts)
    out = agent_heads(apply(nets.graphdef, agent.params, valid.obs, cfg), nets.num_actions, cfg.d, agent.valid)
    logits, y, z, _ = jax.tree.map(lambda x: x[:-1], out)
    values = apply(nets.critic_graphdef, agent.critic, valid.obs, cfg).squeeze(-1)
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
    return loss, (agent, metrics)


def end_of_lifetime(nets: Nets, cfg: Config, agent: Agent, key: jax.Array) -> tuple[Agent, jax.Array]:
    """Ages the agent by the meta step and starts a new lifetime when its own is over; its envs restart at the next
    collect, and the caller empties its buffer."""
    age = agent.age + (cfg.num_updates + cfg.valid_rollouts) * cfg.num_steps * cfg.num_envs
    reset = age >= agent.lifetime
    fresh = init_agent(nets, cfg, agent.valid, key)._replace(age=jnp.int32(0))
    return jax.tree.map(lambda f, a: jnp.where(reset, f, a), fresh, agent._replace(age=age)), reset


def train(cfg: Config):
    run_name = cfg.run_name or f"disco_s{cfg.seed}_{time.strftime('%Y%m%d-%H%M%S')}"
    logger = Logger(cfg, cfg.wandb_project, run_name, cfg.track)  # first, so that wandb records the console of a crash during setup
    games = [make(f"Atari-{game}")[0] for game in cfg.games]
    nets = make_nets(cfg, games)
    num_agents, per_game = len(games) * cfg.agents_per_game, cfg.agents_per_game
    assert cfg.buffer_size // cfg.num_steps >= cfg.num_updates * cfg.num_envs, "a meta step's rollouts must fit in the buffer at once"
    meta_network, start = MetaNetwork(cfg.d, cfg.meta_hidden, nnx.Rngs(cfg.seed)), 0
    if cfg.resume:
        path, start = fetch_checkpoint(cfg.resume)
        load_checkpoint(path, meta_network)
    meta_graphdef, meta_params = nnx.split(meta_network)
    # one clipped Adam per agent, whose steps are averaged, so that agents with large meta-gradients don't dominate (DiscoRL)
    meta_tx = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(cfg.meta_lr))
    meta_opt_state = jax.vmap(lambda _: meta_tx.init(meta_params))(jnp.arange(num_agents))
    key = jax.random.key(cfg.seed)
    # agent k plays game k // agents_per_game
    valid = jnp.repeat(jnp.stack([jnp.arange(nets.num_actions) < game.num_actions for game in games]), per_game, 0)
    agents = jax.jit(jax.vmap(partial(init_agent, nets, cfg)))(valid, jax.random.split(key, num_agents))
    buffers = jax.jit(jax.vmap(lambda _: init_buffer(nets, cfg)))(jnp.arange(num_agents))
    # under jit, so that leaves that are the same array (atari's obs and its state's frame stack) get buffers of their own for donation
    envs = [jax.jit(jax.vmap(partial(init_envs, game, cfg)))(jax.random.split(jax.random.fold_in(key, g + 1), per_game)) for g, game in enumerate(games)]
    reset = jnp.zeros(num_agents, bool)
    # every game's agents write their rollouts in here, in place, so that the meta step takes them without a copy
    rollouts = jax.jit(partial(zero_rollouts, nets, cfg, (num_agents, cfg.num_updates + cfg.valid_rollouts), cfg.num_envs))()

    @partial(jax.jit, static_argnums=0, donate_argnums=(3, 5))
    def collect_game(g, params, valid, envs, reset, rollouts, key):
        """Game g's agents' rollouts for the meta step written into rollouts, their episode stats and their envs after it."""
        agents_of_game = slice(g * per_game, (g + 1) * per_game)
        params, valid, reset = jax.tree.map(lambda x: x[agents_of_game], (params, valid, reset))
        game_rollouts, stats, envs = jax.vmap(partial(collect, nets, cfg, games[g]))(params, valid, envs, reset, jax.random.split(key, per_game))
        return jax.tree.map(lambda r, x: r.at[agents_of_game].set(x), rollouts, game_rollouts), jax.tree.map(jnp.sum, stats), envs

    @partial(jax.jit, donate_argnums=(0, 1, 2, 3))
    def meta_step(meta_params, meta_opt_state, agents, buffers, rollouts, key):
        # the training rollouts go into the buffers first, one rollout at a time, in place
        put = lambda buffers, k: (jax.vmap(store)(buffers, jax.tree.map(lambda x: x[:, k], rollouts)), None)
        buffers = jax.lax.scan(put, buffers, jnp.arange(cfg.num_updates))[0]
        key_meta, key_reset = jax.random.split(key)
        grad_fn = jax.grad(partial(outer_loss, nets, cfg, meta_graphdef), has_aux=True)

        def agent_grad(agent, i, key):
            # the buffers and rollouts are read by index: as inputs of the map, it would copy them into chunks
            return grad_fn(meta_params, agent, *jax.tree.map(lambda x: x[i], (buffers, rollouts)), key)

        xs = (agents, jnp.arange(num_agents), jax.random.split(key_meta, num_agents))
        grad, (agents, metrics) = jax.lax.map(lambda x: agent_grad(*x), xs, batch_size=cfg.agent_chunk)
        agents, reset = jax.vmap(partial(end_of_lifetime, nets, cfg))(agents, jax.random.split(key_reset, num_agents))
        # a new agent empties the old buffer rather than allocating one: its stale trajectories are never sampled
        buffers = buffers._replace(idx=jnp.where(reset, 0, buffers.idx), size=jnp.where(reset, 0, buffers.size))
        updates, meta_opt_state = jax.vmap(meta_tx.update, in_axes=(0, 0, None))(grad, meta_opt_state, meta_params)
        meta_params = optax.apply_updates(meta_params, jax.tree.map(lambda u: u.mean(0), updates))
        return meta_params, meta_opt_state, agents, buffers, reset, metrics | {"meta_grad_norm": jax.vmap(optax.tree.norm)(grad)}

    add = jax.jit(lambda a, b: jax.tree.map(jnp.add, a, b))
    steps_per_meta_step = num_agents * cfg.num_envs * cfg.num_steps * (cfg.num_updates + cfg.valid_rollouts)
    print(
        f"{run_name}: meta-network {num_params(nnx.merge(meta_graphdef, meta_params)):,} params, {len(games)} games x {per_game} agents, {steps_per_meta_step:,} env steps per meta step"
    )

    # compiled ahead, all at once: XLA compiles each program on one thread (tracing stays sequential, Python holds the GIL)
    pool = ThreadPoolExecutor()
    meta_step = pool.submit(meta_step.lower(meta_params, meta_opt_state, agents, buffers, rollouts, key).compile)
    collect_games = [pool.submit(collect_game.lower(g, agents.params, agents.valid, envs[g], reset, rollouts, key).compile) for g in range(len(games))]
    collect_games, meta_step = [c.result() for c in collect_games], meta_step.result()
    pool.shutdown()

    totals, last_time, last_step = None, time.perf_counter(), start
    for i in range(start + 1, cfg.meta_steps + 1):
        key_collect, key_meta = jax.random.split(jax.random.fold_in(key, i))
        stats = []
        for g in range(len(games)):
            rollouts, game_stats, envs[g] = collect_games[g](agents.params, agents.valid, envs[g], reset, rollouts, jax.random.fold_in(key_collect, g))
            stats.append(game_stats)
        meta_params, meta_opt_state, agents, buffers, reset, metrics = meta_step(meta_params, meta_opt_state, agents, buffers, rollouts, key_meta)
        step_totals = (stats, reset.astype(jnp.int32))
        totals = step_totals if totals is None else add(totals, step_totals)

        if i % cfg.log_interval == 0 or i == cfg.meta_steps:
            metrics, (stats, resets) = jax.device_get((metrics, totals))
            now = time.perf_counter()
            log = {"sps": (i - last_step) * steps_per_meta_step / max(1e-6, now - last_time)}
            log["gpu_peak_gb"] = (jax.local_devices()[0].memory_stats() or {}).get("peak_bytes_in_use", 0) / 2**30  # None on a CPU
            for g, name in enumerate(cfg.games):
                agents_of_game = slice(g * per_game, (g + 1) * per_game)
                log |= {f"{name}/{k}": np.mean(v[agents_of_game]) for k, v in metrics.items()}
                s = stats[g]
                log |= {
                    f"{name}/score": s["score"] / max(s["episodes"], 1),
                    f"{name}/episodes": s["episodes"],
                    f"{name}/resets": resets[agents_of_game].sum(),
                }
            logger.log(i, log, console=["sps", "gpu_peak_gb", *(f"{name}/score" for name in cfg.games)])
            totals, last_time, last_step = None, now, i

        if cfg.save_interval and (i % cfg.save_interval == 0 or i == cfg.meta_steps):
            path = save_checkpoint(f"{cfg.checkpoint_dir}/{run_name}", i, nnx.merge(meta_graphdef, meta_params))
            logger.upload_checkpoint(path, i)

    logger.close()
    return nnx.merge(meta_graphdef, meta_params)


if __name__ == "__main__":
    train(tyro.cli(Config))
