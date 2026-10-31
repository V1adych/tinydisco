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
from envs import arc, atari, make
from log import Logger
from nn import MetaNetwork, Recurrent, RuleInputs, agent_heads, normalize, num_params, take
from utils import entropy, kl, mixflow_grad, retrace, scale_by_adam_sg_denom, vtrace


class Spec(NamedTuple):
    """A game's agents: the patch encoder (per layer: pixel block, features; see nn.Recurrent), the dense and GRU
    width, the adam learning rate, the lifetimes in env steps they draw from (with probability proportional to
    1 / lifetime, so that every lifetime gets the same share of experience), and colors when the frame is color indices."""

    convs: tuple[tuple[int, int], ...]
    hidden: int
    lr: float
    lifetimes: tuple[int, ...]
    colors: int = 0


# patch embeddings as nanodisco's frame stem: 4x4 pixel patches, then 3x3 of those (84 -> 21 -> 7); lifetimes: DiscoRL's ratios at 1/20 scale
ATARI = Spec(((4, 32), (3, 64)), 256, lr=3e-4, lifetimes=(5_000_000, 12_500_000, 10_000_000, 20_000_000))
# 64x64 color frames: 4x4 patches of colors, then 2x2 of those (64 -> 16 -> 8)
ARC = Spec(((4, 32), (2, 64)), 256, lr=3e-4, lifetimes=ATARI.lifetimes, colors=arc.NUM_COLORS)
# LPG's grid worlds and delayed chains: no patch layers (the grids don't split into patches), a small network so they
# cost little, and lifetimes short enough that agents are still learning (a chain episode is 5 to 50 steps)
# shortcut: lifetimes are a guess, check when a fresh agent's score stops rising
TOY = Spec((), 32, lr=3e-4, lifetimes=(1_000_000, 2_500_000, 5_000_000, 10_000_000))
TOY_TRAIN = ("Grid-dense", "Grid-long_horizon", "Grid-small_sparse", "Chain-long_noisy", "Chain-distraction")
SPECS = {f"Atari-{game}": ATARI for game in atari.TRAIN} | {f"Arc-{game}": ARC for game in arc.TRAIN} | dict.fromkeys(TOY_TRAIN, TOY)


@dataclass(frozen=True)
class Config:
    games: tuple[str, ...] = tuple(SPECS)  # envs.make names, each with a Spec in SPECS
    agents_per_game: int = 8
    num_envs: int = 8  # per agent
    num_steps: int = 29  # T of a rollout
    num_updates: int = 10  # agent updates per meta step, all backpropagated through
    batch: int = 96  # trajectories per agent update
    replay: float = 0.9  # share of the batch from the replay buffer, the rest from the new rollout
    buffer_size: int = 10_000  # transitions per agent
    valid_rollouts: int = 6  # rollouts that score the updated agent: 6 x 8 = 48 trajectories, as DiscoRL
    meta_steps: int = 20_000
    seed: int = 0
    gamma: float = 0.997  # DiscoRL's on Atari
    lam: float = 0.95  # trace decay of Retrace and V-trace
    # agent
    d: int = 64  # bins of y and z (DiscoRL: 600)
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
    """What a game's agents share: the networks of its Spec, their action axis padded to the most actions any game has."""

    num_actions: int
    obs_space: Any
    hidden: int  # the GRU width: the agent's carry is (hidden,) per env
    lifetimes: jax.Array
    new_agent: Any  # new_agent(rngs=...) -> nn.Recurrent with the outputs of nn.agent_heads
    new_critic: Any  # linear value readout on detached actor GRU features
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
    carry: jax.Array  # (1, N, hidden): the agent's GRU carry before obs 0, so that learning unrolls from where acting was (R2D2's stored state)


class Buffer(NamedTuple):
    data: Rollout  # with capacity in place of N
    idx: jax.Array  # where the next trajectories go
    size: jax.Array  # trajectories stored


class Agent(NamedTuple):
    """One learner: its network, critic, optimisers, its game's legal actions, age and lifetime in env steps. Its envs
    (Envs) and replay buffer (Buffer) travel beside it. In train these carry a leading axis over a game's agents."""

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
    carry: jax.Array  # (N, hidden): the agent's GRU carry before obs


def inner_loss(graphdef, num_actions: int, cfg: Config, params, targets, data: tuple[Rollout, jax.Array]) -> jax.Array:
    """The agent's loss on a batch: KL to the rule's pi, y and z targets, and a Retrace TD loss for its Q head."""
    pi_hat, y_hat, z_hat, returns = targets
    batch, valid = data
    out = nnx.merge(graphdef, params).unroll(batch.carry[0], batch.obs, batch.done)[:-1]
    logits, y, z, q = agent_heads(out, num_actions, cfg.d, valid)
    loss = kl(pi_hat, logits) + kl(y_hat, y) + kl(z_hat, take(z, batch.action)) + cfg.q_cost * (returns - take(q, batch.action)) ** 2
    return loss.mean()


def make_nets(cfg: Config, env, spec: Spec, num_actions: int) -> Nets:
    obs_space = env.observation_space(None)
    new_agent = partial(
        Recurrent, num_actions + cfg.d + num_actions * cfg.d + num_actions, obs_space.shape, spec.convs, spec.hidden, colors=spec.colors
    )
    new_critic = partial(nnx.Linear, spec.hidden, 1, kernel_init=nnx.initializers.normal(1e-2))
    graphdef, _ = nnx.split(new_agent(rngs=nnx.Rngs(0)))
    critic_graphdef, _ = nnx.split(new_critic(rngs=nnx.Rngs(0)))
    agent_tx = optax.chain(scale_by_adam_sg_denom(), optax.clip(cfg.max_abs_update), optax.scale(-spec.lr))
    inner_grad = mixflow_grad(partial(inner_loss, graphdef, num_actions, cfg))
    return Nets(
        num_actions,
        obs_space,
        spec.hidden,
        jnp.array(spec.lifetimes),
        new_agent,
        new_critic,
        graphdef,
        critic_graphdef,
        agent_tx,
        optax.adam(cfg.critic_lr),
        inner_grad,
    )


def init_buffer(nets: Nets, cfg: Config) -> Buffer:
    T, n = cfg.num_steps, cfg.buffer_size // cfg.num_steps
    data = Rollout(
        obs=jnp.zeros((T + 1, n, *nets.obs_space.shape), nets.obs_space.dtype),
        action=jnp.zeros((T, n), jnp.int32),
        reward=jnp.zeros((T, n)),
        done=jnp.zeros((T, n), bool),
        mu=jnp.zeros((T, n, nets.num_actions)),
        carry=jnp.zeros((1, n, nets.hidden)),
    )
    return Buffer(data, jnp.int32(0), jnp.int32(0))


def init_agent(nets: Nets, valid: jax.Array, key: jax.Array) -> Agent:
    key_agent, key_critic, key_lifetime, key_age = jax.random.split(key, 4)
    _, params = nnx.split(nets.new_agent(rngs=nnx.Rngs(key_agent)))
    _, critic = nnx.split(nets.new_critic(rngs=nnx.Rngs(key_critic)))
    lifetime = jax.random.choice(key_lifetime, nets.lifetimes, p=(1 / nets.lifetimes) / (1 / nets.lifetimes).sum())
    # lifetimes start staggered so the population always spans young and old agents
    age = jax.random.randint(key_age, (), 0, lifetime)
    return Agent(params, nets.agent_tx.init(params), critic, nets.critic_tx.init(critic), valid, age, lifetime)


def init_envs(nets: Nets, env, cfg: Config, key: jax.Array) -> Envs:
    obs, state = jax.vmap(env.reset, in_axes=(0, None))(jax.random.split(key, cfg.num_envs), None)
    return Envs(state, obs, jnp.zeros((cfg.num_envs, nets.hidden)))


def act(nets: Nets, cfg: Config, env, params, valid: jax.Array, envs: Envs, key: jax.Array) -> tuple[Rollout, dict, Envs]:
    """T steps in every env of one agent: the rollout, episode stats summed over it, and the envs advanced."""
    model = nnx.merge(nets.graphdef, params)

    def step(envs, key):
        key_action, key_step = jax.random.split(key)
        carry, out = model.step(envs.carry, envs.obs)
        logits = agent_heads(out, nets.num_actions, cfg.d, valid)[0]
        action = jax.random.categorical(key_action, logits)
        next_obs, state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
            jax.random.split(key_step, cfg.num_envs), envs.state, action, None
        )
        stats = {"episodes": done.sum(), "score": (info["score"] * done).sum()}  # an env's score counts on the step its episode ends
        carry = jnp.where(done[:, None], 0.0, carry)  # the next obs starts an episode
        return Envs(state, next_obs, carry), ((envs.obs, action, reward, done, jax.nn.softmax(logits)), stats)

    first = envs.carry[None]
    envs, (transitions, stats) = jax.lax.scan(step, envs, jax.random.split(key, cfg.num_steps))
    rollout = Rollout(jnp.concatenate([transitions[0], envs.obs[None]]), *transitions[1:], first)
    return rollout, jax.tree.map(jnp.sum, stats), envs


def store(buffer: Buffer, rollout: Rollout) -> Buffer:
    """The rollout's trajectories overwrite the oldest ones."""
    capacity, n = buffer.data.action.shape[1], rollout.action.shape[1]
    i = (buffer.idx + jnp.arange(n)) % capacity
    data = jax.tree.map(lambda b, r: b.at[:, i].set(r), buffer.data, rollout)
    return Buffer(data, (buffer.idx + n) % capacity, jnp.minimum(capacity, buffer.size + n))


def sample(cfg: Config, buffer: Buffer, key: jax.Array) -> tuple[Rollout, jax.Array]:
    """A batch: a replay share from the whole buffer, the rest from the rollout stored last; and whether the buffer
    holds a batch at all."""
    key_old, key_new, key_mix = jax.random.split(key, 3)
    capacity = buffer.data.action.shape[1]
    old = buffer.idx - 1 - jax.random.randint(key_old, (cfg.batch,), 0, buffer.size)
    new = buffer.idx - jax.random.randint(key_new, (cfg.batch,), 1, cfg.num_envs + 1)
    i = jnp.where(jax.random.bernoulli(key_mix, cfg.replay, (cfg.batch,)), old, new) % capacity
    return jax.tree.map(lambda x: x[:, i], buffer.data), buffer.size >= cfg.batch


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


def critic_values(nets: Nets, critic, features: jax.Array) -> jax.Array:
    """A learned readout on already-computed, stopped actor features."""
    return nnx.merge(nets.critic_graphdef, critic)(jax.lax.stop_gradient(features)).squeeze(-1)


def learn(nets: Nets, cfg: Config, meta_graphdef, meta_params, params, opt_state, batch: Rollout, valid: jax.Array):
    """One agent update towards the meta-network's targets on a batch. Also returns the policy before it (constant) and
    the meta-objective's regulariser on the targets, target entropy, and detached pre-update features for the critic."""
    # y and z carry the meta-gradient back to the params, through this pass
    model = nnx.merge(nets.graphdef, params)
    features = model.unroll_features(batch.carry[0], batch.obs, batch.done)
    out = agent_heads(model.out(features), nets.num_actions, cfg.d, valid)
    inputs, returns = rule_inputs(cfg, out, batch, valid)
    pi_hat, y_hat, z_hat = nnx.merge(meta_graphdef, meta_params)(inputs)
    grads = nets.inner_grad(params, (pi_hat, y_hat, z_hat, returns), (batch, valid))
    updates, opt_state = nets.agent_tx.update(grads, opt_state, params)

    logits = jax.lax.stop_gradient(out[0][:-1])
    pi_hat_mean = pi_hat.mean(where=jnp.broadcast_to(valid, pi_hat.shape))  # the illegal actions' logits are a constant
    reg = cfg.target_mean_coef * (pi_hat_mean**2 + y_hat.mean() ** 2 + z_hat.mean() ** 2) + cfg.pi_hat_kl_coef * kl(logits, pi_hat).mean()
    return optax.apply_updates(params, updates), opt_state, logits, reg, entropy(pi_hat).mean(), jax.lax.stop_gradient(features)


def agent_update(nets: Nets, cfg: Config, meta_graphdef, env, meta_params, carry: tuple[Agent, Buffer, Envs], key: jax.Array):
    """One inner update: the agent collects a rollout with its current policy (a constant to the meta-gradient), stores
    it and, once the buffer holds a batch, learns on a sampled batch; the critic learns with it."""
    agent, buffer, envs = carry
    key_act, key_sample = jax.random.split(key)
    rollout, stats, envs = act(nets, cfg, env, jax.lax.stop_gradient(agent.params), agent.valid, envs, key_act)
    buffer = store(buffer, rollout)
    batch, ready = sample(cfg, buffer, key_sample)
    params, opt_state, logits, reg, pi_hat_entropy, features = learn(
        nets, cfg, meta_graphdef, meta_params, agent.params, agent.opt_state, batch, agent.valid
    )

    # the critic learns the values of the policy before the update, by V-trace; it never carries the meta-gradient
    rho = take(jax.nn.softmax(logits), batch.action) / take(batch.mu, batch.action)

    def critic_loss_fn(critic_params):
        values = critic_values(nets, critic_params, features)
        targets, _ = vtrace(jax.lax.stop_gradient(values), batch.reward, cfg.gamma * ~batch.done, rho, cfg.lam)
        return 0.5 * ((targets - values[:-1]) ** 2).mean()

    critic = jax.lax.stop_gradient(agent.critic)
    critic_loss, critic_grad = jax.value_and_grad(critic_loss_fn)(critic)
    critic_updates, critic_opt_state = nets.critic_tx.update(critic_grad, agent.critic_opt_state, critic)

    # learning starts once the buffer holds one batch, as in DiscoRL's evaluation
    new = (params, opt_state, optax.apply_updates(critic, critic_updates), critic_opt_state)
    old = (agent.params, agent.opt_state, agent.critic, agent.critic_opt_state)
    params, opt_state, critic, critic_opt_state = jax.tree.map(lambda n, o: jnp.where(ready, n, o), new, old)
    agent = agent._replace(params=params, opt_state=opt_state, critic=critic, critic_opt_state=critic_opt_state)
    metrics = {"pi_hat_entropy": pi_hat_entropy, "rho": jnp.minimum(rho, 1.0).mean(), "ready": ready, "critic_loss": critic_loss}
    return (agent, buffer, envs), (ready * reg, metrics, stats)


def outer_loss(nets: Nets, cfg: Config, meta_graphdef, env, meta_params, agent: Agent, buffer: Buffer, envs: Envs, key: jax.Array):
    """The meta-objective for one agent: num_updates updates by the meta-network, each on a rollout collected just
    before it, then a policy gradient of the updated agent on the rollouts it collects after them, with V-trace
    advantages and DiscoRL's regularisers."""
    key_train, key_valid = jax.random.split(key)
    update = partial(agent_update, nets, cfg, meta_graphdef, env, meta_params)
    (agent, buffer, envs), (reg, metrics, stats) = jax.lax.scan(update, (agent, buffer, envs), jax.random.split(key_train, cfg.num_updates))

    def one(envs, key):
        rollout, stats, envs = act(nets, cfg, env, jax.lax.stop_gradient(agent.params), agent.valid, envs, key)
        return envs, (rollout, stats)

    envs, (rollouts, valid_stats) = jax.lax.scan(one, envs, jax.random.split(key_valid, cfg.valid_rollouts))
    stats = jax.tree.map(lambda s, v: s.sum(0) + v.sum(0), stats, valid_stats)
    # (rollouts, T, N, ...) -> (T, rollouts * N, ...)
    valid = jax.tree.map(lambda x: jnp.moveaxis(x, 0, 1).reshape(x.shape[1], -1, *x.shape[3:]), rollouts)
    model = nnx.merge(nets.graphdef, agent.params)
    features = model.unroll_features(valid.carry[0], valid.obs, valid.done)
    out = agent_heads(model.out(features), nets.num_actions, cfg.d, agent.valid)
    logits, y, z, _ = jax.tree.map(lambda x: x[:-1], out)
    values = critic_values(nets, agent.critic, features)
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
    return loss, (agent, buffer, envs, metrics, stats)


def end_of_lifetime(nets: Nets, cfg: Config, env, agent: Agent, buffer: Buffer, envs: Envs, key: jax.Array):
    """Ages the agent by the meta step and starts a new lifetime when its own is over: a new agent in new envs, with
    the buffer emptied rather than reallocated (its stale trajectories are never sampled)."""
    key_agent, key_envs = jax.random.split(key)
    age = agent.age + (cfg.num_updates + cfg.valid_rollouts) * cfg.num_steps * cfg.num_envs
    reset = age >= agent.lifetime
    fresh = init_agent(nets, agent.valid, key_agent)._replace(age=jnp.int32(0))
    agent = jax.tree.map(lambda f, a: jnp.where(reset, f, a), fresh, agent._replace(age=age))
    buffer = buffer._replace(idx=jnp.where(reset, 0, buffer.idx), size=jnp.where(reset, 0, buffer.size))
    envs = jax.tree.map(lambda f, e: jnp.where(reset, f, e), init_envs(nets, env, cfg, key_envs), envs)
    return agent, buffer, envs, reset


def train(cfg: Config, specs: dict[str, Spec] = SPECS):
    run_name = cfg.run_name or f"disco_s{cfg.seed}_{time.strftime('%Y%m%d-%H%M%S')}"
    logger = Logger(cfg, cfg.wandb_project, run_name, cfg.track)  # first, so that wandb records the console of a crash during setup
    games = [make(game)[0] for game in cfg.games]
    num_actions = max(game.num_actions for game in games)
    nets = [make_nets(cfg, game, specs[name], num_actions) for name, game in zip(cfg.games, games)]
    num_agents, per_game = len(games) * cfg.agents_per_game, cfg.agents_per_game
    meta_network, start = MetaNetwork(cfg.d, cfg.meta_hidden, nnx.Rngs(cfg.seed)), 0
    if cfg.resume:
        path, start = fetch_checkpoint(cfg.resume)
        load_checkpoint(path, meta_network)
    meta_graphdef, meta_params = nnx.split(meta_network)
    # one clipped Adam per agent, whose steps are averaged, so that agents with large meta-gradients don't dominate (DiscoRL)
    meta_tx = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(cfg.meta_lr))
    meta_opt_state = jax.vmap(lambda _: meta_tx.init(meta_params))(jnp.arange(num_agents))
    key = jax.random.key(cfg.seed)

    def init_state():
        """Per game: its agents, their buffers and envs, each with a leading axis over the game's agents."""
        agents, buffers, envs = [], [], []
        for g, game in enumerate(games):
            key_agents, key_envs = jax.random.split(jax.random.fold_in(key, g))
            valid = jnp.broadcast_to(jnp.arange(num_actions) < game.num_actions, (per_game, num_actions))
            agents.append(jax.jit(jax.vmap(partial(init_agent, nets[g])))(valid, jax.random.split(key_agents, per_game)))
            buffers.append(jax.jit(jax.vmap(lambda _: init_buffer(nets[g], cfg)))(jnp.arange(per_game)))
            # under jit, so that leaves that are the same array (atari's obs and its state's frame stack) get buffers of their own for donation
            envs.append(jax.jit(jax.vmap(partial(init_envs, nets[g], game, cfg)))(jax.random.split(key_envs, per_game)))
        return agents, buffers, envs

    def game_step(g: int):
        """The game's program: its agents' meta-gradients (outer_loss) and inner updates, which of them started a new
        lifetime, their metrics and the game's episode stats. One program per game, so that the games run one after
        another and share the scratch memory: in one program XLA interleaves them and their temporaries coexist."""
        grad_fn = jax.grad(partial(outer_loss, nets[g], cfg, meta_graphdef, games[g]), has_aux=True)

        def one(meta_params, agent, buffer, envs, key):
            key_loss, key_reset = jax.random.split(key)
            grad, (agent, buffer, envs, metrics, stats) = grad_fn(meta_params, agent, buffer, envs, key_loss)
            agent, buffer, envs, reset = end_of_lifetime(nets[g], cfg, games[g], agent, buffer, envs, key_reset)
            return grad, agent, buffer, envs, reset, metrics | {"meta_grad_norm": optax.tree.norm(grad)}, stats

        # the game's agents together: their envs step in one kernel (collection is launch-bound), their convs are grouped
        def step(meta_params, agents, buffers, envs, key):
            grad, agents, buffers, envs, reset, metrics, stats = jax.vmap(one, in_axes=(None, 0, 0, 0, 0))(
                meta_params, agents, buffers, envs, jax.random.split(key, per_game)
            )
            return grad, agents, buffers, envs, reset, metrics, jax.tree.map(lambda s: s.sum(0), stats)

        return jax.jit(step, donate_argnums=(1, 2, 3))

    @jax.jit
    def meta_update(meta_params, meta_opt_state, grads):
        """Every agent's clipped Adam step, averaged into one update of the meta-network (DiscoRL)."""
        grad = jax.tree.map(lambda *x: jnp.concatenate(x), *grads)
        updates, meta_opt_state = jax.vmap(meta_tx.update, in_axes=(0, 0, None))(grad, meta_opt_state, meta_params)
        return optax.apply_updates(meta_params, jax.tree.map(lambda u: u.mean(0), updates)), meta_opt_state

    steps_per_meta_step = num_agents * cfg.num_envs * cfg.num_steps * (cfg.num_updates + cfg.valid_rollouts)
    print(
        f"{run_name}: meta-network {num_params(nnx.merge(meta_graphdef, meta_params)):,} params, {len(games)} games x {per_game} agents, {steps_per_meta_step:,} env steps per meta step"
    )

    # compiled ahead, on the state's shapes before the state exists (so that cudnn's autotuning has the whole GPU), in
    # threads: XLA releases the GIL while compiling, so the games' programs compile side by side
    shapes = jax.eval_shape(init_state)
    with ThreadPoolExecutor() as pool:
        steps = list(pool.map(lambda g: game_step(g).lower(meta_params, shapes[0][g], shapes[1][g], shapes[2][g], key).compile(), range(len(games))))
    m = [step.memory_analysis() for step in steps]
    print(
        f"meta step memory: inputs {sum(x.argument_size_in_bytes for x in m) / 2**30:.2f} GB, temporary {max(x.temp_size_in_bytes for x in m) / 2**30:.2f} GB (largest game); per game: "
        + " ".join(f"{name} {x.temp_size_in_bytes / 2**30:.1f}" for name, x in zip(cfg.games, m))
    )
    agents, buffers, envs = init_state()

    totals, last_time, last_step = None, time.perf_counter(), start
    for i in range(start + 1, cfg.meta_steps + 1):
        grads, resets, metrics, stats = [], [], [], []
        for g, step in enumerate(steps):
            grad, agents[g], buffers[g], envs[g], reset, game_metrics, game_stats = step(
                meta_params, agents[g], buffers[g], envs[g], jax.random.fold_in(jax.random.fold_in(key, i), g)
            )
            grads.append(grad)
            resets.append(reset)
            metrics.append(game_metrics)
            stats.append(game_stats)
        meta_params, meta_opt_state = meta_update(meta_params, meta_opt_state, grads)
        step_totals = (stats, [r.astype(jnp.int32) for r in resets])
        totals = step_totals if totals is None else jax.tree.map(jnp.add, totals, step_totals)

        if i % cfg.log_interval == 0 or i == cfg.meta_steps:
            metrics, (stats, resets) = jax.device_get((metrics, totals))
            now = time.perf_counter()
            log = {"sps": (i - last_step) * steps_per_meta_step / max(1e-6, now - last_time)}
            log["gpu_peak_gb"] = (jax.local_devices()[0].memory_stats() or {}).get("peak_bytes_in_use", 0) / 2**30  # None on a CPU
            for g, name in enumerate(cfg.games):
                log |= {f"{name}/{k}": np.mean(v) for k, v in metrics[g].items()}
                s = stats[g]
                log |= {
                    f"{name}/score": s["score"] / max(s["episodes"], 1),
                    f"{name}/episodes": s["episodes"],
                    f"{name}/resets": resets[g].sum(),
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
