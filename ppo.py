import time
from dataclasses import dataclass
from functools import partial
from typing import Any, NamedTuple

import jax
import optax
import tyro
from craftax.craftax_env import make_craftax_env_from_name
from flax import nnx
from jax import numpy as jnp

from log import Logger
from nn import ActorCriticRNN, num_params, save_checkpoint


@dataclass(frozen=True)
class Config:
    env_name: str = "Craftax-Classic-Pixels-v1"
    seed: int = 0
    total_timesteps: int = 5_000_000_000
    num_envs: int = 2048
    num_steps: int = 64  # rollout length, also the BPTT horizon
    # network
    embed_dim: int = 32
    hidden_dim: int = 256
    patch_size: tuple[int, int] = (7, 7)
    # ppo
    update_epochs: int = 4
    num_minibatches: int = 8  # minibatches are split over envs, so it must divide num_envs
    lr: float = 2e-4
    anneal_lr: bool = True
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    gamma: float = 0.99
    gae_lambda: float = 0.8
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    # logging
    log_interval: int = 100  # in updates
    save_interval: int = 10_000  # in updates, 0 disables checkpoints
    checkpoint_dir: str = "checkpoints"
    track: bool = False
    wandb_project: str = "meta-rl"
    run_name: str | None = None

    @property
    def num_updates(self) -> int:
        return self.total_timesteps // (self.num_envs * self.num_steps)


class Transition(NamedTuple):
    obs: jax.Array
    action: jax.Array
    reward: jax.Array
    done: jax.Array
    log_prob: jax.Array
    value: jax.Array


class StepCarry(NamedTuple):
    env_state: Any
    last_obs: jax.Array
    rnn_state: jax.Array
    ep_return: jax.Array
    ep_length: jax.Array


def step(env, env_params, model, carry: StepCarry, rngs: nnx.Rngs) -> tuple[tuple[Transition, dict], StepCarry]:
    rnn_state, logits, value = model(carry.rnn_state, carry.last_obs)
    action = jax.random.categorical(rngs.next_key(), logits)
    log_prob = jnp.take_along_axis(jax.nn.log_softmax(logits), action[..., None], axis=-1).squeeze(-1)
    obs, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(jax.random.split(rngs.next_key(), action.shape[0]), carry.env_state, action, env_params)
    rnn_state = jnp.where(done[:, None], 0.0, rnn_state)
    ep_return, ep_length = carry.ep_return + reward, carry.ep_length + 1
    # craftax fills score and achievements only on the terminal step, so summing over steps and dividing by the episode count gives per-episode means
    stats = {"episodes": done.sum(), "return": (ep_return * done).sum(), "length": (ep_length * done).sum()}
    stats |= {k.lower(): v.sum() for k, v in info.items() if k != "discount"}
    transition = Transition(obs=carry.last_obs, action=action, reward=reward, done=done, log_prob=log_prob, value=value)
    carry = StepCarry(env_state=env_state, last_obs=obs, rnn_state=rnn_state, ep_return=ep_return * (1 - done), ep_length=ep_length * (1 - done))
    return (transition, stats), carry


def collect_rollouts(env, env_params, model, carry: StepCarry, cfg: Config, rngs: nnx.Rngs) -> tuple[Transition, dict, StepCarry]:
    """Returns transitions with leaves of shape (num_steps, num_envs, ...), episode stats summed over the rollout, and the carry to start the next rollout from."""
    rollout = nnx.scan(partial(step, env, env_params), in_axes=(None, nnx.Carry, 0), out_axes=(0, nnx.Carry), length=cfg.num_steps)
    # a broadcast Rngs would hand out the same keys on every step, so give each step its own split
    (transitions, stats), carry = rollout(model, carry, rngs.split(cfg.num_steps))
    return transitions, jax.tree.map(jnp.sum, stats), carry


def compute_gae(transitions: Transition, last_value: jax.Array, cfg: Config) -> tuple[jax.Array, jax.Array]:
    def body(carry, t: Transition):
        gae, next_value = carry
        not_done = 1.0 - t.done
        delta = t.reward + cfg.gamma * next_value * not_done - t.value
        gae = delta + cfg.gamma * cfg.gae_lambda * not_done * gae
        return (gae, t.value), gae

    _, advantages = jax.lax.scan(body, (jnp.zeros_like(last_value), last_value), transitions, reverse=True)
    return advantages, advantages + transitions.value


def unroll(model, rnn_state: jax.Array, obs: jax.Array, done: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Replays (T, B, ...) sequences from the rnn state at the start of the rollout, resetting it exactly as step() does."""

    def body(model, rnn_state, obs_t, done_t):
        rnn_state, logits, value = model(rnn_state, obs_t)
        rnn_state = jnp.where(done_t[:, None], 0.0, rnn_state)
        return rnn_state, (logits, value)

    _, (logits, value) = nnx.scan(body, in_axes=(None, nnx.Carry, 0, 0), out_axes=(nnx.Carry, 0))(model, rnn_state, obs, done)
    return logits, value


def ppo_loss(model, rnn_state, batch: Transition, advantages, targets, cfg: Config):
    logits, value = unroll(model, rnn_state, batch.obs, batch.done)
    log_probs = jax.nn.log_softmax(logits)
    log_prob = jnp.take_along_axis(log_probs, batch.action[..., None], axis=-1).squeeze(-1)
    log_ratio = log_prob - batch.log_prob
    ratio = jnp.exp(log_ratio)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    policy_loss = -jnp.minimum(ratio * advantages, jnp.clip(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * advantages).mean()
    value_loss = 0.5 * ((value - targets) ** 2).mean()
    entropy = -(jnp.exp(log_probs) * log_probs).sum(-1).mean()
    loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy
    metrics = {
        "loss": loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy,
        "approx_kl": ((ratio - 1) - log_ratio).mean(),
        "clip_frac": (jnp.abs(ratio - 1) > cfg.clip_eps).mean(),
    }
    return loss, metrics


def update(env, env_params, model, optimizer: nnx.Optimizer, carry: StepCarry, cfg: Config, rngs: nnx.Rngs) -> tuple[StepCarry, dict, dict]:
    """One PPO iteration: collect a rollout, then update_epochs passes over minibatches of whole env sequences."""
    init_rnn_state = carry.rnn_state
    transitions, stats, carry = collect_rollouts(env, env_params, model, carry, cfg, rngs)
    _, _, last_value = model(carry.rnn_state, carry.last_obs)
    advantages, targets = compute_gae(transitions, last_value, cfg)

    # (update_epochs * num_minibatches, envs_per_minibatch) env indices, a fresh permutation per epoch
    perms = jax.vmap(jax.random.permutation, in_axes=(0, None))(jax.random.split(rngs(), cfg.update_epochs), cfg.num_envs)
    minibatch_idx = perms.reshape(cfg.update_epochs * cfg.num_minibatches, -1)
    grad_fn = nnx.value_and_grad(ppo_loss, has_aux=True)

    def train_minibatch(model_and_opt, idx):
        model, optimizer = model_and_opt
        batch, adv, tgt = jax.tree.map(lambda x: x[:, idx], (transitions, advantages, targets))
        (_, metrics), grads = grad_fn(model, init_rnn_state[idx], batch, adv, tgt, cfg)
        optimizer.update(model, grads)
        return (model, optimizer), metrics

    _, metrics = nnx.scan(train_minibatch, in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))((model, optimizer), minibatch_idx)
    return carry, jax.tree.map(jnp.mean, metrics), stats


def train(cfg: Config):
    assert cfg.num_envs % cfg.num_minibatches == 0, "num_minibatches must divide num_envs"
    run_name = cfg.run_name or f"ppo_{cfg.env_name}_s{cfg.seed}_{time.strftime('%Y%m%d-%H%M%S')}"
    env = make_craftax_env_from_name(cfg.env_name, auto_reset=True)
    env_params = env.default_params

    rngs = nnx.Rngs(cfg.seed)
    model = ActorCriticRNN(env.observation_space(env_params).shape, env.action_space(env_params).n, cfg.embed_dim, cfg.hidden_dim, cfg.patch_size, rngs)
    num_grad_steps = cfg.num_updates * cfg.update_epochs * cfg.num_minibatches
    lr = optax.linear_schedule(cfg.lr, 0.0, num_grad_steps) if cfg.anneal_lr else cfg.lr
    tx = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.adamw(lr, weight_decay=cfg.weight_decay))
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    obs, env_state = jax.vmap(env.reset, in_axes=(0, None))(jax.random.split(rngs(), cfg.num_envs), env_params)
    # separate buffers for ep_return and ep_length, donation rejects the same array passed twice
    carry = StepCarry(env_state=env_state, last_obs=obs, rnn_state=model.initialize_carry(cfg.num_envs), ep_return=jnp.zeros(cfg.num_envs), ep_length=jnp.zeros(cfg.num_envs))

    # jit a pure function of the split state: nnx.jit re-traverses the object graph on every call, which adds up over ~1M updates
    graphdef, state = nnx.split((model, optimizer, rngs))

    @partial(jax.jit, donate_argnums=(0, 1))
    def update_fn(state, carry):
        model, optimizer, rngs = nnx.merge(graphdef, state)
        carry, metrics, stats = update(env, env_params, model, optimizer, carry, cfg, rngs)
        return nnx.state((model, optimizer, rngs)), carry, metrics, stats

    add = jax.jit(lambda a, b: jax.tree.map(jnp.add, a, b))

    logger = Logger(cfg, cfg.wandb_project, run_name, cfg.track)
    steps_per_update = cfg.num_envs * cfg.num_steps
    print(f"{run_name}: {num_params(model):,} params, {cfg.num_updates:,} updates of {steps_per_update:,} steps")

    episode_stats, last_time, last_update = None, time.perf_counter(), 0
    for i in range(1, cfg.num_updates + 1):
        state, carry, metrics, stats = update_fn(state, carry)
        episode_stats = stats if episode_stats is None else add(episode_stats, stats)

        if i % cfg.log_interval == 0 or i == cfg.num_updates:
            metrics, episode_stats = jax.device_get((metrics, episode_stats))
            now = time.perf_counter()
            span = max(1e-6, now - last_time)
            log = {"sps": (i - last_update) * steps_per_update / span, **{f"train/{k}": v for k, v in metrics.items()}}
            n = episode_stats.pop("episodes")
            if n > 0:
                log |= {f"episode/{k}": v / n for k, v in episode_stats.items() if not k.startswith("achievements/")}
                log |= {k: v / n for k, v in episode_stats.items() if k.startswith("achievements/")}
            logger.log(i * steps_per_update, log)
            episode_stats, last_time, last_update = None, now, i

        if cfg.save_interval and (i % cfg.save_interval == 0 or i == cfg.num_updates):
            model, _, _ = nnx.merge(graphdef, state)
            save_checkpoint(f"{cfg.checkpoint_dir}/{run_name}", i * steps_per_update, model)

    logger.close()
    model, _, _ = nnx.merge(graphdef, state)
    return model


if __name__ == "__main__":
    train(tyro.cli(Config))
