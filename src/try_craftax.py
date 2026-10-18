# %%
from dataclasses import dataclass
from functools import partial
from typing import Any, NamedTuple

import jax
from craftax.craftax_env import make_craftax_env_from_name
import optax
from flax import nnx
from jax import numpy as jnp


# %%
@dataclass(frozen=True)
class Config:
    env_name: str = "Craftax-Classic-Pixels-v1"
    num_envs: int = 4
    num_steps: int = 32
    seed: int = 0
    embed_dim: int = 32
    hidden_dim: int = 256
    patch_size: tuple[int, int] = (7, 7)
    # ppo
    num_updates: int = 10
    update_epochs: int = 4
    num_minibatches: int = 2  # minibatches are split over envs, so it must divide num_envs
    lr: float = 3e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    gamma: float = 0.99
    gae_lambda: float = 0.8
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01


cfg = Config()

# %%
env = make_craftax_env_from_name(cfg.env_name, auto_reset=True)
params = env.default_params
obs_shape = env.observation_space(params).shape
num_actions = env.action_space(params).n
print(obs_shape, num_actions)


# %%
class ActorCriticRNN(nnx.Module):
    def __init__(self, cfg: Config, obs_shape: tuple[int, ...], num_actions: int, rngs: nnx.Rngs):
        h, w, c = obs_shape
        num_patches = (h // cfg.patch_size[0]) * (w // cfg.patch_size[1])
        self.hidden_dim = cfg.hidden_dim
        self.encoder = nnx.Conv(
            in_features=c,
            out_features=cfg.embed_dim,
            kernel_size=cfg.patch_size,
            strides=cfg.patch_size,
            padding="VALID",
            rngs=rngs,
        )
        self.proj = nnx.Linear(num_patches * cfg.embed_dim, cfg.hidden_dim, rngs=rngs)
        self.rnn = nnx.GRUCell(cfg.hidden_dim, cfg.hidden_dim, rngs=rngs)
        self.actor = nnx.Linear(cfg.hidden_dim, num_actions, rngs=rngs)
        self.critic = nnx.Linear(cfg.hidden_dim, 1, rngs=rngs)

    def initialize_carry(self, batch_size: int):
        return self.rnn.initialize_carry((batch_size, self.hidden_dim), rngs=nnx.Rngs(0))

    def __call__(self, carry, obs):
        x = nnx.relu(self.encoder(obs))
        x = nnx.relu(self.proj(x.reshape(*x.shape[:-3], -1)))
        carry, x = self.rnn(carry, x)
        return carry, self.actor(x), self.critic(x).squeeze(-1)


model = ActorCriticRNN(cfg, obs_shape, num_actions, nnx.Rngs(cfg.seed))
print(f"{sum(x.size for x in jax.tree.leaves(nnx.state(model, nnx.Param))):,} params")
# %%
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


def step(env, env_params, model, carry: StepCarry, rngs: nnx.Rngs) -> tuple[Transition, StepCarry]:
    rnn_state, logits, value = model(carry.rnn_state, carry.last_obs)
    action = jax.random.categorical(rngs.next_key(), logits)
    log_prob = jnp.take_along_axis(jax.nn.log_softmax(logits), action[..., None], axis=-1).squeeze(-1)
    obs, env_state, reward, done, _ = jax.vmap(env.step, in_axes=(0, 0, 0, None))(jax.random.split(rngs.next_key(), action.shape[0]), carry.env_state, action, env_params)
    rnn_state = jax.tree.map(lambda x: jnp.where(done[:, None], 0.0, x), rnn_state)
    transition = Transition(obs=carry.last_obs, action=action, reward=reward, done=done, log_prob=log_prob, value=value)
    carry = StepCarry(env_state=env_state, last_obs=obs, rnn_state=rnn_state)
    return transition, carry


def collect_rollouts(env, env_params, model, carry: StepCarry, cfg: Config, rngs: nnx.Rngs) -> tuple[Transition, StepCarry]:
    """Returns transitions with leaves of shape (num_steps, num_envs, ...) and the carry to start the next rollout from."""
    rollout = nnx.scan(partial(step, env, env_params), in_axes=(None, nnx.Carry, 0), out_axes=(0, nnx.Carry), length=cfg.num_steps)
    # a broadcast Rngs would hand out the same keys on every step, so give each step its own split
    return rollout(model, carry, rngs.split(cfg.num_steps))


def compute_gae(transitions: Transition, last_value: jax.Array, cfg: Config) -> tuple[jax.Array, jax.Array]:
    """Returns (advantages, value targets), both (num_steps, num_envs)."""
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
    ratio = jnp.exp(log_prob - batch.log_prob)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    policy_loss = -jnp.minimum(ratio * advantages, jnp.clip(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * advantages).mean()
    value_loss = 0.5 * ((value - targets) ** 2).mean()
    entropy = -(jnp.exp(log_probs) * log_probs).sum(-1).mean()
    loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy
    return loss, {"loss": loss, "policy_loss": policy_loss, "value_loss": value_loss, "entropy": entropy}


def train_step(model, optimizer: nnx.Optimizer, transitions: Transition, init_rnn_state: jax.Array, last_carry: StepCarry, cfg: Config, rngs: nnx.Rngs):
    _, _, last_value = model(last_carry.rnn_state, last_carry.last_obs)
    advantages, targets = compute_gae(transitions, last_value, cfg)
    grad_fn = nnx.value_and_grad(ppo_loss, has_aux=True)
    # python loops are unrolled by jit, fine while update_epochs * num_minibatches is small
    for _ in range(cfg.update_epochs):
        env_perm = jax.random.permutation(rngs(), cfg.num_envs)
        for idx in env_perm.reshape(cfg.num_minibatches, -1):
            batch, adv, tgt = jax.tree.map(lambda x: x[:, idx], (transitions, advantages, targets))
            (_, metrics), grads = grad_fn(model, init_rnn_state[idx], batch, adv, tgt, cfg)
            optimizer.update(model, grads)
    return metrics


# %%
def train(cfg: Config):
    rngs = nnx.Rngs(cfg.seed)
    env = make_craftax_env_from_name(cfg.env_name, auto_reset=True)
    env_params = env.default_params
    model = ActorCriticRNN(cfg, env.observation_space(env_params).shape, env.action_space(env_params).n, rngs)
    tx = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.adamw(cfg.lr, weight_decay=cfg.weight_decay))
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    collect = nnx.jit(collect_rollouts, static_argnums=(0, 4))
    update = nnx.jit(train_step, static_argnames="cfg", donate_argnums=(0, 1))

    obs, env_state = jax.vmap(env.reset, in_axes=(0, None))(jax.random.split(rngs(), cfg.num_envs), env_params)
    carry = StepCarry(env_state=env_state, last_obs=obs, rnn_state=model.initialize_carry(cfg.num_envs))
    for i in range(cfg.num_updates):
        init_rnn_state = carry.rnn_state
        transitions, carry = collect(env, env_params, model, carry, cfg, rngs)
        metrics = update(model, optimizer, transitions, init_rnn_state, carry, cfg, rngs)
        metrics = {k: f"{float(v):.4f}" for k, v in metrics.items()}
        print(f"update {i}: reward/step={float(transitions.reward.mean()):.4f} episodes={int(transitions.done.sum())} {metrics}")
    return model


model = train(cfg)
