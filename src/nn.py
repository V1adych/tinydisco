import math
from typing import NamedTuple

import jax
from flax import nnx
from jax import numpy as jnp

def to_float(obs: jax.Array) -> jax.Array:
    assert obs.dtype == jnp.uint8
    return obs / 255


class ActorCriticRNN(nnx.Module):
    """Patch-conv encoder -> GRU -> policy logits, y logits (LPG's predictions, empty for PPO) and value."""

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        num_actions: int,
        embed_dim: int,
        hidden_dim: int,
        patch_size: tuple[int, int],
        rngs: nnx.Rngs,
        y_dim: int = 0,
    ):
        h, w, c = obs_shape
        num_patches = (h // patch_size[0]) * (w // patch_size[1])
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        self.encoder = nnx.Conv(in_features=c, out_features=embed_dim, kernel_size=patch_size, strides=patch_size, padding="VALID", rngs=rngs)
        self.proj = nnx.Linear(num_patches * embed_dim, hidden_dim, rngs=rngs)
        self.rnn = nnx.GRUCell(hidden_dim, hidden_dim, rngs=rngs)
        self.actor = nnx.Linear(hidden_dim, num_actions + y_dim, rngs=rngs)
        self.critic = nnx.Linear(hidden_dim, 1, rngs=rngs)

    def initialize_carry(self, batch_size: int) -> jax.Array:
        return self.rnn.initialize_carry((batch_size, self.hidden_dim), rngs=nnx.Rngs(0))

    def __call__(self, carry: jax.Array, obs: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        x = nnx.relu(self.encoder(to_float(obs)))
        x = nnx.relu(self.proj(x.reshape(*x.shape[:-3], -1)))
        carry, x = self.rnn(carry, x)
        out = self.actor(x)
        return carry, out[..., : self.num_actions], out[..., self.num_actions :], self.critic(x).squeeze(-1)


def space_to_depth(x: jax.Array, k: int) -> jax.Array:
    """Each k x k block of pixels becomes channels: (..., H, W, C) -> (..., H / k, W / k, k * k * C)."""
    *lead, h, w, c = x.shape
    return jnp.moveaxis(x.reshape(*lead, h // k, k, w // k, k, c), -4, -3).reshape(*lead, h // k, w // k, k * k * c)


def projected_gru(cell, carry, projected):
    xi_r, xi_z, xi_n = jnp.split(projected, 3, axis=-1)
    hh_r, hh_z, hh_n = jnp.split(cell.dense_h(carry), 3, axis=-1)
    r = cell.gate_fn(xi_r + hh_r)
    z = cell.gate_fn(xi_z + hh_z)
    n = cell.activation_fn(xi_n + r * hh_n)
    h = (1.0 - z) * n + z * carry
    return h, h


class Recurrent(nnx.Module):
    """Patch layers (per layer: pixel block, features; a linear over each block's pixels, i.e. a stride-block conv as a
    matmul), a dense layer, a GRU, a linear output. With colors, the frame is one channel of color indices, one-hot
    before the first layer. The encoder computes in bf16 with float32 weights; the GRU and the output layer are float32.
    The output starts near zero (weights of stddev 0.01, as DiscoRL's heads): uniform policies and predictions, zero values."""

    def __init__(self, num_outputs: int, obs_shape: tuple[int, ...], convs: tuple[tuple[int, int], ...], hidden: int, rngs: nnx.Rngs, colors: int = 0):
        bf16 = dict(dtype=jnp.bfloat16, param_dtype=jnp.float32, rngs=rngs)
        self.colors, self.blocks, self.layers, self.obs_ndim = colors, tuple(k for k, _ in convs), nnx.List(), len(obs_shape)
        shape = (*obs_shape[:-1], colors or obs_shape[-1])
        for k, features in convs:
            h, w, c = shape
            assert h % k == 0 and w % k == 0, f"a {h}x{w} map does not split into {k}x{k} blocks"
            self.layers.append(nnx.Linear(c * k * k, features, **bf16))
            shape = (h // k, w // k, features)
        self.dense = nnx.Linear(int(math.prod(shape)), hidden, **bf16)
        self.gru = nnx.GRUCell(hidden, hidden, rngs=rngs)
        self.out = nnx.Linear(hidden, num_outputs, kernel_init=nnx.initializers.normal(1e-2), rngs=rngs)

    def encode(self, obs: jax.Array) -> jax.Array:
        """(B, *obs_shape) observations -> (B, hidden) float32 features."""
        x = jax.nn.one_hot(obs.squeeze(-1), self.colors, dtype=jnp.bfloat16) if self.colors else to_float(obs)
        for layer, k in zip(self.layers, self.blocks):
            x = nnx.relu(layer(space_to_depth(x, k)))
        return nnx.relu(self.dense(x.reshape(*x.shape[: x.ndim - self.obs_ndim], -1))).astype(jnp.float32)

    def step(self, carry: jax.Array, obs: jax.Array) -> tuple[jax.Array, jax.Array]:
        """One step of a batch: the carry (B, hidden) after the frames (B, H, W, C), and the output (B, num_outputs)."""
        carry, _ = self.gru(carry, self.encode(obs))
        return carry, self.out(carry)

    def unroll(self, carry: jax.Array, obs: jax.Array, done: jax.Array) -> jax.Array:
        """(T + 1, B, num_outputs) predictions over trajectories, resetting the carry after episode ends."""
        return self.out(self.unroll_features(carry, obs, done))

    def unroll_features(self, carry: jax.Array, obs: jax.Array, done: jax.Array) -> jax.Array:
        """GRU features (T + 1, B, hidden) over trajectories: obs (T + 1, B, ...) from the carry (B, hidden) at the
        first, where done (T, B) at t ends the episode, so the carry restarts from zeros at t + 1."""
        # Batch four timesteps for throughput, retaining encoder recomputation to limit activation memory.
        x = self.gru.dense_i(jax.lax.map(jax.checkpoint(self.encode), obs, batch_size=4))
        reset = jnp.concatenate([jnp.zeros_like(done[:1]), done])

        def body(carry, step):
            x_t, reset_t = step
            carry, _ = projected_gru(self.gru, jnp.where(reset_t[:, None], 0.0, carry), x_t)
            return carry, carry

        _, carries = jax.lax.scan(body, carry, (x, reset))
        return carries


def agent_heads(out: jax.Array, num_actions: int, d: int) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Splits an agent's output (A + D + A*D + A) into policy logits (A), y logits (D), z logits (A, D) and Q (A)."""
    logits, y, z, q = jnp.split(out, [num_actions, num_actions + d, num_actions + d + num_actions * d], -1)
    return logits, y, z.reshape(*z.shape[:-1], num_actions, d), q


class RuleInputs(NamedTuple):
    """What the update rule sees of a rollout: T steps, N envs, A actions, D bins. All of it is constant to the agent's gradient."""

    action: jax.Array  # (T, N)
    reward: jax.Array  # (T, N)
    discount: jax.Array  # (T, N): 0 where the episode ended, so t + 1 starts another one, else 1
    pi: jax.Array  # (T + 1, N, A): the agent's policy
    mu: jax.Array  # (T, N, A): the behaviour policy that acted
    y: jax.Array  # (T + 1, N, D) probabilities
    z: jax.Array  # (T + 1, N, A, D) probabilities
    q: jax.Array  # (T, N, A)
    v: jax.Array  # (T, N): sum_a pi Q
    v_next: jax.Array  # (T, N): v at t + 1
    adv: jax.Array  # (T, N): Retrace return minus v


def symlog(x: jax.Array) -> jax.Array:
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))


def normalize(x: jax.Array) -> jax.Array:
    # shortcut: batch statistics, where DiscoRL keeps running averages; upgrade if small batches make them noisy
    return (x - x.mean()) / (x.std() + 1e-6)


def take(x: jax.Array, action: jax.Array) -> jax.Array:
    """x (..., A, ...) at the taken action (...): the action axis is the one right after action's axes."""
    index = action.reshape(action.shape + (1,) * (x.ndim - action.ndim))
    return jnp.take_along_axis(x, index, axis=action.ndim).squeeze(action.ndim)


class ActionMix(nnx.Module):
    """DiscoRL's per-action 1x1 conv: every action's features next to their mean over actions, so the same weights serve any number of actions."""

    def __init__(self, in_features: int, out_features: int, rngs: nnx.Rngs):
        self.linear = nnx.Linear(2 * in_features, out_features, rngs=rngs)

    def __call__(self, x: jax.Array, shared: jax.Array | None = None) -> jax.Array:
        """x (..., A, C); shared (..., out_features): features of the step,
        the same for every action, added before the activation (a linear share computed once, not per action)."""
        mean = jnp.broadcast_to(x.mean(-2, keepdims=True), x.shape)
        out = self.linear(jnp.concatenate([x, mean], -1))
        return nnx.relu(out if shared is None else out + shared[..., None, :])


class MetaNetwork(nnx.Module):
    """DiscoRL's update rule (Oh et al. 2025), its per-trajectory part: per-step and per-action features of a rollout go
    through a GRU unrolled backwards in time, which emits targets for the agent: pi_hat (A logits), y_hat and z_hat (D logits each).
    DiscoRL used an LSTM, and a second, per-lifetime LSTM; the GRU is the simpler equivalent and the second is left out."""

    def __init__(self, d: int, hidden_dim: int, rngs: nnx.Rngs):
        out_init = nnx.initializers.truncated_normal(0.3)  # DiscoRL's output stddev
        self.hidden_dim = hidden_dim
        # y and z enter through small learned embeddings to one number each, as in DiscoRL
        self.y_net = nnx.Sequential(nnx.Linear(d, 16, rngs=rngs), nnx.relu, nnx.Linear(16, 1, rngs=rngs))
        self.z_net = nnx.Sequential(nnx.Linear(d, 16, rngs=rngs), nnx.relu, nnx.Linear(16, 1, rngs=rngs))
        self.action_net = nnx.Sequential(ActionMix(7, 16, rngs), ActionMix(16, 2, rngs))
        self.gru = nnx.GRUCell(19, hidden_dim, rngs=rngs)
        self.y_hat = nnx.Linear(hidden_dim, d, kernel_init=out_init, rngs=rngs)
        self.z_hat = nnx.Linear(hidden_dim, d, kernel_init=out_init, rngs=rngs)
        self.pi_h = nnx.Linear(hidden_dim, 16, rngs=rngs)  # the GRU state's share of pi_mix's input, the same for every action
        self.pi_mix = ActionMix(2, 16, rngs)
        self.pi_hat = nnx.Linear(16, 1, kernel_init=out_init, rngs=rngs)

    def __call__(self, x: RuleInputs) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Returns pi_hat (T, N, A), y_hat (T, N, D) and z_hat (T, N, D) logits; z_hat is for the taken action."""
        num_actions = x.pi.shape[-1]
        pi = x.pi[:-1]
        y_emb = self.y_net(x.y).squeeze(-1)  # (T + 1, N)
        z_emb = self.z_net(x.z).squeeze(-1)  # (T + 1, N, A)
        z_avg, z_max = (x.pi * z_emb).sum(-1), z_emb.max(-1)
        qv = x.q - x.v[..., None]
        per_action = jnp.stack(
            [
                pi,
                x.mu,
                z_emb[:-1],
                symlog(x.q),
                symlog(qv),
                normalize(qv),
                jax.nn.one_hot(x.action, num_actions),
            ],
            -1,
        )
        action_emb = per_action
        for layer in self.action_net.layers:
            action_emb = layer(action_emb)  # (T, N, A, 2)
        per_step = jnp.stack(
            [
                take(pi, x.action),
                take(x.mu, x.action),
                symlog(x.reward),
                x.discount,
                symlog(x.v),
                symlog(x.v_next),
                symlog(x.adv),
                normalize(x.adv),
                y_emb[:-1],
                y_emb[1:],
                take(z_emb[:-1], x.action),
                z_avg[:-1],
                z_avg[1:],
                z_max[:-1],
                z_max[1:],
            ],
            -1,
        )
        inputs = jnp.concatenate([per_step, action_emb.mean(-2), take(action_emb, x.action)], -1)

        inputs = self.gru.dense_i(inputs)

        def body(h, step):
            x_t, ended = step
            # the carry comes from step t+1, which is another episode when t ended one
            h = jnp.where(ended[:, None], 0.0, h)
            return projected_gru(self.gru, h, x_t)

        # a plain scan: nnx.scan rejects the cell's params when they are gradient tracers of the meta-gradient
        _, h = jax.lax.scan(body, jnp.zeros((x.reward.shape[1], self.hidden_dim)), (inputs, x.discount == 0), reverse=True)
        pi_hat = self.pi_hat(self.pi_mix(action_emb, self.pi_h(h))).squeeze(-1)
        return pi_hat, self.y_hat(h), self.z_hat(h)


def num_params(model: nnx.Module) -> int:
    return sum(x.size for x in jax.tree.leaves(nnx.state(model, nnx.Param)))
