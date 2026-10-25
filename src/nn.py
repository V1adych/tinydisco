import math

import jax
from flax import nnx
from jax import numpy as jnp


class ActorCriticRNN(nnx.Module):
    """Patch-conv encoder -> GRU -> policy logits, y logits (LPG's predictions, empty for PPO) and value."""

    def __init__(self, obs_shape: tuple[int, ...], num_actions: int, embed_dim: int, hidden_dim: int, patch_size: tuple[int, int], rngs: nnx.Rngs, y_dim: int = 0):
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
        x = nnx.relu(self.encoder(obs))
        x = nnx.relu(self.proj(x.reshape(*x.shape[:-3], -1)))
        carry, x = self.rnn(carry, x)
        out = self.actor(x)
        return carry, out[..., : self.num_actions], out[..., self.num_actions :], self.critic(x).squeeze(-1)


class FeedForward(nnx.Module):
    """3x3 convs for (H, W, C) observations, then dense layers, then a linear output; flat observations skip the convs.
    channels and hidden follow LPG's table 3, e.g. C(16)-D(32) is channels=(16,), hidden=(32,).
    With neither, on a one-hot observation, this is a lookup table: LPG's tabular agent."""

    def __init__(self, obs_shape: tuple[int, ...], num_outputs: int, channels: tuple[int, ...], hidden: tuple[int, ...], rngs: nnx.Rngs):
        self.obs_ndim = len(obs_shape)
        in_channels = [obs_shape[-1], *channels] if self.obs_ndim == 3 else []
        self.convs = nnx.List([nnx.Conv(c_in, c_out, (3, 3), padding="SAME", rngs=rngs) for c_in, c_out in zip(in_channels, channels)])
        size = math.prod(obs_shape[:-1]) * (channels[-1] if self.convs else obs_shape[-1])
        sizes = [size, *hidden]
        self.dense = nnx.List([nnx.Linear(a, b, rngs=rngs) for a, b in zip(sizes, hidden)])
        self.out = nnx.Linear(sizes[-1], num_outputs, rngs=rngs)

    def __call__(self, obs: jax.Array) -> jax.Array:
        x = obs.astype(jnp.float32)
        for conv in self.convs:
            x = nnx.relu(conv(x))
        x = x.reshape(*x.shape[: -self.obs_ndim], -1)
        for dense in self.dense:
            x = nnx.relu(dense(x))
        return self.out(x)


class ActorCritic(nnx.Module):
    """LPG's agent: an actor with policy logits and prediction logits y, whose updates come from the meta-network,
    and a separate critic used only for the outer A2C loss; the meta-network never sees its value."""

    def __init__(self, obs_shape: tuple[int, ...], num_actions: int, y_dim: int, channels: tuple[int, ...], hidden: tuple[int, ...], rngs: nnx.Rngs):
        self.num_actions = num_actions
        self.actor = FeedForward(obs_shape, num_actions + y_dim, channels, hidden, rngs)
        self.critic = FeedForward(obs_shape, 1, channels, hidden, rngs)

    def __call__(self, obs: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Returns policy logits, y logits and value."""
        out = self.actor(obs)
        return out[..., : self.num_actions], out[..., self.num_actions :], self.critic(obs).squeeze(-1)


class MetaNetwork(nnx.Module):
    """Learned Policy Gradient (Oh et al. 2020): a GRU unrolled backwards over a (T, B) trajectory turns rewards and
    the agent's outputs into update targets: pi_hat, a scalar per step that scales grad log pi(a|s), and y_hat logits.
    LPG used an LSTM; a GRU is the simpler equivalent. The paper's inputs: [r, terminated, gamma, pi(a|s), phi(y), phi(y_next)]."""

    def __init__(self, y_dim: int, hidden_dim: int, rngs: nnx.Rngs):
        self.hidden_dim = hidden_dim
        # phi: the same small embedding squashes y(s) and y(s') to a scalar each
        self.phi = nnx.Sequential(nnx.Linear(y_dim, 16, rngs=rngs), nnx.relu, nnx.Linear(16, 1, rngs=rngs))
        self.gru = nnx.GRUCell(6, hidden_dim, rngs=rngs)
        self.pi_hat = nnx.Linear(hidden_dim, 1, rngs=rngs)
        self.y_hat = nnx.Linear(hidden_dim, y_dim, rngs=rngs)

    def __call__(self, reward, terminated, boundary, gamma, prob, y, y_next) -> tuple[jax.Array, jax.Array]:
        """reward, terminated, boundary (terminated | truncated), gamma and prob (pi of the taken action) are (T, B);
        y and y_next are (T, B, y_dim) probabilities, y_next of the true next state, pre-reset. Returns pi_hat (T, B), y_hat logits (T, B, y_dim)."""
        x = jnp.stack([reward, terminated, gamma, prob, self.phi(y)[..., 0], self.phi(y_next)[..., 0]], axis=-1).astype(jnp.float32)

        def body(h, inputs):
            x_t, boundary_t = inputs
            # the carry comes from step t+1, which is another episode when t ended one
            h = jnp.where(boundary_t[:, None], 0.0, h)
            return self.gru(h, x_t)

        # a plain scan: nnx.scan rejects the cell's params when they are gradient tracers of the meta-gradient
        _, out = jax.lax.scan(body, jnp.zeros((reward.shape[1], self.hidden_dim)), (x, boundary), reverse=True)
        return self.pi_hat(out)[..., 0], self.y_hat(out)


def num_params(model: nnx.Module) -> int:
    return sum(x.size for x in jax.tree.leaves(nnx.state(model, nnx.Param)))
