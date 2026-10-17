import os

import jax
import orbax.checkpoint as ocp
from flax import nnx


class ActorCriticRNN(nnx.Module):
    """Patch-conv encoder -> GRU -> policy logits and value."""

    def __init__(self, obs_shape: tuple[int, ...], num_actions: int, embed_dim: int, hidden_dim: int, patch_size: tuple[int, int], rngs: nnx.Rngs):
        h, w, c = obs_shape
        num_patches = (h // patch_size[0]) * (w // patch_size[1])
        self.hidden_dim = hidden_dim
        self.encoder = nnx.Conv(in_features=c, out_features=embed_dim, kernel_size=patch_size, strides=patch_size, padding="VALID", rngs=rngs)
        self.proj = nnx.Linear(num_patches * embed_dim, hidden_dim, rngs=rngs)
        self.rnn = nnx.GRUCell(hidden_dim, hidden_dim, rngs=rngs)
        self.actor = nnx.Linear(hidden_dim, num_actions, rngs=rngs)
        self.critic = nnx.Linear(hidden_dim, 1, rngs=rngs)

    def initialize_carry(self, batch_size: int) -> jax.Array:
        return self.rnn.initialize_carry((batch_size, self.hidden_dim), rngs=nnx.Rngs(0))

    def __call__(self, carry: jax.Array, obs: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        x = nnx.relu(self.encoder(obs))
        x = nnx.relu(self.proj(x.reshape(*x.shape[:-3], -1)))
        carry, x = self.rnn(carry, x)
        return carry, self.actor(x), self.critic(x).squeeze(-1)


def num_params(model: nnx.Module) -> int:
    return sum(x.size for x in jax.tree.leaves(nnx.state(model, nnx.Param)))


def save_checkpoint(directory: str, step: int, model: nnx.Module):
    path = os.path.abspath(os.path.join(directory, str(step)))
    with ocp.StandardCheckpointer() as ckptr:
        ckptr.save(path, nnx.state(model))


def load_checkpoint(path: str, model: nnx.Module):
    """Restores params saved by save_checkpoint into an already constructed model."""
    with ocp.StandardCheckpointer() as ckptr:
        state = ckptr.restore(os.path.abspath(path), nnx.state(model))
    nnx.update(model, state)
