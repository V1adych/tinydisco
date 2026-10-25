"""Self-checks for meta.py: python src/test_meta.py"""

import tempfile
from dataclasses import replace

import jax
from flax import nnx
from jax import numpy as jnp

from checkpoint import save_checkpoint
from meta import Config, train
from nn import MetaNetwork


def test_updates_run():
    """A random meta-network saved as a checkpoint trains a tiny recurrent agent on a grid world: the agent must change and stay finite."""
    with tempfile.TemporaryDirectory() as tmp:
        path = save_checkpoint(tmp, 0, MetaNetwork(4, 8, nnx.Rngs(1)))
        cfg = Config(meta_rule=str(path), env_name="Grid-dense", num_envs=4, num_steps=5, total_timesteps=2 * 4 * 5, embed_dim=4, hidden_dim=8, patch_size=(1, 1), y_dim=4, meta_hidden=8, log_interval=1, save_interval=0)
        before = jax.tree.leaves(nnx.state(train(replace(cfg, total_timesteps=0))))  # the same seed gives the same initial agent
        after = jax.tree.leaves(nnx.state(train(cfg)))
    assert all(jnp.isfinite(a).all() for a in after)
    assert not all(jnp.allclose(b, a) for b, a in zip(before, after)), "the agent did not change"


if __name__ == "__main__":
    test_updates_run()
    print("ok")
