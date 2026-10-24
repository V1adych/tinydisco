"""Self-checks for disco.py: python src/test_disco.py"""

import tempfile
from dataclasses import replace

import jax
from flax import nnx
from jax import numpy as jnp

from disco import Config, lambda_returns, train


def test_lambda_returns():
    cfg = Config(gamma=0.5, gae_lambda=1.0)
    reward, value_next = jnp.array([[1.0], [2.0], [3.0]]), jnp.array([[10.0], [20.0], [30.0]])
    boundary = jnp.array([[False], [True], [False]])
    # the episode ends at step 1: no bootstrap there, the last step bootstraps from its next value
    returns = lambda_returns(reward, boundary, boundary, value_next, cfg)
    assert jnp.allclose(returns[:, 0], jnp.array([2.0, 2.0, 18.0])), returns
    # the episode is cut at step 1: it bootstraps from the true next state, not from the next stored return
    returns = lambda_returns(reward, jnp.zeros_like(boundary), boundary, value_next, cfg)
    assert jnp.allclose(returns[:, 0], jnp.array([7.0, 12.0, 18.0])), returns


def test_meta_steps_run():
    """A tabular and a conv block with per-lifetime env params, tiny sizes, two meta steps: the meta-network must change and stay finite."""
    cfg = Config(envs=("Chain-short", "Grid-small"), agents_per_env=2, num_envs=4, num_steps=5, num_updates=2, meta_steps=2, log_interval=1, save_interval=0, meta_hidden=8)
    before = jax.tree.leaves(nnx.state(train(replace(cfg, meta_steps=0))))  # the same seed gives the same initial meta-network
    after = jax.tree.leaves(nnx.state(train(cfg)))
    assert all(jnp.isfinite(a).all() for a in after)
    assert not all(jnp.allclose(b, a) for b, a in zip(before, after)), "the meta-network did not change"


def test_resume_restores_meta_network():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Config(envs=("Chain-short",), agents_per_env=2, num_envs=4, num_steps=5, num_updates=1, meta_steps=1, log_interval=1, save_interval=1, meta_hidden=8, checkpoint_dir=tmp, run_name="r")
        trained = jax.tree.leaves(nnx.state(train(cfg)))
        resumed = jax.tree.leaves(nnx.state(train(replace(cfg, meta_steps=0, resume=f"{tmp}/r/1"))))
    assert all(jnp.array_equal(a, b) for a, b in zip(trained, resumed)), "resume did not restore the trained meta-network"


if __name__ == "__main__":
    test_lambda_returns()
    test_meta_steps_run()
    test_resume_restores_meta_network()
    print("ok")
