"""Self-checks for disco.py: python src/test_disco.py"""

import tempfile
from dataclasses import replace

import jax
from flax import nnx
from jax import numpy as jnp

from disco import Config, retrace, train, vtrace

# a 3-step rollout, gamma 0.5, whose episode ends at step 1; on-policy with full traces both estimators are Monte Carlo
REWARD = jnp.array([[1.0], [2.0], [3.0]])
VALUE = jnp.array([[5.0], [6.0], [7.0]])  # V(s_t), and Q(s_t, a_t) for Retrace
VALUE_NEXT = jnp.array([[6.0], [99.0], [30.0]])  # step 1's is the pre-reset state, other steps' the next stored one
BOUNDARY = jnp.array([[False], [True], [False]])
ONES = jnp.ones_like(REWARD)


def test_off_policy_returns():
    terminated = 0.5 * (1 - BOUNDARY)
    truncated = 0.5 * ONES
    # terminated at step 1: no bootstrap there; the last step bootstraps from its next value
    expected = jnp.array([[2.0], [2.0], [18.0]])
    assert jnp.allclose(retrace(VALUE, VALUE_NEXT, REWARD, terminated, BOUNDARY, ONES), expected)
    assert jnp.allclose(vtrace(VALUE, VALUE_NEXT, REWARD, terminated, BOUNDARY, ONES, 1.0)[0], expected)
    # truncated at step 1: it bootstraps from the pre-reset state, and step 0's trace runs through it
    expected = jnp.array([[26.75], [51.5], [18.0]])
    assert jnp.allclose(retrace(VALUE, VALUE_NEXT, REWARD, truncated, BOUNDARY, ONES), expected)
    assert jnp.allclose(vtrace(VALUE, VALUE_NEXT, REWARD, truncated, BOUNDARY, ONES, 1.0)[0], expected)
    # zero traces: one-step targets
    assert jnp.allclose(retrace(VALUE, VALUE_NEXT, REWARD, truncated, BOUNDARY, 0 * ONES), REWARD + 0.5 * VALUE_NEXT)


TINY = Config(games=("breakout", "pong"), agents_per_game=2, num_envs=2, num_steps=3, num_updates=2, d=4, meta_hidden=8, meta_steps=2, log_interval=1, save_interval=0)


def test_meta_steps_run():
    """Two games with different action counts, tiny sizes, two meta steps: the meta-network must change and stay finite."""
    before = jax.tree.leaves(nnx.state(train(replace(TINY, meta_steps=0))))  # the same seed gives the same initial meta-network
    after = jax.tree.leaves(nnx.state(train(TINY)))
    assert all(jnp.isfinite(a).all() for a in after)
    assert not all(jnp.allclose(b, a) for b, a in zip(before, after)), "the meta-network did not change"


def test_resume_restores_meta_network():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = replace(TINY, games=("breakout",), meta_steps=1, save_interval=1, checkpoint_dir=tmp, run_name="r")
        trained = jax.tree.leaves(nnx.state(train(cfg)))
        resumed = jax.tree.leaves(nnx.state(train(replace(cfg, meta_steps=0, resume=f"{tmp}/r/1"))))
    assert all(jnp.array_equal(a, b) for a, b in zip(trained, resumed)), "resume did not restore the trained meta-network"


if __name__ == "__main__":
    test_off_policy_returns()
    test_meta_steps_run()
    test_resume_restores_meta_network()
    print("ok")
