"""Real CPU/CUDA simulator checks, plus batching and gradient tracing."""

import jax
import numpy as np
import pytest
from jax import numpy as jnp
from jax.extend import core

from modules.envs import arc, make


def test_no_click_catalog():
    assert set(arc.TRAIN) <= set(arc.PROCEDURAL)
    assert not set(arc.TRAIN) & set(arc.HOLDOUT)
    for name, meta in arc.ARC_GAMES.items():
        env, _ = make(f"Arc-{name}")
        assert 6 not in meta["actions"] and not meta["has_click"]
        assert env.num_actions == len(meta["actions"])
        assert env.observation_space().shape == (64, 64, 1)
        assert len(meta["budget_steps"]) >= meta["win_levels"]
    with pytest.raises(ValueError, match="no-click"):
        make("Arc-cf01")


def _ffi_calls(jaxpr):
    """Inspect actual JAX tracing, without substituting a simulator."""
    for eq in jaxpr.eqns:
        if eq.primitive.name == "ffi_call":
            yield eq
        for value in eq.params.values():
            for item in value if isinstance(value, (tuple, list)) else (value,):
                if isinstance(item, core.ClosedJaxpr):
                    yield from _ffi_calls(item.jaxpr)
                elif isinstance(item, core.Jaxpr):
                    yield from _ffi_calls(item)


def test_nested_vmap_scan_and_grad_keep_one_batched_step():
    env, _ = make("Arc-wk01")
    env.prepare(num_envs=3, agents=2, seed=891)
    keys = jax.random.split(jax.random.key(0), 6).reshape(2, 3)
    previous, _ = jax.jit(jax.vmap(jax.vmap(env.reset)))(keys)
    raw = jax.vmap(jax.vmap(env.pool.step))

    def loss(weight, rows, previous):
        def step(carry, _):
            total, previous = carry
            frame, reward, _, _ = raw(rows, previous)
            return (total + weight * reward.sum(), frame.astype(jnp.uint8)[..., None]), None
        return jax.lax.scan(step, (0.0, previous), None, length=4)[0][0]

    rows = jnp.zeros((2, 3, 4), jnp.uint8)
    traced = jax.make_jaxpr(jax.grad(loss))(1.0, rows, previous)
    calls = list(_ffi_calls(traced.jaxpr))
    assert len(calls) == 1
    assert calls[0].invars[0].aval.shape == (2, 3, 4)
    assert calls[0].params["has_side_effect"]
    assert jnp.isfinite(jax.jit(jax.grad(loss))(1.0, rows, previous))


def test_simulator_diversity_masked_resets_and_rollouts():
    env, _ = make("Arc-wk01")
    env.prepare(num_envs=4, agents=2, seed=9876)
    keys = jax.random.split(jax.random.key(0), 8).reshape(2, 4)
    reset = jax.jit(jax.vmap(jax.vmap(env.reset)))
    step = jax.jit(jax.vmap(jax.vmap(env.step)))
    obs, state = reset(keys)
    assert len(np.unique(np.asarray(obs).reshape(8, -1), axis=0)) > 1
    old = np.asarray(obs)
    mask = jnp.array([[True, False, False, False], [False, False, False, False]])
    reset_where = jax.jit(jax.vmap(jax.vmap(env.reset_where)))
    obs, state = reset_where(state, mask)
    np.testing.assert_array_equal(np.asarray(obs)[~np.asarray(mask)], old[~np.asarray(mask)])
    # None-reset must preserve the entire pytree, not advance simulation time.
    same_obs, same_state = reset_where(state, jnp.zeros_like(mask))
    for a, b in zip(jax.tree.leaves(state), jax.tree.leaves(same_state)):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(obs, same_obs)
    total = jnp.zeros((2, 4))
    ended = False
    for i in range(120):
        if i == 60:  # Exercise truncation without relying on a random policy reaching WIN.
            state = state._replace(steps=jnp.full_like(state.steps, max(env.meta["budget_steps"])))
        action = jax.random.randint(jax.random.key(i), (2, 4), 0, env.num_actions)
        obs, state, reward, done, info = step(keys, state, action)
        total += reward
        np.testing.assert_array_equal(np.asarray(info["score"])[np.asarray(done)], np.asarray(total)[np.asarray(done)])
        total = jnp.where(done, 0.0, total)
        ended |= bool(done.any())
        assert obs.shape == (2, 4, 64, 64, 1) and obs.dtype == jnp.uint8
        assert bool((obs < 16).all())
        assert bool((state.steps[done] == 0).all())
        assert bool((state.score[done] == 0).all())
    assert ended
