"""Self-checks for the envs package: python src/test_envs.py"""

import jax
from jax import numpy as jnp

from envs import make
from envs.chain import CHAINS
from envs.grid import GRIDS, GridState


def test_chain_pays_for_the_first_action():
    for variant in CHAINS:
        env, params = make(f"Chain-{variant}")
        params = params.replace(chain_length=7)
        for first_correct in (True, False):
            _, state = env.reset(jax.random.key(0), params)
            first = state.correct if first_correct else 1 - state.correct
            rewards = []
            for t in range(7):
                # later actions are deliberately the wrong one: they must not matter
                action = first if t == 0 else 1 - state.correct
                _, state, reward, done, _ = env.step(jax.random.key(t + 1), state, action, params)
                rewards.append(float(reward))
                assert bool(done) == (t == 6), (variant, t)
            assert rewards[-1] == (1.0 if first_correct else -1.0), variant
            if "noisy" not in variant:
                assert rewards[:-1] == [0.0] * 6, variant


def test_grid_collects():
    key = jax.random.key(0)
    state = GridState(time=jnp.int32(0), agent=jnp.array([0, 0]), objects=jnp.array([[0, 1], [5, 5], [6, 6], [7, 7]]), active=jnp.ones(4, bool))
    right = 5  # MOVES[5] is (0, 1)

    env, params = make("Grid-dense")
    _, moved, reward, _, _ = env.step(key, state, right, params)
    assert reward == 1.0 and not moved.active[0] and (moved.agent == jnp.array([0, 1])).all()

    env, params = make("Grid-dense-collect")
    _, moved, reward, _, _ = env.step(key, state, right, params)
    assert reward == 0.0 and moved.active[0], "with collect actions, stepping on an object must not collect it"
    _, stayed, reward, _, _ = env.step(key, state, 9 + right, params)
    assert reward == 1.0 and not stayed.active[0] and (stayed.agent == jnp.array([0, 0])).all()


def test_random_rollouts():
    """Every env runs vmapped and jitted through make()'s interface with in-range observations and rewards."""
    names = [f"Chain-{v}" for v in CHAINS] + [f"Grid-{v}" for v in GRIDS] + ["Grid-dense-collect", "Asterix-MinAtar", "Craftax-Classic-Symbolic-v1"]
    for name in names:
        env, params = make(name)
        num_envs, num_steps = 64, 50 if name.startswith("Craftax") else 3000

        @jax.jit
        def run(key):
            obs, state = jax.vmap(env.reset, in_axes=(0, None))(jax.random.split(key, num_envs), params)

            def body(carry, key):
                state, _ = carry
                key_action, key_step = jax.random.split(key)
                action = jax.random.randint(key_action, (num_envs,), 0, env.num_actions)
                obs, state, reward, done, _ = jax.vmap(env.step, in_axes=(0, 0, 0, None))(jax.random.split(key_step, num_envs), state, action, params)
                return (state, obs), (reward, done)

            (_, obs), (reward, done) = jax.lax.scan(body, (state, obs), jax.random.split(key, num_steps))
            return obs, reward, done

        obs, reward, done = run(jax.random.key(0))
        assert obs.shape == (num_envs, *env.observation_space(params).shape), name
        assert jnp.isfinite(obs).all() and jnp.isfinite(reward).all(), name
        # random craftax agents rarely die within a short test
        assert done.any() or name.startswith("Craftax"), f"{name}: no episode ended in {num_steps} steps"


def test_atari_keeps_the_final_observation():
    """When an episode ends, info holds its last frame and obs is the new episode's first; otherwise they are the same frame."""
    env, params = make("Atari-breakout")
    num_envs = 8

    @jax.jit
    def run(key):
        obs, state = jax.vmap(env.reset, in_axes=(0, None))(jax.random.split(key, num_envs), params)

        def body(carry, key):
            state, obs = carry
            action = jax.random.randint(key, (num_envs,), 0, env.num_actions)
            next_obs, state, _, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(jax.random.split(key, num_envs), state, action, params)
            return (state, next_obs), (obs, next_obs, done, info)

        _, out = jax.lax.scan(body, (state, obs), jax.random.split(key, 300))
        return out

    obs, next_obs, done, info = run(jax.random.key(0))
    assert done.any(), "no breakout episode ended in 300 random steps"
    assert obs.shape[2:] == (84, 84, 1) and obs.dtype == jnp.uint8
    assert (info["final_observation"][~done] == next_obs[~done]).all()
    assert (info["score"][~done] == 0).all() and (info["score"][done] >= 0).all()


if __name__ == "__main__":
    test_chain_pays_for_the_first_action()
    test_grid_collects()
    test_random_rollouts()
    test_atari_keeps_the_final_observation()
    print("ok")
