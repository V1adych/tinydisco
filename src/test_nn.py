"""Self-checks for the toy agent and the meta-network: python src/test_nn.py"""

import jax
from flax import nnx
from jax import numpy as jnp

from envs import make
from nn import ActorCritic, MetaNetwork


def test_agent_shapes():
    for name, channels, hidden in [("Chain-short", (), ()), ("Chain-distraction", (), (16,)), ("Grid-dense", (16,), (32,)), ("Asterix-MinAtar", (16,), (32,))]:
        env, params = make(name)
        obs, _ = jax.vmap(env.reset, in_axes=(0, None))(jax.random.split(jax.random.key(0), 5), params)
        agent = ActorCritic(env.observation_space(params).shape, env.num_actions, 30, channels, hidden, nnx.Rngs(0))
        logits, y, value = agent(obs)
        assert logits.shape == (5, env.num_actions) and y.shape == (5, 30) and value.shape == (5,), name


def test_meta_looks_only_ahead_within_an_episode():
    T, B, Y = 10, 3, 30
    keys = jax.random.split(jax.random.key(0), 3)
    reward = jax.random.normal(keys[0], (T, B))
    y = jax.nn.softmax(jax.random.normal(keys[1], (T, B, Y)))
    boundary = jnp.zeros((T, B), bool).at[5].set(True)  # an episode ends at step 5
    net = MetaNetwork(Y, 64, nnx.Rngs(0))
    inputs = dict(terminated=boundary, boundary=boundary, gamma=jnp.full((T, B), 0.99), prob=jnp.full((T, B), 0.5), y=y, y_next=y)

    pi_hat, y_hat = net(reward, **inputs)
    assert pi_hat.shape == (T, B) and y_hat.shape == (T, B, Y)

    # the next episode's reward must not reach steps 0..5, and must reach its own steps
    pi_hat_changed, _ = net(reward.at[7].add(10.0), **inputs)
    assert jnp.allclose(pi_hat[:6], pi_hat_changed[:6]), "information leaked across the episode boundary"
    assert not jnp.allclose(pi_hat[6:8], pi_hat_changed[6:8])
    # within an episode a later reward reaches earlier steps (backward), but never later ones
    pi_hat_changed, _ = net(reward.at[3].add(10.0), **inputs)
    assert not jnp.allclose(pi_hat[:4], pi_hat_changed[:4]) and jnp.allclose(pi_hat[4:], pi_hat_changed[4:])


if __name__ == "__main__":
    test_agent_shapes()
    test_meta_looks_only_ahead_within_an_episode()
    print("ok")
