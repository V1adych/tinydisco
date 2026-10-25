"""Self-checks for the agent network and the meta-network: python src/test_nn.py"""

import jax
from flax import nnx
from jax import numpy as jnp

from envs import make
from nn import FeedForward, MetaNetwork, RuleInputs, agent_heads


def test_agent_heads():
    env, params = make("Atari-breakout")
    obs, _ = jax.vmap(env.reset, in_axes=(0, None))(jax.random.split(jax.random.key(0), 5), params)
    num_actions, d = env.num_actions, 8
    agent = FeedForward(num_actions + d + num_actions * d + num_actions, nnx.Rngs(0))
    valid = jnp.arange(num_actions) < 2
    logits, y, z, q = agent_heads(agent(obs), num_actions, d, valid)
    assert logits.shape == (5, num_actions) and y.shape == (5, d) and z.shape == (5, num_actions, d) and q.shape == (5, num_actions)
    assert logits.dtype == jnp.float32
    assert (jax.nn.softmax(logits)[:, 2:] == 0).all(), "illegal actions must get no probability"


def rule_inputs(reward):
    """Random inputs for T = 10 steps, N = 3 envs, A = 4 actions, D = 6 bins, where every episode ends at step 5."""
    T, N, A, D = 10, 3, 4, 6
    keys = jax.random.split(jax.random.key(0), 6)
    boundary = jnp.zeros((T, N), bool).at[5].set(True)
    probs = lambda key, shape: jax.nn.softmax(jax.random.normal(key, shape))
    pi = probs(keys[0], (T + 1, N, A))
    q = jax.random.normal(keys[1], (T, N, A))
    v = (pi[:-1] * q).sum(-1)
    return RuleInputs(
        action=jax.random.randint(keys[2], (T, N), 0, A), reward=reward, discount=1.0 - boundary,
        pi=pi, mu=probs(keys[3], (T, N, A)), y=probs(keys[4], (T + 1, N, D)), z=probs(keys[5], (T + 1, N, A, D)),
        q=q, v=v, v_next=jnp.roll(v, -1, 0), adv=jnp.zeros((T, N)), valid=jnp.ones(A, bool),
    )


def test_meta_looks_only_ahead_within_an_episode():
    net = MetaNetwork(6, 16, nnx.Rngs(0))
    reward = jax.random.normal(jax.random.key(1), (10, 3))
    pi_hat, y_hat, z_hat = net(rule_inputs(reward))
    assert pi_hat.shape == (10, 3, 4) and y_hat.shape == (10, 3, 6) and z_hat.shape == (10, 3, 6)

    # the next episode's reward must not reach steps 0..5, and must reach its own steps
    changed, _, _ = net(rule_inputs(reward.at[7].add(10.0)))
    assert jnp.allclose(pi_hat[:6], changed[:6]), "information leaked across the episode boundary"
    assert not jnp.allclose(pi_hat[6:8], changed[6:8])
    # within an episode a later reward reaches earlier steps (backward), but never later ones
    changed, _, _ = net(rule_inputs(reward.at[3].add(10.0)))
    assert not jnp.allclose(pi_hat[:4], changed[:4]) and jnp.allclose(pi_hat[4:], changed[4:])


def test_meta_serves_any_number_of_actions():
    net = MetaNetwork(6, 16, nnx.Rngs(0))
    x = rule_inputs(jnp.zeros((10, 3)))
    wider = x._replace(pi=jnp.concatenate([x.pi, x.pi], -1) / 2, mu=jnp.concatenate([x.mu, x.mu], -1) / 2,
                       z=jnp.concatenate([x.z, x.z], -2), q=jnp.concatenate([x.q, x.q], -1), valid=jnp.ones(8, bool))
    assert net(wider)[0].shape == (10, 3, 8)


def test_meta_ignores_padded_actions():
    """Illegal padding actions, whatever their inputs, change none of the rule's outputs on the legal ones."""
    net = MetaNetwork(6, 16, nnx.Rngs(0))
    x = rule_inputs(jax.random.normal(jax.random.key(1), (10, 3)))
    pad = lambda a, value: jnp.concatenate([a, jnp.full(a.shape[:-1] + (3,), value, a.dtype)], -1)
    padded = x._replace(pi=pad(x.pi, 0.0), mu=pad(x.mu, 0.0), q=pad(x.q, 5.0), valid=pad(x.valid, False),
                        z=jnp.concatenate([x.z, jnp.full(x.z.shape[:-2] + (3, x.z.shape[-1]), 0.5)], -2))
    (pi_hat, y_hat, z_hat), (pi_pad, y_pad, z_pad) = net(x), net(padded)
    assert jnp.allclose(pi_hat, pi_pad[..., :4], atol=1e-5) and jnp.allclose(y_hat, y_pad, atol=1e-5) and jnp.allclose(z_hat, z_pad, atol=1e-5)
    assert (jax.nn.softmax(pi_pad)[..., 4:] == 0).all()


if __name__ == "__main__":
    test_agent_heads()
    test_meta_looks_only_ahead_within_an_episode()
    test_meta_serves_any_number_of_actions()
    test_meta_ignores_padded_actions()
    print("ok")
