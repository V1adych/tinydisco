"""Self-checks for the agent network and the meta-network: python src/test_nn.py"""

import jax
from flax import nnx
from jax import numpy as jnp

from envs import make
from envs.arc import NUM_COLORS
from nn import MetaNetwork, Recurrent, RuleInputs, agent_heads, space_to_depth

PATCH = (((4, 32, 1), (3, 64, 1)), 512)  # convs and width, as disco.ATARI


def test_agent_heads():
    env, params = make("Atari-breakout")
    obs, _ = jax.vmap(env.reset, in_axes=(0, None))(jax.random.split(jax.random.key(0), 5), params)
    num_actions, d = env.num_actions, 8
    agent = Recurrent(num_actions + d + num_actions * d + num_actions, obs.shape[1:], *PATCH, nnx.Rngs(0))
    valid = jnp.arange(num_actions) < 2
    carry, out = agent.step(agent.initial_carry(5), obs)
    logits, y, z, q = agent_heads(out, num_actions, d, valid)
    assert carry.shape == (5, 512)
    assert logits.shape == (5, num_actions) and y.shape == (5, d) and z.shape == (5, num_actions, d) and q.shape == (5, num_actions)
    assert logits.dtype == jnp.float32
    assert (jax.nn.softmax(logits)[:, 2:] == 0).all(), "illegal actions must get no probability"


def test_unroll_matches_steps():
    """Unrolling a trajectory gives what stepping through it gives, with the carry restarted after an episode end."""
    T, B = 4, 2
    net = Recurrent(3, (8, 8, 1), ((4, 8, 1),), 16, nnx.Rngs(0), colors=NUM_COLORS)
    obs = jax.random.randint(jax.random.key(0), (T + 1, B, 8, 8, 1), 0, NUM_COLORS, jnp.uint8)
    done = jnp.zeros((T, B), bool).at[1, 0].set(True)
    carry, outs = net.initial_carry(B), []
    for t in range(T + 1):
        carry = jnp.where((done[t - 1] if t else jnp.zeros(B, bool))[:, None], 0.0, carry)
        carry, out = net.step(carry, obs[t])
        outs.append(out)
    assert jnp.allclose(net.unroll(net.initial_carry(B), obs, done), jnp.stack(outs), atol=1e-6)


def test_block_convs_are_the_strided_convs():
    """A 2x2 conv over k x k blocks equals the k*2 x k*2 stride-k conv whose kernel is the block conv's, unfolded."""
    net = Recurrent(1, (84, 84, 4), ((4, 32, 2), (2, 64, 2), (1, 64, 3)), 32, nnx.Rngs(0))
    x = jax.random.normal(jax.random.key(1), (3, 84, 84, 4))
    for conv, k in zip(net.convs[:2], net.blocks):
        kernel, c = conv.kernel.value, conv.in_features // (k * k)
        unfolded = kernel.reshape(2, 2, k, k, c, -1).transpose(0, 2, 1, 3, 4, 5).reshape(2 * k, 2 * k, c, -1)
        strided = jax.lax.conv_general_dilated(x, unfolded, (k, k), "VALID", dimension_numbers=("NHWC", "HWIO", "NHWC")) + conv.bias.value
        blocks = jax.lax.conv_general_dilated(space_to_depth(x, k), kernel, (1, 1), "VALID", dimension_numbers=("NHWC", "HWIO", "NHWC")) + conv.bias.value
        assert jnp.allclose(strided, blocks, atol=1e-4), f"block conv {k} differs from the strided conv"
        x = jax.random.normal(jax.random.key(2), (3, 20, 20, 32))


def test_color_frames_encode_the_same_per_step_and_per_trajectory():
    """The per-step one-hot path of a trajectory's frames equals encoding the frames as one batch."""
    env, _ = make("Arc-ls20")
    obs, _ = jax.vmap(env.reset)(jax.random.split(jax.random.key(0), 6))
    net = Recurrent(7, obs.shape[1:], ((4, 32, 1), (2, 64, 1)), 512, nnx.Rngs(0), colors=NUM_COLORS)
    assert jnp.array_equal(net.encode(obs.reshape(3, 2, *obs.shape[1:])), net.encode(obs).reshape(3, 2, -1))
    _, out = net.step(net.initial_carry(6), obs)
    assert out.shape == (6, 7) and out.dtype == jnp.float32


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
    test_unroll_matches_steps()
    test_block_convs_are_the_strided_convs()
    test_color_frames_encode_the_same_per_step_and_per_trajectory()
    test_meta_looks_only_ahead_within_an_episode()
    test_meta_serves_any_number_of_actions()
    test_meta_ignores_padded_actions()
    print("ok")
