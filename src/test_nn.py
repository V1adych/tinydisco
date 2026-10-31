"""Self-checks for the agent network and the meta-network: python src/test_nn.py"""

import jax
import numpy as np
from flax import nnx
from jax import numpy as jnp

from envs import make
from envs.arc import NUM_COLORS
from nn import MetaNetwork, Recurrent, RuleInputs, agent_heads, projected_gru, space_to_depth

PATCH = (((4, 32), (3, 64)), 512)  # patch layers and width, as disco.ATARI


def test_agent_heads():
    env, params = make("Atari-breakout")
    obs, _ = jax.vmap(env.reset, in_axes=(0, None))(jax.random.split(jax.random.key(0), 5), params)
    num_actions, d = env.num_actions, 8
    agent = Recurrent(num_actions + d + num_actions * d + num_actions, obs.shape[1:], *PATCH, nnx.Rngs(0))
    valid = jnp.arange(num_actions) < 2
    carry, out = agent.step(jnp.zeros((5, 512)), obs)
    logits, y, z, q = agent_heads(out, num_actions, d, valid)
    assert carry.shape == (5, 512)
    assert logits.shape == (5, num_actions) and y.shape == (5, d) and z.shape == (5, num_actions, d) and q.shape == (5, num_actions)
    assert logits.dtype == jnp.float32
    assert (jax.nn.softmax(logits)[:, 2:] == 0).all(), "illegal actions must get no probability"


def test_unroll_matches_steps():
    """Unrolling a trajectory gives what stepping through it gives, with the carry restarted after an episode end."""
    T, B = 4, 2
    net = Recurrent(3, (8, 8, 1), ((4, 8),), 16, nnx.Rngs(0), colors=NUM_COLORS)
    obs = jax.random.randint(jax.random.key(0), (T + 1, B, 8, 8, 1), 0, NUM_COLORS, jnp.uint8)
    done = jnp.zeros((T, B), bool).at[1, 0].set(True)
    carry, outs = jnp.zeros((B, 16)), []
    for t in range(T + 1):
        carry = jnp.where((done[t - 1] if t else jnp.zeros(B, bool))[:, None], 0.0, carry)
        carry, out = net.step(carry, obs[t])
        outs.append(out)
    assert jnp.allclose(net.unroll(jnp.zeros((B, 16)), obs, done), jnp.stack(outs), atol=1e-6)


def test_projected_gru_matches_flax():
    """Preserve outputs, gradients and meta-gradients, including resets in both scan directions."""
    for inputs, hidden, reverse in ((16, 16, False), (19, 8, True)):
        graph, params = nnx.split(nnx.GRUCell(inputs, hidden, rngs=nnx.Rngs(0)))
        x = jax.random.normal(jax.random.key(1), (5, 3, inputs))
        h = jax.random.normal(jax.random.key(2), (3, hidden))
        resets = jnp.zeros((5, 3), bool).at[2, 0].set(True).at[3, 2].set(True)

        def unroll(args, projected):
            params, x, h = args
            cell = nnx.merge(graph, params)
            if projected:
                x = cell.dense_i(x)

            def body(h, item):
                xi, reset = item
                h = jnp.where(reset[:, None], 0, h)
                return projected_gru(cell, h, xi) if projected else cell(h, xi)

            return jax.lax.scan(body, h, (x, resets), reverse=reverse)[1]

        def loss(args, projected):
            y = unroll(args, projected)
            return jnp.sin(y).sum() + 0.01 * (y * y).sum()

        args = (params, x, h)
        direction = jax.tree.map(lambda a: jnp.full_like(a, 0.01), args)
        outputs = []
        for projected in (False, True):
            grad = jax.grad(lambda a: loss(a, projected))
            outputs.append((jax.jit(lambda a: unroll(a, projected))(args), jax.jit(grad)(args),
                            jax.jit(lambda a: jax.jvp(grad, (a,), (direction,))[1])(args)))
        for left, right in zip(jax.tree.leaves(outputs[0]), jax.tree.leaves(outputs[1])):
            np.testing.assert_allclose(left, right, rtol=1e-4, atol=1e-5)


def test_patch_layer_is_the_strided_conv():
    """A patch layer over k x k blocks is the k x k stride-k conv whose kernel is the layer's weight, unfolded."""
    net = Recurrent(1, (84, 84, 1), *PATCH, nnx.Rngs(0))
    x = jax.random.normal(jax.random.key(1), (3, 84, 84, 1)).astype(jnp.bfloat16)  # the layer computes in bf16
    layer, k = net.layers[0], net.blocks[0]
    kernel = layer.kernel.value.astype(jnp.bfloat16).astype(jnp.float32).reshape(k, k, 1, -1)
    strided = jax.lax.conv_general_dilated(x.astype(jnp.float32), kernel, (k, k), "VALID", dimension_numbers=("NHWC", "HWIO", "NHWC")) + layer.bias.value
    assert jnp.allclose(strided, layer(space_to_depth(x, k)).astype(jnp.float32), rtol=1e-2, atol=2e-2), "the patch layer differs from the strided conv"


def test_color_frames():
    env, _ = make("Arc-ls20")
    obs, _ = jax.vmap(env.reset)(jax.random.split(jax.random.key(0), 6))
    net = Recurrent(7, obs.shape[1:], ((4, 32), (2, 64)), 512, nnx.Rngs(0), colors=NUM_COLORS)
    _, out = net.step(jnp.zeros((6, 512)), obs)
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
    test_projected_gru_matches_flax()
    test_patch_layer_is_the_strided_conv()
    test_color_frames()
    test_meta_looks_only_ahead_within_an_episode()
    test_meta_serves_any_number_of_actions()
    test_meta_ignores_padded_actions()
    print("ok")
