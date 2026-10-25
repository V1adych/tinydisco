"""Self-checks for utils.py: python src/test_utils.py"""

import jax
import optax
from jax import numpy as jnp

from utils import mixflow_grad, retrace, vtrace

# a 3-step rollout, gamma 0.5, whose episode ends at step 1; on-policy with full traces both estimators are Monte Carlo
REWARD = jnp.array([[1.0], [2.0], [3.0]])
VALUE = jnp.array([[5.0], [6.0], [7.0], [30.0]])  # V(s_t) on T + 1 observations, and Q(s_t, a_t) for Retrace
DISCOUNT = 0.5 * jnp.array([[1.0], [0.0], [1.0]])
ONES = jnp.ones_like(REWARD)


def test_off_policy_returns():
    # no bootstrap where the episode ended; the last step bootstraps from the value after it
    expected = jnp.array([[2.0], [2.0], [18.0]])
    assert jnp.allclose(retrace(VALUE[:-1], VALUE, REWARD, DISCOUNT, ONES), expected)
    assert jnp.allclose(vtrace(VALUE, REWARD, DISCOUNT, ONES, 1.0)[0], expected)
    # zero traces: one-step targets
    assert jnp.allclose(retrace(VALUE[:-1], VALUE, REWARD, DISCOUNT, 0 * ONES), REWARD + DISCOUNT * VALUE[1:])
    assert jnp.allclose(vtrace(VALUE, REWARD, DISCOUNT, ONES, 0.0)[0], REWARD + DISCOUNT * VALUE[1:])


def test_mixflow_grad_matches_plain_meta_gradient():
    """The MixFlow-MG backward pass of a gradient equals reverse-over-reverse, in params and in targets."""
    keys = jax.random.split(jax.random.key(0), 5)
    params = {"w": jax.random.normal(keys[0], (4, 3)), "b": jax.random.normal(keys[1], (3,))}
    targets = jax.nn.softmax(jax.random.normal(keys[2], (8, 3)))
    data = jax.random.normal(keys[3], (8, 4))
    probe = jax.tree.map(lambda p: jax.random.normal(keys[4], p.shape), params)

    def loss(params, targets, data):  # a KL to the targets, like the agent's
        log_q = jax.nn.log_softmax(jnp.tanh(data @ params["w"] + params["b"]))
        return (targets * (jnp.log(targets) - log_q)).sum(-1).mean()

    def meta_objective(grad_fn):  # a scalar of the gradient, as a meta-loss is
        return jax.grad(lambda p, t: optax.tree.vdot(grad_fn(p, t, data), probe), (0, 1))(params, targets)

    expected, got = meta_objective(jax.grad(loss)), meta_objective(mixflow_grad(loss))
    for e, g in zip(jax.tree.leaves(expected), jax.tree.leaves(got)):
        assert jnp.allclose(e, g, rtol=1e-5, atol=1e-6), "MixFlow-MG differs from the plain meta-gradient"


if __name__ == "__main__":
    test_off_policy_returns()
    test_mixflow_grad_matches_plain_meta_gradient()
    print("ok")
