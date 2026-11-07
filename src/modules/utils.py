import jax
import optax
from jax import numpy as jnp


def kl(p_logits, q_logits) -> jax.Array:
    """KL(p || q) over the last axis."""
    log_p = jax.nn.log_softmax(p_logits)
    return (jnp.exp(log_p) * (log_p - jax.nn.log_softmax(q_logits))).sum(-1)


def entropy(logits) -> jax.Array:
    log_p = jax.nn.log_softmax(logits)
    return -(jnp.exp(log_p) * log_p).sum(-1)


def retrace(q_a, v, reward, discount, c) -> jax.Array:
    """Retrace targets for Q(s_t, a_t) over T steps: G_t = r_t + discount_t (v_{t+1} + c_{t+1} (G_{t+1} - Q(s_{t+1}, a_{t+1}))),
    with c = lambda min(1, pi/mu). q_a, reward, discount and c are (T, ...), v is (T + 1, ...); a zero discount (an
    episode's end) cuts the trace, and so does the end of the rollout."""

    def body(trace, t):
        q_a, v_next, reward, discount, c = t
        ret = reward + discount * (v_next + trace)
        return c * (ret - q_a), ret

    _, returns = jax.lax.scan(body, jnp.zeros_like(reward[0]), (q_a, v[1:], reward, discount, c), reverse=True)
    return returns


def vtrace(v, reward, discount, rho, lam) -> tuple[jax.Array, jax.Array]:
    """V-trace (Espeholt et al. 2018) over T steps: value targets and policy-gradient advantages for rho = pi/mu.
    v is (T + 1, ...), the rest (T, ...); a zero discount cuts the trace."""
    rho = jnp.minimum(rho, 1.0)
    delta = rho * (reward + discount * v[1:] - v[:-1])

    def body(acc, t):
        delta, discount, c = t
        acc = delta + discount * c * acc
        return acc, acc

    _, acc = jax.lax.scan(body, jnp.zeros_like(v[0]), (delta, discount, lam * rho), reverse=True)
    targets = v[:-1] + acc
    targets_next = jnp.concatenate([targets[1:], v[-1:]])
    return targets, rho * (reward + discount * targets_next - v[:-1])


def scale_by_adam_sg_denom(b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8) -> optax.GradientTransformation:
    """Adam's scaling whose denominator passes no meta-gradient, as DiscoRL's optimizers.py and nanodisco: the
    meta-gradient flows through the step's direction, not through Adam's per-parameter step size."""

    def init(params):
        zeros = jax.tree.map(jnp.zeros_like, params)
        return optax.ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=zeros, nu=zeros)

    def update(grads, state, params=None):
        count = state.count + 1
        mu = jax.tree.map(lambda m, g: b1 * m + (1 - b1) * g, state.mu, grads)
        nu = jax.tree.map(lambda v, g: b2 * v + (1 - b2) * jax.lax.stop_gradient(g) ** 2, state.nu, grads)
        updates = jax.tree.map(lambda m, v: m / (1 - b1**count) / (jnp.sqrt(v / (1 - b2**count)) + eps), mu, nu)
        return updates, optax.ScaleByAdamState(count=count, mu=mu, nu=nu)

    return optax.GradientTransformation(init, update)


def mixflow_grad(loss):
    """jax.grad(loss) in its first argument, for loss(params, targets, data), whose vjp is one forward-over-reverse
    pass (MixFlow-MG, Kemaev et al. 2025): v^T H = H v, as the Hessian is symmetric. Plain reverse-over-reverse stores
    the whole backward pass of every update a meta-gradient goes through; this stores only the update's inputs."""

    @jax.custom_vjp
    def grad(params, targets, data):
        return jax.grad(loss)(params, targets, data)

    def fwd(params, targets, data):
        return grad(params, targets, data), (params, targets, data)

    def bwd(saved, v):
        params, targets, data = saved
        _, (h_params, h_targets) = jax.jvp(lambda p: jax.grad(loss, (0, 1))(p, targets, data), (params,), (v,))
        return h_params, h_targets, None

    grad.defvjp(fwd, bwd)
    return grad
