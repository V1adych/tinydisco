"""Self-checks for disco.py: python -m pytest tests/test_disco.py"""

import tempfile
from dataclasses import replace
from functools import partial

import jax
import pytest
from flax import nnx
from jax import numpy as jnp

from scripts.disco import Config, Spec, critic_values, init_buffer, make_nets, pack_obs, sample, store, train, unpack_obs
from modules.envs import arc, make

TINY = Config(
    games=("Atari-breakout", "Arc-wk01"), agents_per_game=2, num_envs=2, num_steps=3, num_updates=3, batch=4, buffer_size=30, valid_rollouts=1,
    d=4, meta_hidden=8, meta_steps=2, log_interval=1, save_interval=0,
)
TINY_SPEC = Spec(((4, 8),), 16, lr=3e-4, lifetimes=(100, 1000))
TINY_SPECS = {"Atari-breakout": TINY_SPEC, "Arc-wk01": TINY_SPEC._replace(colors=arc.NUM_COLORS)}


def test_buffer_wraps_and_samples_the_new_rollout():
    """Rollouts overwrite the oldest trajectories, and the last update's batch without replay holds only the rollout stored last."""
    cfg = replace(TINY, buffer_size=5 * TINY.num_steps, replay=0.0, batch=8)  # room for 5 trajectories, 2 per rollout
    env, _ = make("Atari-breakout")
    nets = make_nets(cfg, env, TINY_SPEC)
    buffer = init_buffer(nets, cfg)
    assert buffer.data.mu.shape[-1] == env.num_actions
    for k in range(3):  # rollout k into slots 2k and 2k + 1: rollout 2 into slots 4 and 0, over rollout 0's first
        rollout = jax.tree.map(lambda x: x[:, : cfg.num_envs], buffer.data)._replace(reward=jnp.full((cfg.num_steps, cfg.num_envs), k, jnp.float32))
        buffer = store(nets, buffer, rollout)
    assert buffer.size == 5 and buffer.idx == 1
    assert jnp.array_equal(buffer.data.reward[0], jnp.array([2, 0, 1, 1, 2], jnp.float32))
    batch, ready = sample(nets, cfg, buffer, jax.random.key(0))
    assert batch.obs.shape[:2] == (cfg.num_steps + 1, 8) and batch.reward.shape == (cfg.num_steps, 8) and not ready
    assert (batch.reward == 2).all(), "the new share must come from the rollout stored last"
    batch, _ = sample(nets, replace(cfg, replay=1.0), buffer, jax.random.key(0))
    assert set(batch.reward[0].tolist()) == {0.0, 1.0, 2.0}, "the replay share must come from the whole buffer"


def test_packed_replay_matches_unpacked_after_wrap_and_reset():
    cfg = replace(TINY, buffer_size=5 * TINY.num_steps)
    nets = make_nets(cfg, make("Arc-wk01")[0], TINY_SPECS["Arc-wk01"])
    raw_nets = nets._replace(colors=0)
    packed, raw = init_buffer(nets, cfg), init_buffer(raw_nets, cfg)
    assert packed.data.obs.nbytes * 2 == raw.data.obs.nbytes
    # Every possible pair of 4-bit colors, including zeros and 15s.
    bytes_ = jnp.arange(256, dtype=jnp.uint8).reshape(2, 1, 8, 16, 1)
    assert jnp.array_equal(pack_obs(unpack_obs(bytes_)), bytes_)
    for k in range(6):
        if k == 3:  # A lifetime reset leaves stale storage behind.
            packed = packed._replace(idx=jnp.int32(0), size=jnp.int32(0))
            raw = raw._replace(idx=jnp.int32(0), size=jnp.int32(0))
        rollout = jax.tree.map(lambda x: x[:, :cfg.num_envs], raw.data)
        obs = (jnp.arange(rollout.obs.size).reshape(rollout.obs.shape) + k) % 16
        rollout = rollout._replace(obs=obs.astype(jnp.uint8), reward=jnp.full_like(rollout.reward, k))
        packed = jax.jit(partial(store, nets))(packed, rollout)
        raw = store(raw_nets, raw, rollout)
        assert jnp.array_equal(unpack_obs(packed.data.obs), raw.data.obs)
        for replay in (0.0, 0.9, 1.0):
            sample_cfg = replace(cfg, replay=replay)
            actual = jax.jit(partial(sample, nets, sample_cfg))(packed, jax.random.key(k))
            expected = sample(raw_nets, sample_cfg, raw, jax.random.key(k))
            assert all(jnp.array_equal(a, b) for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)))


def test_critic_learns_without_changing_actor_features():
    nets = make_nets(TINY, make("Atari-breakout")[0], TINY_SPEC)
    _, critic = nnx.split(nets.new_critic(rngs=nnx.Rngs(0)))
    assert sum(x.size for x in jax.tree.leaves(critic)) == TINY_SPEC.hidden + 1
    features = jax.random.normal(jax.random.key(1), (TINY.num_steps + 1, TINY.num_envs, TINY_SPEC.hidden))
    assert critic_values(nets, critic, features).shape == features.shape[:2]

    def loss(critic, features):
        return ((critic_values(nets, critic, features) - 1.0) ** 2).mean()

    critic_grad, feature_grad = jax.grad(loss, argnums=(0, 1))(critic, features)
    assert jnp.all(feature_grad == 0), "the value loss must not train the actor's representation"
    assert all(jnp.isfinite(g).all() for g in jax.tree.leaves(critic_grad))
    assert any(jnp.any(g != 0) for g in jax.tree.leaves(critic_grad)), "the critic head must receive gradients"


@pytest.mark.skipif(jax.local_devices()[0].platform != "gpu", reason="arcgpu requires CUDA")
def test_meta_steps_run():
    """Two games with different action counts and observation shapes, tiny sizes, two meta steps: the meta-network must
    change and stay finite."""
    before = jax.tree.leaves(nnx.state(train(replace(TINY, meta_steps=0), TINY_SPECS)))  # the same seed gives the same initial meta-network
    after = jax.tree.leaves(nnx.state(train(TINY, TINY_SPECS)))
    assert all(jnp.isfinite(a).all() for a in after)
    assert not all(jnp.allclose(b, a) for b, a in zip(before, after)), "the meta-network did not change"


def test_resume_restores_meta_network():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = replace(TINY, games=("Atari-breakout",), meta_steps=1, save_interval=1, checkpoint_dir=tmp, run_name="r")
        trained = jax.tree.leaves(nnx.state(train(cfg, TINY_SPECS)))
        resumed = jax.tree.leaves(nnx.state(train(replace(cfg, meta_steps=0, resume=f"{tmp}/r/1"), TINY_SPECS)))
    assert all(jnp.array_equal(a, b) for a, b in zip(trained, resumed)), "resume did not restore the trained meta-network"


if __name__ == "__main__":
    test_buffer_wraps_and_samples_the_new_rollout()
    test_packed_replay_matches_unpacked_after_wrap_and_reset()
    test_critic_learns_without_changing_actor_features()
    test_meta_steps_run()
    test_resume_restores_meta_network()
    print("ok")
