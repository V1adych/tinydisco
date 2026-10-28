"""Self-checks for disco.py: python src/test_disco.py"""

import tempfile
from dataclasses import replace

import jax
from flax import nnx
from jax import numpy as jnp

from disco import Config, init_buffer, make_nets, sample, store, train
from envs import make

TINY = Config(
    games=("Atari-breakout", "Atari-pong"), agents_per_game=2, num_envs=2, num_steps=3, num_updates=3, batch=4, buffer_size=30, valid_rollouts=1,
    lifetimes=(100, 1000), d=4, meta_hidden=8, meta_steps=2, log_interval=1, save_interval=0,
)


def test_buffer_wraps_and_samples_the_new_rollout():
    """Rollouts overwrite the oldest trajectories, and the last update's batch without replay holds only the rollout stored last."""
    cfg = replace(TINY, buffer_size=5 * TINY.num_steps, replay=0.0, batch=8)  # room for 5 trajectories, 2 per rollout
    buffer = init_buffer(make_nets(cfg, [make("Atari-breakout")[0]]), cfg)
    for k in range(3):  # rollout k into slots 2k and 2k + 1: rollout 2 into slots 4 and 0, over rollout 0's first
        rollout = jax.tree.map(lambda x: x[:, : cfg.num_envs], buffer.data)._replace(reward=jnp.full((cfg.num_steps, cfg.num_envs), k, jnp.float32))
        buffer = store(buffer, rollout)
    assert buffer.size == 5 and buffer.idx == 1
    assert jnp.array_equal(buffer.data.reward[0], jnp.array([2, 0, 1, 1, 2], jnp.float32))
    batch, ready = sample(cfg, buffer, jax.random.key(0))
    assert batch.obs.shape[:2] == (cfg.num_steps + 1, 8) and batch.reward.shape == (cfg.num_steps, 8) and not ready
    assert (batch.reward == 2).all(), "the new share must come from the rollout stored last"
    batch, _ = sample(replace(cfg, replay=1.0), buffer, jax.random.key(0))
    assert set(batch.reward[0].tolist()) == {0.0, 1.0, 2.0}, "the replay share must come from the whole buffer"


def test_meta_steps_run():
    """Two games with different action counts, tiny sizes, two meta steps: the meta-network must change and stay finite."""
    before = jax.tree.leaves(nnx.state(train(replace(TINY, meta_steps=0))))  # the same seed gives the same initial meta-network
    after = jax.tree.leaves(nnx.state(train(TINY)))
    assert all(jnp.isfinite(a).all() for a in after)
    assert not all(jnp.allclose(b, a) for b, a in zip(before, after)), "the meta-network did not change"


def test_resume_restores_meta_network():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = replace(TINY, games=("Atari-breakout",), meta_steps=1, save_interval=1, checkpoint_dir=tmp, run_name="r")
        trained = jax.tree.leaves(nnx.state(train(cfg)))
        resumed = jax.tree.leaves(nnx.state(train(replace(cfg, meta_steps=0, resume=f"{tmp}/r/1"))))
    assert all(jnp.array_equal(a, b) for a, b in zip(trained, resumed)), "resume did not restore the trained meta-network"


if __name__ == "__main__":
    test_buffer_wraps_and_samples_the_new_rollout()
    test_meta_steps_run()
    test_resume_restores_meta_network()
    print("ok")
