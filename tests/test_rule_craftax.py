"""Frozen-rule transfer uses the same replay batch and agent update as Disco's inner learner."""

from dataclasses import replace
from functools import partial

import jax
import numpy as np
from flax import nnx
from jax import numpy as jnp

from modules.checkpoint import load_checkpoint, save_checkpoint
from modules.envs import make
from modules.nn import MetaNetwork
from scripts.disco import Spec, act, init_buffer, init_envs, learn, make_nets, sample, store
from scripts.rule_craftax import Config, train, update


def assert_same(a, b):
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        np.testing.assert_allclose(x, y, rtol=1e-5, atol=1e-6)


def test_craftax_warmup_then_same_update_as_disco():
    cfg = Config("unused", num_envs=1, num_steps=2, batch=2, buffer_size=8, d=4, meta_hidden=8, hidden=8, convs=((7, 4),))
    learner = cfg.learner_config()
    env, _ = make(cfg.env_name)
    nets = make_nets(learner, env, Spec(cfg.convs, cfg.hidden, cfg.lr, (1,)))
    assert nets.obs_space.shape == (63, 63, 3) and nets.num_actions == 17
    _, params = nnx.split(nets.new_agent(rngs=nnx.Rngs(1)))
    opt_state = nets.agent_tx.init(params)
    meta_graphdef, meta_params = nnx.split(MetaNetwork(cfg.d, cfg.meta_hidden, nnx.Rngs(2)))
    frozen = jax.device_get(meta_params)
    buffer = init_buffer(nets, learner)
    envs = jax.jit(partial(init_envs, nets, env, learner))(jax.random.key(3))
    # End a real Craftax episode during collection, with an achievement to verify terminal logging.
    envs = envs._replace(state=envs.state.replace(
        timestep=jnp.full_like(envs.state.timestep, env.default_params.max_timesteps - 1),
        achievements=envs.state.achievements.at[:, 0].set(True),
    ))
    step = jax.jit(partial(update, nets, learner, meta_graphdef, env))
    warm_params, warm_opt, buffer, envs, metrics, stats = step(meta_params, params, opt_state, buffer, envs, jax.random.key(4))
    assert not metrics["ready"]
    assert_same((warm_params, warm_opt), (params, opt_state))
    assert stats["episodes"] == 1
    assert max(float(v) for k, v in stats.items() if k.startswith("achievements/")) == 100

    key = jax.random.key(5)
    key_act, key_sample = jax.random.split(key)
    rollout, _, _ = jax.jit(partial(act, nets, learner, env))(params, envs, key_act)
    batch, ready = sample(nets, learner, store(nets, buffer, rollout), key_sample)
    assert ready
    np.testing.assert_allclose(batch.mu.sum(-1), 1, atol=1e-6)
    expected = jax.jit(partial(learn, nets, learner, meta_graphdef))(meta_params, params, opt_state, batch)
    actual = step(meta_params, params, opt_state, buffer, envs, key)
    assert actual[4]["ready"]
    assert_same(actual[:2], expected[:2])
    assert any(not np.array_equal(a, b) for a, b in zip(jax.tree.leaves(params), jax.tree.leaves(actual[0])))
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(actual[0]))
    assert_same(meta_params, frozen)


def test_train_loads_rule_and_saves_agent(tmp_path):
    rule = MetaNetwork(4, 8, nnx.Rngs(0))
    frozen = jax.device_get(nnx.state(rule))
    path = save_checkpoint(str(tmp_path / "rule"), 1, rule)
    cfg = Config(
        str(path), env_name="Chain-short", convs=(), hidden=8, d=4, meta_hidden=8,
        num_envs=1, num_steps=2, batch=2, buffer_size=8, total_timesteps=6,
        log_interval=1, save_interval=3, checkpoint_dir=str(tmp_path), run_name="agent",
    )
    initial = nnx.state(train(replace(cfg, total_timesteps=0)))
    trained = train(cfg)
    after = jax.device_get(nnx.state(trained))
    assert any(not np.array_equal(a, b) for a, b in zip(jax.tree.leaves(initial), jax.tree.leaves(after)))
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(after))
    load_checkpoint(str(tmp_path / "agent" / "6"), trained)
    assert_same(nnx.state(trained), after)
    load_checkpoint(path, rule)
    assert_same(nnx.state(rule), frozen)
