"""No-click arcgpu games. CUDA state lives in the simulator, not in the JAX pytree.

Call prepare() before compiling resets/steps, with the total batch layout. Each
vmap adds to that batch; never rematerialize an environment call or branch/fork
its state. Hard resets draw fresh curricula from the engine's seeded generator.
"""

import ctypes
import json
import math
import subprocess
from pathlib import Path
from typing import NamedTuple

import jax
from gymnax.environments import spaces
from jax import numpy as jnp
from jax.custom_batching import custom_vmap

ROOT = Path(__file__).with_name("arcgpu")
LIBRARY = ROOT / "build/libarcgpu_ffi.so"
NUM_COLORS = 16
_METADATA = json.loads((ROOT / "training/arcjax/game_meta.json").read_text())["games"]
ARC_GAMES = {name: meta for name, meta in _METADATA.items() if not meta["has_click"] and 6 not in meta["actions"]}
PROCEDURAL = tuple(name for name, meta in ARC_GAMES.items() if meta.get("procedural"))
OFFICIAL = tuple(name for name in ARC_GAMES if name not in PROCEDURAL)
TRAIN = ("wk01", "dr02", "g014", "g001")
HOLDOUT = tuple(name for name in PROCEDURAL if name not in TRAIN)
_LIB = None


class State(NamedTuple):
    steps: jax.Array
    score: jax.Array
    obs: jax.Array


def _batched_call(target, attrs, output_dtypes, trailing_shapes, input_dims=0, masked=False):
    @custom_vmap
    def call(*args):
        shape = args[0].shape[:args[0].ndim - input_dims]
        n = math.prod(shape)
        inputs = [x.reshape((n,) + x.shape[len(shape):]) for x in args]
        outputs = [jax.ShapeDtypeStruct((n,) + tail, dtype) for tail, dtype in zip(trailing_shapes, output_dtypes)]
        def run():
            return tuple(jax.ffi.ffi_call(target, outputs, has_side_effect=True)(*inputs, **attrs))
        result = jax.lax.cond(jnp.any(inputs[0]), run, lambda: (inputs[1],)) if masked else run()
        return tuple(x.reshape(shape + tail) for x, tail in zip(result, trailing_shapes))

    @call.def_vmap
    def batch(size, batched, *args):
        args = [x if b else jnp.broadcast_to(x, (size,) + x.shape) for x, b in zip(args, batched)]
        out = call(*args)
        return out, (True,) * len(out)

    return call


class Arc:
    def __init__(self, game):
        if game not in ARC_GAMES:
            raise ValueError(f"{game!r} is not a supported no-click ARC game")
        self.game = game
        self.meta = ARC_GAMES[game]
        self.num_actions = len(self.meta["actions"])
        self._raw = self._reset = None

    def observation_space(self, params=None):
        return spaces.Box(0, NUM_COLORS - 1, (64, 64, 1), jnp.uint8)

    def action_space(self, params=None):
        return spaces.Discrete(self.num_actions)

    def prepare(self, num_envs, agents=1, seed=0):
        global _LIB
        if self._raw is not None:
            raise RuntimeError("Create a new Arc object for a new simulator batch")
        if jax.local_devices()[0].platform != "gpu":
            raise RuntimeError("arcgpu needs an NVIDIA GPU; metadata and replay tests can run on CPU")
        if _LIB is None:
            subprocess.run(["bash", str(ROOT / "bindings/jax/build_ffi.sh")], cwd=ROOT, check=True)
            _LIB = ctypes.CDLL(str(LIBRARY))
            for name, symbol in (("tinydisco_arc_step", "TinydiscoArcStep"), ("tinydisco_arc_reset", "TinydiscoArcReset")):
                jax.ffi.register_ffi_target(name, jax.ffi.pycapsule(getattr(_LIB, symbol)), platform="CUDA")
            _LIB.arcgpu_prewarm2.argtypes = [ctypes.c_char_p, ctypes.c_long, ctypes.c_int, ctypes.c_long,
                                            ctypes.c_longlong, ctypes.c_long, ctypes.c_char_p, ctypes.c_long]
            _LIB.arcgpu_prewarm2.restype = ctypes.c_int
        attrs = dict(games=self.game, envs_per_game=num_envs, copies=agents, seed=seed, opts=1)
        error = ctypes.create_string_buffer(1024)
        if _LIB.arcgpu_prewarm2(self.game.encode(), num_envs, 0, agents, seed, 1, error, len(error)):
            raise RuntimeError(error.value.decode())
        # Flatten both vmap axes (agents and environments) into one CUDA call.
        self._raw = _batched_call("tinydisco_arc_step", attrs, (jnp.int8, jnp.float32, jnp.uint8, jnp.uint8), ((64, 64), (), (), ()), input_dims=1)
        self._reset = _batched_call("tinydisco_arc_reset", attrs, (jnp.uint8,), ((64, 64, 1),), masked=True)

    def reset_where(self, state, mask):
        if self._reset is None:
            raise RuntimeError("Call Arc.prepare(num_envs, agents, seed) before stepping")
        obs, = self._reset(mask.astype(jnp.uint8), state.obs)
        return obs, State(jnp.where(mask, 0, state.steps), jnp.where(mask, 0, state.score), obs)

    def reset(self, key, params=None):
        state = State(jnp.int32(0), jnp.int32(0), jnp.zeros((64, 64, 1), jnp.uint8))
        # Keep the key's vmap axes even though the engine owns the seeded PRNG.
        return self.reset_where(state, jax.random.key_data(key)[0] >= jnp.uint32(0))

    def step(self, key, state, action, params=None):
        # Only declared keyboard actions are exposed; reset is controlled by the wrapper.
        rows = jnp.stack((action, jnp.zeros_like(action), jnp.zeros_like(action), jnp.zeros_like(action)), -1).astype(jnp.uint8)
        frame, _, flags, levels = self._raw(rows, state.obs)
        obs = frame.astype(jnp.uint8)[..., None]
        score = jnp.maximum(state.score, levels.astype(jnp.int32))
        reward = (score - state.score).astype(jnp.float32)
        steps = jnp.where(score > state.score, 0, state.steps + 1)
        budgets = jnp.asarray(self.meta["budget_steps"], jnp.int32)
        done = ((flags & 7) != 0) | (steps >= budgets[jnp.minimum(score, len(budgets) - 1)])
        new = State(steps, score, obs)
        next_obs, new = self.reset_where(new, done)
        return next_obs, new, reward, done, {"score": score.astype(jnp.float32), "final_observation": obs}
