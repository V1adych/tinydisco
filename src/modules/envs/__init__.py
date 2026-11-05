"""Every environment by name, behind one interface.

    env, params = make(name)
    obs, state = env.reset(key, params)
    obs, state, reward, done, info = env.step(key, state, action, params)   # auto-resets when done

Names: 'Chain-<variant>' (envs.chain.CHAINS), 'Grid-<variant>' or 'Grid-<variant>-collect' for the 18-action set (envs.grid.GRIDS),
Craftax names such as 'Craftax-Classic-Pixels-v1', 'Atari-<game>' with a jaxtari game (envs.atari.TRAIN, HOLDOUT),
'Arc-<game>' with a no-click arcgpu game (call env.prepare(num_envs, agents, seed) before compiling),
and gymnax names such as 'Asterix-MinAtar'. ARC uses stateful CUDA calls; other environments are pure JAX.
env.num_actions, env.observation_space(params) and env.action_space(params) work for all of them.
Observations are uint8 (0-255 for [0, 1]); the models scale them back.
"""

import gymnax
from craftax.craftax_env import make_craftax_env_from_name
from gymnax.environments import spaces
from jax import numpy as jnp

from .arc import Arc
from .atari import Atari
from .chain import CHAINS, DelayedChain
from .grid import GRIDS, GridWorld


class CraftaxStep:
    """A gymnax env with Craftax's step signature: done = terminated | truncated, the two kept apart in info,
    with info["final_observation"] the observation before any auto-reset."""

    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, key, state, action, params=None):
        obs, state, reward, terminated, truncated, info = self.env.step(key, state, action, params)
        return obs, state, reward, terminated | truncated, info


class Uint8:
    """Observations in [0, 1] as uint8 0-255: a quarter of the memory in rollouts, and the models take only uint8."""

    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)

    def observation_space(self, params=None):
        return spaces.Box(0, 255, self.env.observation_space(params).shape, jnp.uint8)

    def reset(self, key, params=None):
        obs, state = self.env.reset(key, params)
        return to_uint8(obs), state

    def step(self, key, state, action, params=None):
        obs, state, reward, done, info = self.env.step(key, state, action, params)
        if "final_observation" in info:
            info = info | {"final_observation": to_uint8(info["final_observation"])}
        return to_uint8(obs), state, reward, done, info


def to_uint8(obs):
    return jnp.round(obs * 255).astype(jnp.uint8)


def make(name: str):
    """Returns (env, default params); see the module docstring for names."""
    kind, _, variant = name.partition("-")
    if kind == "Chain":
        env = CraftaxStep(DelayedChain(**CHAINS[variant]))
    elif kind == "Grid":
        env = CraftaxStep(GridWorld(**GRIDS[variant.removesuffix("-collect")], collect_actions=variant.endswith("-collect")))
    elif kind == "Atari":
        return Atari(variant), None  # already uint8
    elif kind == "Arc":
        return Arc(variant), None  # already uint8
    elif kind == "Craftax":
        env = make_craftax_env_from_name(name, auto_reset=True)
    else:
        env, _ = gymnax.make(name)
        env = CraftaxStep(env)
    env = Uint8(env)
    return env, env.default_params
