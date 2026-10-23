"""Every environment by name, behind one interface.

    env, params = make(name)
    obs, state = env.reset(key, params)
    obs, state, reward, done, info = env.step(key, state, action, params)   # auto-resets when done

Names: 'Chain-<variant>' (envs.chain.CHAINS), 'Grid-<variant>' or 'Grid-<variant>-collect' for the 18-action set (envs.grid.GRIDS),
Craftax names such as 'Craftax-Classic-Pixels-v1', and gymnax names such as 'Asterix-MinAtar'.
env.num_actions, env.observation_space(params) and env.action_space(params) work for all of them.
"""

import gymnax
from craftax.craftax_env import make_craftax_env_from_name

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


def make(name: str):
    """Returns (env, default params); see the module docstring for names."""
    kind, _, variant = name.partition("-")
    if kind == "Chain":
        env = CraftaxStep(DelayedChain(**CHAINS[variant]))
    elif kind == "Grid":
        env = CraftaxStep(GridWorld(**GRIDS[variant.removesuffix("-collect")], collect_actions=variant.endswith("-collect")))
    elif kind == "Craftax":
        env = make_craftax_env_from_name(name, auto_reset=True)
    else:
        env, _ = gymnax.make(name)
        env = CraftaxStep(env)
    return env, env.default_params
