"""JAXAtari games (JAX ports of Atari 2600) as our envs, set up like ALE v5: sticky actions 0.25, no no-op starts,
frame skip 4 with max-pooling, 84x84 grayscale, rewards clipped to their sign, a life lost is not an episode end. The
engine draws 84x84 grayscale frames itself (jaxtari's native downscaling). Frames are not stacked: the agents have memory."""

import jax
import jaxtari
from flax import struct
from gymnax.environments import spaces
from jax import numpy as jnp
from jaxtari.modification import apply_native_downscaling
from jaxtari.wrappers import AtariState, AtariWrapper

# 8 of jaxtari's 🥇 games (statuses from jaxtari's games_covered.md): a shooter, a driver, a crosser, a platformer, two
# mazes, a digger and a sub; not Fishing Derby (its port takes ~35 ms a step on a GPU, 10x the others)
TRAIN = ("asterix", "asteroids", "enduro", "freeway", "frostbite", "gopher", "kangaroo", "seaquest")
HOLDOUT = ("breakout", "pong", "boxing")
SIZE, FRAME_SKIP = 84, 4


@struct.dataclass
class State:
    atari: AtariState
    score: jax.Array  # unclipped return of the episode so far


class Atari:
    """One game behind make()'s interface; jaxtari keeps its random key in the state, so step's key and params are unused."""

    def __init__(self, game: str):
        core = jaxtari.make(game)
        apply_native_downscaling(core, (SIZE, SIZE), grayscale=True)  # render() now gives (SIZE, SIZE, 1) uint8
        self.env = AtariWrapper(core, episodic_life=False, noop_max=0)
        self.num_actions = self.env.action_space().n
        self.default_params = None

    def observation_space(self, params=None):
        return spaces.Box(0, 255, (SIZE, SIZE, 1), jnp.uint8)

    def action_space(self, params=None):
        return spaces.Discrete(self.num_actions)

    def reset(self, key, params=None):
        _, atari = self.env.reset(key)
        return self.env.render(atari.env_state), State(atari, jnp.float32(0))

    def step(self, key, state, action, params=None):
        def skip(atari, _):
            _, atari, reward, terminated, truncated, _ = self.env.step(atari, action)
            return atari, (atari.env_state, reward, terminated, truncated)

        atari, (screens, rewards, terminated, truncated) = jax.lax.scan(skip, state.atari, length=FRAME_SKIP)
        # max over the last two frames, as ALE does against sprite flicker
        obs = jnp.maximum(*(self.env.render(jax.tree.map(lambda x: x[i], screens)) for i in (-2, -1)))
        reward, terminated, truncated = rewards.sum().astype(jnp.float32), terminated.any(), truncated.any()
        done = terminated | truncated
        score = state.score + reward
        # score is reported on the step an episode ends, like craftax's achievements
        info = {"terminated": terminated, "truncated": truncated, "final_observation": obs, "score": score * done}
        reset_obs, reset_state = self.reset(atari.key)
        obs, state = jax.tree.map(lambda r, x: jnp.where(done, r, x), (reset_obs, reset_state), (obs, State(atari, score)))
        return obs, state, jnp.sign(reward), done, info
