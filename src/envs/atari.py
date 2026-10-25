"""JAXAtari games (JAX ports of Atari 2600) as our envs, set up like ALE v5: sticky actions 0.25, no no-op starts,
frame skip 4 with max-pooling, 84x84 grayscale, 4 stacked frames, rewards clipped to their sign, a life lost is not an
episode end. Frames are stacked here rather than by jaxtari's PixelObsWrapper, which drops the observation before a reset."""

import jax
import jaxtari
from flax import struct
from gymnax.environments import spaces
from jax import numpy as jnp
from jaxtari.wrappers import AtariState, AtariWrapper

# jaxtari's 🥇 games of the Atari 57 except Montezuma's Revenge (no reward without exploration) and the held-out three,
# plus four 🥈 ones (close to the originals, may miss parts); statuses from jaxtari's games_covered.md
TRAIN_GAMES = (
    "amidar", "asterix", "asteroids", "bankheist", "enduro", "fishingderby", "freeway", "frostbite", "gopher", "gravitar",
    "kangaroo", "phoenix", "seaquest", "surround", "tennis", "yarsrevenge", "mspacman", "spaceinvaders", "qbert", "beamrider",
)
HOLDOUT_GAMES = ("breakout", "pong", "boxing")
SIZE, FRAME_SKIP, FRAME_STACK = 84, 4, 4


@struct.dataclass
class StackState:
    atari: AtariState
    obs: jax.Array  # (SIZE, SIZE, FRAME_STACK), oldest frame first
    score: jax.Array  # unclipped return of the episode so far


class Atari:
    """One game behind make()'s interface; jaxtari keeps its random key in the state, so step's key and params are unused."""

    def __init__(self, game: str):
        self.env = AtariWrapper(jaxtari.make(game), episodic_life=False, noop_max=0)
        self.num_actions = self.env.action_space().n
        self.default_params = None

    def observation_space(self, params=None):
        return spaces.Box(0, 255, (SIZE, SIZE, FRAME_STACK), jnp.uint8)

    def action_space(self, params=None):
        return spaces.Discrete(self.num_actions)

    def frame(self, image):
        """A (210, 160, 3) uint8 screen -> a (SIZE, SIZE, 1) uint8 grayscale frame; the models scale it."""
        image = jax.image.resize(image.astype(jnp.float32), (SIZE, SIZE, 3), "bilinear")
        return jnp.round(image @ jnp.array([0.2989, 0.5870, 0.1140]))[..., None].astype(jnp.uint8)

    def reset(self, key, params=None):
        _, atari = self.env.reset(key)
        obs = jnp.repeat(self.frame(self.env.render(atari.env_state)), FRAME_STACK, -1)
        return obs, StackState(atari, obs, jnp.float32(0))

    def step(self, key, state, action, params=None):
        def skip(atari, _):
            _, atari, reward, terminated, truncated, _ = self.env.step(atari, action)
            return atari, (atari.env_state, reward, terminated, truncated)

        atari, (screens, rewards, terminated, truncated) = jax.lax.scan(skip, state.atari, length=FRAME_SKIP)
        # max over the last two screens, as ALE does against sprite flicker
        image = jnp.maximum(*(self.env.render(jax.tree.map(lambda x: x[i], screens)) for i in (-2, -1)))
        obs = jnp.concatenate([state.obs[..., 1:], self.frame(image)], -1)
        reward, terminated, truncated = rewards.sum().astype(jnp.float32), terminated.any(), truncated.any()
        done = terminated | truncated
        score = state.score + reward
        # score is reported on the step an episode ends, like craftax's achievements
        info = {"terminated": terminated, "truncated": truncated, "final_observation": obs, "score": score * done}
        reset_obs, reset_state = self.reset(atari.key)
        obs, state = jax.tree.map(lambda r, x: jnp.where(done, r, x), (reset_obs, reset_state), (obs, StackState(atari, obs, score)))
        return obs, state, jnp.sign(reward), done, info
