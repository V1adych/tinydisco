"""LPG's delayed chain MDPs (Oh et al. 2020, appendix A.3) as a gymnax environment."""

import jax
from flax import struct
from gymnax.environments import environment, spaces
from jax import numpy as jnp

# variant name -> DelayedChain arguments
CHAINS = {
    "short": dict(lengths=(5, 30)),
    "short_noisy": dict(lengths=(5, 30), noisy=True),
    "long": dict(lengths=(5, 50)),
    "long_noisy": dict(lengths=(5, 50), noisy=True),
    "distraction": dict(lengths=(5, 30), distractor_bits=20),
}


@struct.dataclass
class ChainState(environment.EnvState):
    correct: jax.Array  # the action that pays +1 when the chain ends
    chose_correctly: jax.Array


@struct.dataclass
class ChainParams(environment.EnvParams):
    chain_length: int = 30
    max_steps_in_episode: int = 1000  # the chain always ends first


class DelayedChain(environment.Environment[ChainState, ChainParams]):
    """Two actions: the first one of an episode decides the +1/-1 paid when the chain ends, later actions do nothing.
    Noisy chains pay random +1/-1 on the steps in between. The observation is a one-hot state index,
    or with distractor_bits: [a0 is correct, chose correctly] followed by that many random bits."""

    def __init__(self, lengths: tuple[int, int], noisy: bool = False, distractor_bits: int = 0):
        super().__init__()
        self.lengths, self.noisy, self.distractor_bits = lengths, noisy, distractor_bits

    @property
    def default_params(self) -> ChainParams:
        return ChainParams(chain_length=self.lengths[1])

    def sample_params(self, key: jax.Array) -> ChainParams:
        """The chain length is drawn once per agent lifetime and stays fixed within it."""
        return ChainParams(chain_length=jax.random.randint(key, (), self.lengths[0], self.lengths[1] + 1))

    def step_env(self, key, state: ChainState, action, params: ChainParams):
        key_noise, key_obs = jax.random.split(key)
        chose_correctly = jnp.where(state.time == 0, action == state.correct, state.chose_correctly)
        state = state.replace(time=state.time + 1, chose_correctly=chose_correctly)
        end = state.time >= params.chain_length
        between = 2.0 * jax.random.bernoulli(key_noise) - 1 if self.noisy else 0.0
        reward = jnp.where(end, 2.0 * chose_correctly - 1, between)
        return self.get_obs(state, params, key_obs), state, reward, end, {}

    def reset_env(self, key, params: ChainParams):
        key_correct, key_obs = jax.random.split(key)
        state = ChainState(time=jnp.int32(0), correct=jax.random.randint(key_correct, (), 0, 2), chose_correctly=jnp.array(False))
        return self.get_obs(state, params, key_obs), state

    def get_obs(self, state: ChainState, params=None, key=None):
        if self.distractor_bits:
            relevant = jnp.stack([state.correct == 0, state.chose_correctly])
            return jnp.concatenate([relevant, jax.random.bernoulli(key, shape=(self.distractor_bits,))]).astype(jnp.float32)
        # every state is distinct: step x which action is correct x whether the agent chose it
        return jax.nn.one_hot(state.time * 4 + state.correct * 2 + state.chose_correctly, 4 * (self.lengths[1] + 1))

    @property
    def num_actions(self) -> int:
        return 2

    def action_space(self, params=None):
        return spaces.Discrete(2)

    def observation_space(self, params=None):
        return spaces.Box(0, 1, (2 + self.distractor_bits,) if self.distractor_bits else (4 * (self.lengths[1] + 1),))
