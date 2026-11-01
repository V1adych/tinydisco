"""LPG's random grid worlds (Oh et al. 2020, appendix A.2) as a gymnax environment.
LPG's tabular grid worlds are skipped: they are these with the objects pinned for a whole lifetime."""

import jax
import numpy as np
from flax import struct
from gymnax.environments import environment, spaces
from jax import numpy as jnp

# variant name -> GridWorld arguments; objects are (count, reward, p_terminate, p_respawn) per kind
GRIDS = {
    "dense": dict(size=(11, 11), objects=[(2, 1, 0, 0.05), (1, -1, 0.5, 0.1), (1, -1, 0, 0.5)], max_steps=500),
    "long_horizon": dict(size=(11, 11), objects=[(2, 1, 0, 0.01), (2, -1, 0.5, 1)], max_steps=1000),
    "small": dict(size=(5, 7), objects=[(2, 1, 0, 0.05), (2, -1, 0.5, 0.1)], max_steps=500),
    "small_sparse": dict(size=(5, 7), objects=[(1, 1, 1, 1), (2, -1, 1, 1)], max_steps=50),
    "very_dense": dict(size=(11, 11), objects=[(1, 1, 0, 1)], max_steps=2000),
}
# 8 neighbours and staying put, as (row, column) offsets
MOVES = jnp.array([(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)])


@struct.dataclass
class GridState(environment.EnvState):
    agent: jax.Array  # (2,) row, column
    objects: jax.Array  # (K, 2)
    active: jax.Array  # (K,) whether each object is on the board
    score: jax.Array  # return of the episode so far, in info every step: it is the episode's score on the step it ends


class GridWorld(environment.Environment[GridState, environment.EnvParams]):
    """Collecting an object pays its reward, ends the episode with p_terminate, and removes it;
    each step a removed object reappears at a random cell with p_respawn.
    9 actions move and collect by stepping on an object; with collect_actions 9 more collect from a neighbouring cell instead.
    The observation is (H, W, 1 + kinds) planes, the agent's first."""

    def __init__(self, size: tuple[int, int], objects: list[tuple[int, float, float, float]], max_steps: int, collect_actions: bool = False):
        super().__init__()
        self.size, self.max_steps, self.collect_actions = size, max_steps, collect_actions
        counts = [o[0] for o in objects]
        self.kind = np.repeat(np.arange(len(objects)), counts)
        self.reward, self.p_terminate, self.p_respawn = np.repeat(np.array([o[1:] for o in objects], np.float32), counts, axis=0).T

    @property
    def default_params(self) -> environment.EnvParams:
        return environment.EnvParams(max_steps_in_episode=self.max_steps)

    def cells(self, key: jax.Array, n: int, replace: bool) -> jax.Array:
        index = jax.random.choice(key, self.size[0] * self.size[1], (n,), replace=replace)
        return jnp.stack([index // self.size[1], index % self.size[1]], axis=-1)

    def step_env(self, key, state: GridState, action, params):
        key_terminate, key_respawn, key_cells = jax.random.split(key, 3)
        target = state.agent + MOVES[action % 9]
        at_target = (state.objects == target).all(-1)
        if self.collect_actions:
            agent = jnp.where(action < 9, jnp.clip(target, 0, np.array(self.size) - 1), state.agent)
            collected = state.active & at_target & (action >= 9)
        else:
            agent = jnp.clip(target, 0, np.array(self.size) - 1)
            collected = state.active & (state.objects == agent).all(-1)
        reward = (collected * self.reward).sum()
        terminated = (collected & jax.random.bernoulli(key_terminate, self.p_terminate)).any()
        respawn = ~state.active & jax.random.bernoulli(key_respawn, self.p_respawn)
        # shortcut: a respawned object may land on the agent or another object, as nothing in LPG's appendix rules it out
        objects = jnp.where(respawn[:, None], self.cells(key_cells, len(self.kind), replace=True), state.objects)
        active = (state.active & ~collected) | respawn
        state = GridState(time=state.time + 1, agent=agent, objects=objects, active=active, score=state.score + reward)
        return self.get_obs(state), state, reward, terminated, {"score": state.score}

    def reset_env(self, key, params):
        cells = self.cells(key, len(self.kind) + 1, replace=False)
        state = GridState(time=jnp.int32(0), agent=cells[0], objects=cells[1:], active=jnp.ones(len(self.kind), bool), score=jnp.float32(0))
        return self.get_obs(state), state

    def get_obs(self, state: GridState, params=None, key=None):
        obs = jnp.zeros(self.observation_space().shape).at[state.agent[0], state.agent[1], 0].set(1)
        return obs.at[state.objects[:, 0], state.objects[:, 1], 1 + self.kind].max(state.active.astype(jnp.float32))

    @property
    def num_actions(self) -> int:
        return 18 if self.collect_actions else 9

    def action_space(self, params=None):
        return spaces.Discrete(self.num_actions)

    def observation_space(self, params=None):
        return spaces.Box(0, 1, (*self.size, 1 + self.kind.max() + 1))
