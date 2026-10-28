"""The community ARC-AGI-3 game sn01 (github.com/theredbluepill/arc-interactive) in JAX, frame-exact to the original Python game.
Snake on a 12x12 grid: the head moves one cell per action, eating a food grows the body by one, and eating every food passes
the level, 5 levels. Walls and the grid's edge only block (despite the game's description); moving the head into the body
(except onto the tail's cell) loses. The camera scales the grid up to the 64x64 frame; a row of ticks near the bottom-left
counts the food left. The arrays come from extract.py; a frame here is the last frame the original renders for an action
(a level pass ends on the next level's first frame)."""

from pathlib import Path

import jax
import numpy as np
from flax import struct
from jax import numpy as jnp

from ..base import SIZE, ArcGame

DATA = dict(np.load(Path(__file__).with_name("sn01.npz")))
NUM_LEVELS, GRID = DATA["walls"].shape[:2]
LENGTH = DATA["foods"].shape[1] + 1  # the longest the snake gets
MOVES = np.array([(0, -1), (0, 1), (-1, 0), (1, 0)])  # the original's actions 1-4: up, down, left, right, as (dx, dy)
HEAD, FOOD, WALL, BODY = (int(c) for c in DATA["colors"])
BACKGROUND, PADDING = int(DATA["background"]), int(DATA["padding"])
TICK, MAX_TICKS = 11, 12  # Sn01UI: one tick per food left, on the second-to-last row from x = 1


@struct.dataclass
class State:
    level: jax.Array
    body: jax.Array  # (LENGTH, 2) x, y, head first; slots from length on are stale
    length: jax.Array
    alive: jax.Array  # (F,) food not yet eaten


def level_data(name: str, level: jax.Array) -> jax.Array:
    return jnp.asarray(DATA[name])[level]


class Sn01(ArcGame):
    num_levels = NUM_LEVELS

    def init_level(self, level: jax.Array) -> State:
        body = jnp.full((LENGTH, 2), GRID, jnp.int32).at[0].set(level_data("start", level))
        return State(level, body, jnp.int32(1), level_data("real", level))

    def act(self, state: State, action: jax.Array) -> tuple[State, jax.Array, jax.Array, jax.Array]:
        target = state.body[0] + jnp.asarray(MOVES)[action]
        inside = ((target >= 0) & (target < level_data("grid_size", state.level))).all()
        wall = level_data("walls", state.level)[jnp.clip(target[1], 0, GRID - 1), jnp.clip(target[0], 0, GRID - 1)]
        free = inside & ~wall
        in_body = ((jnp.arange(LENGTH) < state.length) & (state.body == target).all(-1)).any()
        lost = free & in_body & (target != state.body[state.length - 1]).any()
        moved = free & ~lost
        eaten = moved & state.alive & (level_data("foods", state.level) == target).all(-1)
        alive = state.alive & ~eaten
        body = jnp.where(moved, jnp.concatenate([target[None], state.body[:-1]]), state.body)
        advanced = eaten.any() & ~alive.any()
        after = State(state.level, body, state.length + eaten.any(), alive)
        won = advanced & (state.level == NUM_LEVELS - 1)
        after = jax.tree.map(lambda a, b: jnp.where(advanced & ~won, a, b), self.init_level(jnp.minimum(state.level + 1, NUM_LEVELS - 1)), after)
        return after, advanced, lost, won

    def render(self, state: State) -> jax.Array:
        # draw order: head (the player sprite), foods, walls, then the body sprites (re-added after every move)
        foods = level_data("foods", state.level)
        grid = jnp.full((GRID, GRID), BACKGROUND, jnp.int32).at[state.body[0, 1], state.body[0, 0]].set(HEAD)
        grid = grid.at[jnp.where(state.alive, foods[:, 1], GRID), foods[:, 0]].set(FOOD, mode="drop")
        grid = jnp.where(level_data("walls", state.level), WALL, grid)
        segment = (jnp.arange(LENGTH) >= 1) & (jnp.arange(LENGTH) < state.length)
        grid = grid.at[jnp.where(segment, state.body[:, 1], GRID), state.body[:, 0]].set(BODY, mode="drop")
        cells = level_data("cells", state.level)
        frame = jnp.where(cells >= 0, grid.reshape(-1)[cells], PADDING)
        ticks = np.arange(MAX_TICKS)
        return frame.at[SIZE - 2, 1 + ticks].set(jnp.where(ticks < state.alive.sum(), TICK, frame[SIZE - 2, 1 + ticks]))
