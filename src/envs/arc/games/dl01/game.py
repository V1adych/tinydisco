"""The community ARC-AGI-3 game dl01 (github.com/theredbluepill/arc-interactive) in JAX, frame-exact to the original Python game.
Delay line: the player (one cell) walks a 12x12 grid around walls and hazards (both just block); a move (actions 1-4) is
queued and runs one action later, action 5 drops the queued move. Stepping onto the goal passes the level, 5 levels;
there is no way to lose. The original's queue holds up to 3 moves, but every action first runs the oldest, so it never
holds more than one. The camera scales the grid up to the 64x64 frame; along the bottom, three queue slots (the first
shows the queued move as an arrow) and a red mark for 6 rendered frames after a queued move is blocked. The arrays come
from extract.py; a frame here is the last frame the original renders for an action (a level pass ends on the next
level's first frame).
shortcut: the original's red-mark countdown lives in its UI object and survives a game restart, ours restarts with the
episode; upgrade if the base class gets a way to carry state across episodes."""

from pathlib import Path

import jax
import numpy as np
from flax import struct
from jax import numpy as jnp

from ..base import SIZE, ArcGame

DATA = dict(np.load(Path(__file__).with_name("dl01.npz")))
NUM_LEVELS, GRID = DATA["walls"].shape[:2]
MOVES = np.array([(0, -1), (0, 1), (-1, 0), (1, 0)])  # the original's actions 1-4: up, down, left, right, as (dx, dy)
PLAYER, GOAL, WALL, HAZARD = (int(c) for c in DATA["colors"])
BACKGROUND, PADDING = int(DATA["background"]), int(DATA["padding"])
ARROW = np.array([10, 6, 11, 9])  # the UI's color per queued move
ARROW_TIP = np.array([(-1, 0), (1, 0), (0, -1), (0, 1)])  # (row, column) from the slot's pixel
SLOTS, EMPTY, BUMP, BUMP_FRAMES = (2, 7, 12), 3, 8, 6


@struct.dataclass
class State:
    level: jax.Array
    player: jax.Array  # (2,) x, y on the grid
    queued: jax.Array  # the queued move 0-3, -1 for none
    bump: jax.Array  # the UI's red-mark countdown as the last frame showed it (it counts down per rendered frame)


def level_data(name: str, level: jax.Array) -> jax.Array:
    return jnp.asarray(DATA[name])[level]


class Dl01(ArcGame):
    num_actions = 5
    num_levels = NUM_LEVELS

    def init_level(self, level: jax.Array, bump: jax.Array = 0) -> State:
        return State(level, level_data("start", level), jnp.int32(-1), jnp.int32(bump))

    def act(self, state: State, action: jax.Array) -> tuple[State, jax.Array, jax.Array, jax.Array]:
        size = level_data("grid_size", state.level)
        blocked = level_data("walls", state.level) | level_data("hazards", state.level)
        runs = (action < 4) & (state.queued >= 0)
        target = state.player + jnp.asarray(MOVES)[jnp.maximum(state.queued, 0)]
        inside = ((target >= 0) & (target < size)).all()
        free = inside & ~blocked[jnp.clip(target[1], 0, GRID - 1), jnp.clip(target[0], 0, GRID - 1)]
        moved, bumped = runs & free, runs & ~free
        player = jnp.where(moved, target, state.player)
        advanced = moved & (player == level_data("goal", state.level)).all()
        bump = jnp.where(bumped, BUMP_FRAMES, jnp.maximum(state.bump - 1, 0))
        after = State(state.level, player, jnp.where(action < 4, action, -1), bump)
        won = advanced & (state.level == NUM_LEVELS - 1)
        # a level pass renders twice: the passing step, then the next level's first frame
        following = self.init_level(jnp.minimum(state.level + 1, NUM_LEVELS - 1), jnp.maximum(bump - 1, 0))
        after = jax.tree.map(lambda a, b: jnp.where(advanced & ~won, a, b), following, after)
        return after, advanced, jnp.bool_(False), won

    def render(self, state: State) -> jax.Array:
        # the level's sprite order is the draw order: player, goal, walls, hazards
        grid = jnp.full((GRID, GRID), BACKGROUND, jnp.int32).at[state.player[1], state.player[0]].set(PLAYER)
        goal = level_data("goal", state.level)
        grid = grid.at[goal[1], goal[0]].set(GOAL)
        grid = jnp.where(level_data("walls", state.level), WALL, grid)
        grid = jnp.where(level_data("hazards", state.level), HAZARD, grid)
        cells = level_data("cells", state.level)
        frame = jnp.where(cells >= 0, grid.reshape(-1)[cells], PADDING)
        for x in SLOTS[1:]:
            frame = frame.at[SIZE - 2, x].set(EMPTY)
        queued = jnp.maximum(state.queued, 0)
        color = jnp.where(state.queued >= 0, jnp.asarray(ARROW)[queued], EMPTY)
        tip = jnp.asarray([SIZE - 2, SLOTS[0]]) + jnp.where(state.queued >= 0, jnp.asarray(ARROW_TIP)[queued], 0)
        frame = frame.at[SIZE - 2, SLOTS[0]].set(color).at[tip[0], tip[1]].set(color)
        return frame.at[SIZE - 2, SIZE - 3].set(jnp.where(state.bump > 0, BUMP, frame[SIZE - 2, SIZE - 3]))
