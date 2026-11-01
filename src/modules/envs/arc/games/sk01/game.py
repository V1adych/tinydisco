"""The community ARC-AGI-3 game sk01 (github.com/theredbluepill/arc-interactive) in JAX, frame-exact to the original Python game.
Sokoban: the player (one cell) pushes crates, one at a time, around walls on a small grid (8x8, one level 12x12); a crate
that has once reached a target stays marked done (recolored) even when pushed off again, and the level passes when every
crate is done, 5 levels with 1-3 crates. Every move that is not blocked counts toward a per-level step limit; reaching it
loses the game. The camera scales the grid up to the 64x64 frame (letterboxed when 64 is no multiple of the grid) and
draws in the level's sprite order, so targets cover the player and crates standing on them; a 4x4 square in the
bottom-right corner turns from orange to green when every crate is done. The arrays come from extract.py; a frame here
is the last frame the original renders for an action (a level pass ends on the next level's first frame)."""

from pathlib import Path

import jax
import numpy as np
from flax import struct
from jax import numpy as jnp

from ..base import SIZE, ArcGame

DATA = dict(np.load(Path(__file__).with_name("sk01.npz")))
NUM_LEVELS, GRID = DATA["walls"].shape[:2]
MOVES = np.array([(0, -1), (0, 1), (-1, 0), (1, 0)])  # the original's actions 1-4: up, down, left, right, as (dx, dy)
PLAYER, CRATE, DONE, TARGET, WALL = (int(c) for c in DATA["colors"])
BACKGROUND, PADDING = int(DATA["background"]), int(DATA["padding"])
UI_SIZE, UI_ALL_DONE, UI_PENDING = 4, 11, 14  # the indicator in the frame's bottom-right corner


@struct.dataclass
class State:
    level: jax.Array
    player: jax.Array  # (2,) x, y on the grid
    crates: jax.Array  # (C, 2) x, y; padding slots sit at (GRID, GRID)
    done: jax.Array  # (C,) has been on a target
    steps: jax.Array


def level_data(name: str, level: jax.Array) -> jax.Array:
    return jnp.asarray(DATA[name])[level]


class Sk01(ArcGame):
    num_levels = NUM_LEVELS

    def init_level(self, level: jax.Array) -> State:
        return State(level, level_data("start", level), level_data("crates", level), jnp.zeros(DATA["crates"].shape[1], bool), jnp.int32(0))

    def act(self, state: State, action: jax.Array) -> tuple[State, jax.Array, jax.Array, jax.Array]:
        size, walls, targets = (level_data(k, state.level) for k in ("grid_size", "walls", "targets"))
        inside = lambda v: ((v >= 0) & (v < size)).all()
        at = lambda grid, v: grid[jnp.clip(v[1], 0, GRID - 1), jnp.clip(v[0], 0, GRID - 1)]
        move = jnp.asarray(MOVES)[action]
        target, pushed = state.player + move, state.player + 2 * move
        hit = (state.crates == target).all(-1)
        push_free = inside(pushed) & ~at(walls, pushed) & ~(state.crates == pushed).all(-1).any()
        moved = inside(target) & ~at(walls, target) & (~hit.any() | push_free)
        pushing = hit & moved
        crates = jnp.where(pushing[:, None], pushed, state.crates)
        done = state.done | (pushing & at(targets, pushed))
        steps = state.steps + moved
        advanced = moved & (done | ~level_data("real", state.level)).all()
        lost = moved & ~advanced & (steps >= level_data("step_limit", state.level))
        after = State(state.level, jnp.where(moved, target, state.player), crates, done, steps)
        won = advanced & (state.level == NUM_LEVELS - 1)
        after = jax.tree.map(lambda a, b: jnp.where(advanced & ~won, a, b), self.init_level(jnp.minimum(state.level + 1, NUM_LEVELS - 1)), after)
        return after, advanced, lost, won

    def render(self, state: State) -> jax.Array:
        # the level's sprite order is the draw order: player, crates, targets, walls
        grid = jnp.full((GRID, GRID), BACKGROUND, jnp.int32).at[state.player[1], state.player[0]].set(PLAYER)
        grid = grid.at[state.crates[:, 1], state.crates[:, 0]].set(jnp.where(state.done, DONE, CRATE), mode="drop")
        grid = jnp.where(level_data("targets", state.level), TARGET, grid)
        grid = jnp.where(level_data("walls", state.level), WALL, grid)
        cells = level_data("cells", state.level)
        frame = jnp.where(cells >= 0, grid.reshape(-1)[cells], PADDING)
        pending = (level_data("real", state.level) & ~state.done).any()
        return frame.at[SIZE - UI_SIZE :, SIZE - UI_SIZE :].set(jnp.where(pending, UI_PENDING, UI_ALL_DONE))
