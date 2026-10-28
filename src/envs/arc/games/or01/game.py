"""The community ARC-AGI-3 game or01 (github.com/theredbluepill/arc-interactive) in JAX, frame-exact to the original Python game.
Orbit keys: on a 10x10 grid, keys sit on the 8-cell ring around a pillar and move one slot clockwise on every action; the
player picks up one key per action when orthogonally next to it after moving, and reaching the goal with every key passes
the level, 5 levels. There is no way to lose. The camera scales the grid up to the 64x64 frame, drawing keys and the goal
over the player; ticks near the bottom-left count the keys held, and a pickup flashes a patch bottom-right for 6 rendered
frames (a level pass renders two). The arrays come from extract.py; a frame here is the last frame the original renders for
an action (a level pass ends on the next level's first frame).
shortcut: the original's flash carries over a game reset (its UI object outlives it), here a reset starts without it; upgrade
if episodes must chain the way one original instance does (test_arc.py's guide never wins with the flash still running)."""

from pathlib import Path

import jax
import numpy as np
from flax import struct
from jax import numpy as jnp

from ..base import SIZE, ArcGame

DATA = dict(np.load(Path(__file__).with_name("or01.npz")))
NUM_LEVELS, GRID = DATA["walls"].shape[:2]
MOVES = np.array([(0, -1), (0, 1), (-1, 0), (1, 0)])  # the original's actions 1-4: up, down, left, right, as (dx, dy)
PLAYER, PILLAR, KEY, GOAL, WALL = (int(c) for c in DATA["colors"])
BACKGROUND, PADDING = int(DATA["background"]), int(DATA["padding"])
HELD, EMPTY, MAX_TICKS, FLASH_FRAMES, FLASH_EVEN, FLASH_ODD = 11, 5, 4, 6, 0, 11  # Or01UI


@struct.dataclass
class State:
    level: jax.Array
    player: jax.Array  # (2,) x, y on the grid
    slots: jax.Array  # (K,) each key's ring slot
    alive: jax.Array  # (K,) keys not yet picked up
    flash: jax.Array  # the pickup flash counter as this frame renders it


def level_data(name: str, level: jax.Array) -> jax.Array:
    return jnp.asarray(DATA[name])[level]


def key_cells(state: State) -> jax.Array:
    return level_data("pillar", state.level) + jnp.asarray(DATA["ring"])[state.slots]


class Or01(ArcGame):
    num_levels = NUM_LEVELS

    def init_level(self, level: jax.Array, flash=jnp.int32(0)) -> State:
        return State(level, level_data("start", level), level_data("slots", level), level_data("real", level), flash)

    def act(self, state: State, action: jax.Array) -> tuple[State, jax.Array, jax.Array, jax.Array]:
        state = state.replace(slots=(state.slots + 1) % len(DATA["ring"]))
        target = state.player + jnp.asarray(MOVES)[action]
        inside = ((target >= 0) & (target < level_data("grid_size", state.level))).all()
        wall = level_data("walls", state.level)[jnp.clip(target[1], 0, GRID - 1), jnp.clip(target[0], 0, GRID - 1)]
        moved = inside & ~wall & (target != level_data("pillar", state.level)).any()
        player = jnp.where(moved, target, state.player)
        near = moved & state.alive & (jnp.abs(key_cells(state) - player).sum(-1) == 1)
        picked = near & (jnp.arange(near.shape[0]) == jnp.argmax(near))  # the first such key in list order
        alive = state.alive & ~picked
        flash = jnp.where(picked.any(), FLASH_FRAMES, jnp.maximum(state.flash - 1, 0))
        advanced = moved & ~alive.any() & (player == level_data("goal", state.level)).all()
        after = State(state.level, player, state.slots, alive, flash)
        won = advanced & (state.level == NUM_LEVELS - 1)
        # a level pass renders the old level once more, so the new level's first frame shows the flash one count lower
        fresh = self.init_level(jnp.minimum(state.level + 1, NUM_LEVELS - 1), jnp.maximum(flash - 1, 0))
        after = jax.tree.map(lambda a, b: jnp.where(advanced & ~won, a, b), fresh, after)
        return after, advanced, jnp.bool_(False), won

    def render(self, state: State) -> jax.Array:
        # the level's sprite order is the draw order: player, pillar, keys, goal, walls
        pillar, goal, keys = level_data("pillar", state.level), level_data("goal", state.level), key_cells(state)
        grid = jnp.full((GRID, GRID), BACKGROUND, jnp.int32).at[state.player[1], state.player[0]].set(PLAYER)
        grid = grid.at[pillar[1], pillar[0]].set(PILLAR)
        grid = grid.at[jnp.where(state.alive, keys[:, 1], GRID), keys[:, 0]].set(KEY, mode="drop")
        grid = jnp.where(level_data("walls", state.level), WALL, grid.at[goal[1], goal[0]].set(GOAL))
        cells = level_data("cells", state.level)
        frame = jnp.where(cells >= 0, grid.reshape(-1)[cells], PADDING)
        ticks, real = np.arange(MAX_TICKS), level_data("real", state.level)
        held = (real & ~state.alive).sum()
        frame = frame.at[SIZE - 2, 2 + ticks].set(jnp.where(ticks < real.sum(), jnp.where(ticks < held, HELD, EMPTY), frame[SIZE - 2, 2 + ticks]))
        patch = frame[SIZE - 4 : SIZE - 2, SIZE - 5 : SIZE - 2]
        color = jnp.where(state.flash % 2 == 0, FLASH_EVEN, FLASH_ODD)
        return frame.at[SIZE - 4 : SIZE - 2, SIZE - 5 : SIZE - 2].set(jnp.where(state.flash > 0, color, patch))
