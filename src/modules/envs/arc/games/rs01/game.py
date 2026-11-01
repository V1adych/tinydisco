"""The community ARC-AGI-3 game rs01 (github.com/theredbluepill/arc-interactive) in JAX, frame-exact to the original Python game.
Rule Switcher: the player (one cell) collects colored targets on a small grid (8x8 to 16x16, some walls), 5 levels. The safe
color cycles through the level's list every few counted moves (moves that are not blocked); stepping onto a target of
another color loses, until the cycle has come round once, after which any target may be taken. Taking the last one starts
a 14-action hold (actions ignored) before the next level. The camera scales the grid up to the 64x64 frame; a HUD drawn over
its top 5 rows shows the level, the color cycle (active slot highlighted), whether it has come round, the targets left and
the hold. The arrays come from extract.py; a frame here is the last frame the original renders for an action."""

from pathlib import Path

import jax
import numpy as np
from flax import struct
from jax import numpy as jnp

from ..base import ArcGame

DATA = dict(np.load(Path(__file__).with_name("rs01.npz")))
NUM_LEVELS, GRID = DATA["walls"].shape[:2]
MOVES = np.array([(0, -1), (0, 1), (-1, 0), (1, 0)])  # the original's actions 1-4: up, down, left, right, as (dx, dy)
PLAYER, WALL = (int(c) for c in DATA["colors"])
BACKGROUND, PADDING = int(DATA["background"]), int(DATA["padding"])
WIN_HOLD = 14
# the HUD (Rs01UI): rows 0-4 filled, level square, a 4-wide slot per safe color from x = 6, cycled cue, target ticks, hold line
HUD_ROWS, HUD_FILL, LEVEL_COLORS, SLOT_X, SLOT_W, SLOT_DIM, SLOT_CAP = 5, 5, np.array([9, 10, 11, 12, 14]), 6, 5, 3, 0
CYCLED, TICK, MAX_TICKS = 14, 11, 14


@struct.dataclass
class State:
    level: jax.Array
    player: jax.Array  # (2,) x, y on the grid
    alive: jax.Array  # (T,) targets not yet taken
    steps: jax.Array
    cycle: jax.Array  # index into the safe colors
    all_cycled: jax.Array
    hold: jax.Array  # actions left before the next level


def level_data(name: str, level: jax.Array) -> jax.Array:
    return jnp.asarray(DATA[name])[level]


class Rs01(ArcGame):
    num_levels = NUM_LEVELS

    def init_level(self, level: jax.Array) -> State:
        return State(level, level_data("start", level), level_data("real", level), jnp.int32(0), jnp.int32(0), jnp.bool_(False), jnp.int32(0))

    def act(self, state: State, action: jax.Array) -> tuple[State, jax.Array, jax.Array, jax.Array]:
        holding = state.hold > 0
        target = state.player + jnp.asarray(MOVES)[action]
        inside = ((target >= 0) & (target < level_data("grid_size", state.level))).all()
        hit = state.alive & (level_data("targets", state.level) == target).all(-1)
        first = jnp.argmax(hit)  # the first target in list order (targets come before walls)
        on_target = inside & hit.any()
        wall = level_data("walls", state.level)[jnp.clip(target[1], 0, GRID - 1), jnp.clip(target[0], 0, GRID - 1)]
        safe = level_data("safe_colors", state.level)[state.cycle]
        lost = ~holding & on_target & (level_data("target_colors", state.level)[first] != safe) & ~state.all_cycled
        moved = ~holding & inside & (on_target | ~wall) & ~lost
        alive = state.alive & ~(moved & on_target & (jnp.arange(hit.shape[0]) == first))
        steps = state.steps + moved
        turn = moved & (steps % level_data("cycle_interval", state.level) == 0)
        cycle = jnp.where(turn, (state.cycle + 1) % level_data("num_safe", state.level), state.cycle)
        hold = jnp.where(holding, state.hold - 1, jnp.where(moved & ~alive.any(), WIN_HOLD, 0))
        advanced = holding & (hold == 0)
        after = State(state.level, jnp.where(moved, target, state.player), alive, steps, cycle, state.all_cycled | (turn & (cycle == 0)), hold)
        won = advanced & (state.level == NUM_LEVELS - 1)
        after = jax.tree.map(lambda a, b: jnp.where(advanced & ~won, a, b), self.init_level(jnp.minimum(state.level + 1, NUM_LEVELS - 1)), after)
        return after, advanced, lost, won

    def render(self, state: State) -> jax.Array:
        # the level's sprite order is the draw order: player, targets, walls
        targets = level_data("targets", state.level)
        colors = jnp.where(state.alive, level_data("target_colors", state.level), -1)
        grid = jnp.full((GRID, GRID), BACKGROUND, jnp.int32).at[state.player[1], state.player[0]].set(PLAYER)
        grid = grid.at[jnp.where(state.alive, targets[:, 1], GRID), targets[:, 0]].set(colors, mode="drop")
        grid = jnp.where(level_data("walls", state.level), WALL, grid)
        cells = level_data("cells", state.level)
        frame = jnp.where(cells >= 0, grid.reshape(-1)[cells], PADDING)
        # HUD, as Rs01UI draws it
        difficulty, safe = level_data("difficulty", state.level), level_data("safe_colors", state.level)
        frame = frame.at[:HUD_ROWS].set(HUD_FILL).at[1:3, 1:3].set(jnp.asarray(LEVEL_COLORS)[(difficulty - 1) % len(LEVEL_COLORS)])
        for i in range(safe.shape[0]):
            c, active = safe[i], i == state.cycle
            column = jnp.where(active, jnp.stack([SLOT_CAP, c, c, c]), jnp.stack([HUD_FILL, SLOT_DIM, c, SLOT_DIM]))
            x = SLOT_X + SLOT_W * i
            frame = frame.at[:4, x : x + 4].set(jnp.where(i < level_data("num_safe", state.level), column[:, None], frame[:4, x : x + 4]))
        frame = frame.at[0, 60:62].set(jnp.where(state.all_cycled, CYCLED, HUD_FILL))
        left = state.alive.sum()
        ticks = 62 - np.arange(MAX_TICKS)
        frame = frame.at[2, ticks].set(jnp.where(np.arange(MAX_TICKS) < left, TICK, frame[2, ticks]))
        pulse = jnp.where((difficulty + left) % 2 == 0, CYCLED, TICK)
        return frame.at[4, 4:40].set(jnp.where(state.hold > 0, pulse, HUD_FILL))
