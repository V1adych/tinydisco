"""The community ARC-AGI-3 game hs01 (github.com/theredbluepill/arc-interactive) in JAX, frame-exact to the original Python game.
Heatseeker: the player (one cell) walks a 10x10 grid to a goal cell while a hunter steps one cell toward it every second
action (along the longer axis first, the other if that is blocked), 5 levels. The original loses when the hunter reaches
the player, but neither can enter the other's cell (each blocks the other), so the game cannot be lost. The camera scales
the grid up to the 64x64 frame and draws in the level's sprite order, so the goal covers a hunter on it; a HUD under the
grid shows the difficulty as ticks and one more tick while the hunter waits. The arrays come from extract.py; a frame here
is the last frame the original renders for an action (a level pass ends on the next level's first frame)."""

from pathlib import Path

import jax
import numpy as np
from flax import struct
from jax import numpy as jnp

from ..base import SIZE, ArcGame

DATA = dict(np.load(Path(__file__).with_name("hs01.npz")))
NUM_LEVELS, GRID = DATA["walls"].shape[:2]
MOVES = np.array([(0, -1), (0, 1), (-1, 0), (1, 0)])  # the original's actions 1-4: up, down, left, right, as (dx, dy)
PLAYER, HUNTER, GOAL, WALL = (int(c) for c in DATA["colors"])
BACKGROUND, PADDING = int(DATA["background"]), int(DATA["padding"])
HUD = 8  # the HUD's color: difficulty ticks on the frame's second-to-last row, the waiting tick above at column 20


@struct.dataclass
class State:
    level: jax.Array
    player: jax.Array  # (2,) x, y on the grid
    hunter: jax.Array
    counter: jax.Array  # player actions since the hunter's last step (0 or 1), also the HUD's waiting tick


def level_data(name: str, level: jax.Array) -> jax.Array:
    return jnp.asarray(DATA[name])[level]


class Hs01(ArcGame):
    num_levels = NUM_LEVELS

    def init_level(self, level: jax.Array) -> State:
        return State(level, level_data("start", level), level_data("hunter", level), jnp.int32(0))

    def act(self, state: State, action: jax.Array) -> tuple[State, jax.Array, jax.Array, jax.Array]:
        size, walls = level_data("grid_size", state.level), level_data("walls", state.level)
        free = lambda v: ((v >= 0) & (v < size)).all() & ~walls[jnp.clip(v[1], 0, GRID - 1), jnp.clip(v[0], 0, GRID - 1)]
        target = state.player + jnp.asarray(MOVES)[action]
        player = jnp.where(free(target) & ~(target == state.hunter).all(), target, state.player)
        counter = (state.counter + 1) % 2
        d = player - state.hunter
        along_x = jnp.abs(d[0]) >= jnp.abs(d[1])
        x_step, y_step = jnp.array([1, 0]) * jnp.sign(d), jnp.array([0, 1]) * jnp.sign(d)
        first, second = jnp.where(along_x, x_step, y_step), jnp.where(along_x, y_step, x_step)
        ok = lambda step: step.any() & free(state.hunter + step) & ~(state.hunter + step == player).all()
        stepped = jnp.where(ok(first), state.hunter + first, jnp.where(ok(second), state.hunter + second, state.hunter))
        hunter = jnp.where(counter == 0, stepped, state.hunter)
        advanced = (player == level_data("goal", state.level)).all()
        after = State(state.level, player, hunter, counter)
        won = advanced & (state.level == NUM_LEVELS - 1)
        after = jax.tree.map(lambda a, b: jnp.where(advanced & ~won, a, b), self.init_level(jnp.minimum(state.level + 1, NUM_LEVELS - 1)), after)
        return after, advanced, jnp.bool_(False), won

    def render(self, state: State) -> jax.Array:
        # the level's sprite order is the draw order: player, hunter, goal, walls
        grid = jnp.full((GRID, GRID), BACKGROUND, jnp.int32).at[state.player[1], state.player[0]].set(PLAYER)
        grid = grid.at[state.hunter[1], state.hunter[0]].set(HUNTER)
        goal = level_data("goal", state.level)
        grid = jnp.where(level_data("walls", state.level), WALL, grid.at[goal[1], goal[0]].set(GOAL))
        cells = level_data("cells", state.level)
        frame = jnp.where(cells >= 0, grid.reshape(-1)[cells], PADDING)
        ticks = (jnp.arange(SIZE) >= 1) & (jnp.arange(SIZE) <= jnp.minimum(level_data("difficulty", state.level), 12))
        frame = frame.at[SIZE - 2].set(jnp.where(ticks, HUD, frame[SIZE - 2]))
        return frame.at[SIZE - 3, 20].set(jnp.where(state.counter > 0, HUD, frame[SIZE - 3, 20]))
