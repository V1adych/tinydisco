"""The ARC-AGI-3 game wa30 (official games have ids, not names) in JAX, frame-exact to the original Python game.
A warehouse on a 16x16 grid of 4-pixel cells: the player (black side = facing) walks one cell per action; the fifth action
grabs the crate in front (or drops the held one), and a held crate is dragged along. A level ends when every crate sits in a
goal zone and nobody holds it; the step bar at the bottom runs out = game over, 9 levels. Collector bots fetch loose crates
to the goals (helping), thief bots drag crates to sinks (and steal them from whoever holds them); the fifth action on a
thief in front removes it. Walls and checkered tiles block walkers, but a dragged crate may pass over checkered tiles.
Nothing of this is told to the agent: it sees 64x64 pixels of 16 colors. The arrays come from extract.py."""

from pathlib import Path

import jax
import numpy as np
from flax import struct
from jax import numpy as jnp

from ..base import SIZE, ArcGame

DATA = dict(np.load(Path(__file__).with_name("wa30.npz")))
NUM_LEVELS, CELLS = DATA["blocked"].shape[:2]
NUM_CRATES, NUM_COLLECTORS, NUM_THIEVES = (DATA[k].shape[1] for k in ("crates", "collectors", "thieves"))
NUM_HOLDERS = 1 + NUM_COLLECTORS + NUM_THIEVES  # the player, then the collectors, then the thieves
CELL = 4
MOVES = np.array([(0, -1), (0, 1), (-1, 0), (1, 0)])  # the original's actions 1-4: up, down, left, right, in cells; also the facings
GRAB = 4  # our action for the original's action 5
SEARCH_ORDER = np.array([(-1, 0), (1, 0), (0, -1), (0, 1)])  # the neighbours in the order the bots' breadth-first search tries them
FAR = 1 << 20  # a rank no searched cell reaches
ROTATION = (0, 180, 270, 90)  # the player sprite's rotation per facing, rotated as the original's sprite renderer does
PLAYER = np.stack([np.rot90(DATA["player_pixels"], k=(-r % 360) // 90) for r in ROTATION])
CRATE_INNER, COLLECTOR_COLOR, THIEF_COLOR = 9, 12, 15
CRATE_BORDER, CRATE_FACED, CRATE_HELD_BY_PLAYER, CRATE_HELD_BY_BOT = 4, 3, 0, 5
THIEF_BORDER, THIEF_FACED = 15, 11
HUD_ROW, HUD_LEFT, HUD_SPENT = 63, 7, 4
EDGE = np.ones((CELL, CELL), bool)
EDGE[1:-1, 1:-1] = False


@struct.dataclass
class State:
    level: jax.Array
    player: jax.Array  # (2,) column, row on the 18x18 cell grid (a blocked ring around the 16x16 playfield)
    facing: jax.Array  # index into MOVES
    crates: jax.Array  # (C, 2)
    collectors: jax.Array  # (B, 2)
    thieves: jax.Array  # (T, 2); a removed or padded thief sits on the ring cell (0, 0), where nothing can reach it
    held: jax.Array  # (1 + B + T,) the crate each holder drags, -1 for none
    steps: jax.Array


def level_data(name: str, level: jax.Array) -> jax.Array:
    return jnp.asarray(DATA[name])[level]


def at(grid: jax.Array, cells: jax.Array) -> jax.Array:
    return grid[cells[..., 1], cells[..., 0]]


def one_hot(cell: jax.Array) -> jax.Array:
    return jnp.zeros((CELLS, CELLS), bool).at[cell[1], cell[0]].set(True)


def shifted(grid: jax.Array, delta: jax.Array) -> jax.Array:
    """shifted[v] = grid[v + delta]; the ring wraps onto itself, and the ring is never valid."""
    return jnp.roll(grid, (-delta[1], -delta[0]), (0, 1))


def search(valid: jax.Array, start: jax.Array, is_target: jax.Array) -> tuple[jax.Array, jax.Array]:
    """The original's breadth-first search: whether it finds a target other than the start, and the first step of its path.
    Cells get the rank the search pops them in (a cell's parent is its lowest-ranked found neighbour, children are ranked by
    parent then SEARCH_ORDER), so the target found first is the reachable one of lowest rank."""
    rank = jnp.full((CELLS, CELLS), FAR).at[start[1], start[0]].set(0)
    first = jnp.full((CELLS, CELLS), -1)

    def expand(carry):
        rank, first, found, previously = carry
        frontier = (rank >= previously) & (rank < found)
        keys = jnp.stack([jnp.where(shifted(frontier, -d), shifted(rank, -d) * 4 + j, FAR) for j, d in enumerate(SEARCH_ORDER)])
        parent = keys.argmin(0)
        key = keys.min(0)
        new = (key < FAR) & valid & (rank == FAR)
        flat = jnp.where(new, key, FAR).ravel()
        order = (flat[None] < flat[:, None]).sum(1).reshape(CELLS, CELLS)
        inherited = jnp.take_along_axis(jnp.stack([shifted(first, -d) for d in SEARCH_ORDER]), parent[None], 0).squeeze(0)
        first = jnp.where(new, jnp.where(inherited < 0, parent, inherited), first)
        return jnp.where(new, found + order, rank), first, found + new.sum(), found

    rank, first, _, _ = jax.lax.while_loop(lambda c: c[2] > c[3], expand, (rank, first, jnp.int32(1), jnp.int32(0)))
    score = jnp.where(is_target, rank, FAR).ravel()
    best = score.argmin()
    return (score[best] < FAR) & (score[best] > 0), jnp.asarray(SEARCH_ORDER)[first.ravel()[best]]


def can_drag(blocked: jax.Array, checkered: jax.Array, holder: jax.Array, crate: jax.Array, target: jax.Array) -> jax.Array:
    """Holder and crate may both shift so the holder lands on target; swapping into each other's cell is allowed."""
    ahead = target + crate - holder
    return (~at(blocked, target) | (target == crate).all()) & ~at(checkered, target) & (~at(blocked, ahead) | (ahead == holder).all())


def shift_crate(crates: jax.Array, index: jax.Array, delta: jax.Array, do: jax.Array) -> jax.Array:
    return jnp.where((jnp.arange(NUM_CRATES) == index)[:, None] & do, crates + delta, crates)


class Wa30(ArcGame):
    num_actions = 5
    num_levels = NUM_LEVELS

    def init_level(self, level: jax.Array) -> State:
        crates, collectors, thieves = (level_data(k, level) for k in ("crates", "collectors", "thieves"))
        return State(level, level_data("player", level), jnp.int32(0), crates, collectors, thieves, jnp.full(NUM_HOLDERS, -1), level_data("steps", level))

    def blocked(self, state: State) -> jax.Array:
        cells = jnp.concatenate([state.player[None], state.crates, state.collectors, state.thieves])
        return level_data("blocked", state.level).at[cells[:, 1], cells[:, 0]].set(True)

    def bot_turn(self, state: State, holder: jax.Array, zone: jax.Array, wanted: jax.Array, bot: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """One bot's move: drop its crate once it is in the zone, else drag it one step towards the zone; without a crate take an
        adjacent wanted one, else walk one step towards a cell next to one. Returns (drop, take, which crate, move, step)."""
        blocked, checkered = self.blocked(state), level_data("checkered", state.level)
        crate = state.crates[state.held[holder]]
        holding = state.held[holder] >= 0
        offset = crate - bot
        dragging = (~blocked | one_hot(crate)) & ~checkered & (~shifted(blocked, offset) | one_hot(bot - offset))
        adjacent = wanted & (jnp.abs(state.crates - bot).sum(-1) == 1)
        beside = jnp.zeros((CELLS, CELLS), bool)
        for d in SEARCH_ORDER:
            beside = beside.at[state.crates[:, 1] + d[1], state.crates[:, 0] + d[0]].max(wanted, mode="drop")
        found, step = search(jnp.where(holding, dragging, ~blocked & ~checkered), bot, jnp.where(holding, shifted(zone, offset), beside))
        drop = holding & at(zone, crate)
        take = ~holding & adjacent.any()
        move = jnp.where(holding, ~drop, ~take) & found
        return drop, take, adjacent.argmax(), move, step

    def act(self, state: State, action: jax.Array) -> tuple[State, jax.Array, jax.Array, jax.Array]:
        steps = jnp.maximum(state.steps - 1, 0)
        real = jnp.arange(NUM_CRATES) < level_data("num_crates", state.level)
        goal, sink, checkered = (level_data(k, state.level) for k in ("goal", "sink", "checkered"))
        held = state.held
        # the player: a move turns it (unless it drags a crate) and walks if the cell is free
        holding = held[0] >= 0
        delta = jnp.asarray(MOVES)[jnp.minimum(action, GRAB - 1)]
        target = state.player + delta
        facing = jnp.where((action < GRAB) & ~holding, action, state.facing)
        blocked = self.blocked(state)
        free = ~at(blocked, target) & ~at(checkered, target)
        moved = (action < GRAB) & jnp.where(holding, can_drag(blocked, checkered, state.player, state.crates[held[0]], target), free)
        player = jnp.where(moved, target, state.player)
        crates = shift_crate(state.crates, held[0], delta, moved & holding)
        # the grab: drop, or take the crate in front from whoever holds it, or remove the thief in front
        front = player + jnp.asarray(MOVES)[facing]
        crate_front = real & (crates == front).all(-1)
        thief_front = (state.thieves == front).all(-1)
        taking = (action == GRAB) & ~holding & crate_front.any()
        removing = (action == GRAB) & ~holding & thief_front.any()
        held = jnp.where((action == GRAB) & holding, held.at[0].set(-1), held)
        held = jnp.where(taking, jnp.where(held == crate_front.argmax(), -1, held).at[0].set(crate_front.argmax()), held)
        held = jnp.where(removing & (jnp.arange(NUM_HOLDERS) == 1 + NUM_COLLECTORS + thief_front.argmax()), -1, held)
        thieves = jnp.where(removing & (jnp.arange(NUM_THIEVES) == thief_front.argmax())[:, None], 0, state.thieves)
        state = state.replace(player=player, facing=facing, crates=crates, held=held, thieves=thieves)
        # the collectors, in order; the first one to take a crate ends the collectors' turn (the original returns there)
        collectors, stop = state.collectors, jnp.bool_(False)
        for i in range(NUM_COLLECTORS):
            holder = 1 + i
            unheld = ~(state.held[:, None] == jnp.arange(NUM_CRATES)).any(0)
            drop, take, which, move, step = self.bot_turn(state, holder, goal, real & unheld & ~at(goal, state.crates), collectors[i])
            drop, take, move = drop & ~stop, take & ~stop, move & ~stop
            collectors = collectors.at[i].add(jnp.where(move, step, 0))
            crates = shift_crate(state.crates, state.held[holder], step, move & (state.held[holder] >= 0))
            held = state.held.at[holder].set(jnp.where(drop, -1, jnp.where(take, which, state.held[holder])))
            state = state.replace(crates=crates, collectors=collectors, held=held)
            stop = stop | take
        # the thieves, in order, the same way; they also take crates held by the player or a collector
        thieves, stop = state.thieves, jnp.bool_(False)
        for j in range(NUM_THIEVES):
            holder = 1 + NUM_COLLECTORS + j
            by_thief = (state.held[1 + NUM_COLLECTORS :, None] == jnp.arange(NUM_CRATES)).any(0)
            drop, take, which, move, step = self.bot_turn(state, holder, sink, real & ~by_thief & ~at(sink, state.crates), thieves[j])
            drop, take, move = drop & ~stop, take & ~stop, move & ~stop
            thieves = thieves.at[j].add(jnp.where(move, step, 0))
            crates = shift_crate(state.crates, state.held[holder], step, move & (state.held[holder] >= 0))
            held = jnp.where(take & (state.held == which), -1, state.held).at[holder].set(jnp.where(drop, -1, jnp.where(take, which, state.held[holder])))
            state = state.replace(crates=crates, thieves=thieves, held=held)
            stop = stop | take
        unheld = ~(state.held[:, None] == jnp.arange(NUM_CRATES)).any(0)
        advanced = (~real | (at(goal, state.crates) & unheld)).all()
        lost = ~advanced & (steps == 0)
        won = advanced & (state.level == NUM_LEVELS - 1)
        after = state.replace(steps=steps)
        after = jax.tree.map(lambda a, b: jnp.where(advanced & ~won, a, b), self.init_level(jnp.minimum(state.level + 1, NUM_LEVELS - 1)), after)
        return after, advanced, lost, won

    def render(self, state: State) -> jax.Array:
        front = state.player + jnp.asarray(MOVES)[state.facing]
        by_player = jnp.arange(NUM_CRATES) == state.held[0]
        by_bot = (state.held[1:, None] == jnp.arange(NUM_CRATES)).any(0)
        faced = (state.crates == front).all(-1)
        crate_border = jnp.where(by_player, CRATE_HELD_BY_PLAYER, jnp.where(faced, CRATE_FACED, jnp.where(by_bot, CRATE_HELD_BY_BOT, CRATE_BORDER)))
        thief_border = jnp.where((state.thieves == front).all(-1), THIEF_FACED, THIEF_BORDER)
        cells = jnp.concatenate([state.crates, state.collectors, state.thieves])
        border = jnp.concatenate([crate_border, jnp.full(NUM_COLLECTORS, COLLECTOR_COLOR), thief_border])
        inner = jnp.concatenate([jnp.full(NUM_CRATES, CRATE_INNER), jnp.full(NUM_COLLECTORS, COLLECTOR_COLOR), jnp.full(NUM_THIEVES, THIEF_COLOR)])
        # the 4x4 sprites never overlap, so a cell grid of colors blown up 4x draws them all; the ring (padding) is cut off
        grids = [jnp.full((CELLS, CELLS), -1).at[cells[:, 1], cells[:, 0]].set(c)[1:-1, 1:-1] for c in (border, inner)]
        border, inner = (jnp.repeat(jnp.repeat(g, CELL, 0), CELL, 1) for g in grids)
        color = jnp.where(jnp.tile(jnp.asarray(EDGE), (CELLS - 2, CELLS - 2)), border, inner)
        frame = jnp.where(color >= 0, color, level_data("background", state.level))
        column, row = (state.player - 1) * CELL
        frame = jax.lax.dynamic_update_slice(frame, jnp.asarray(PLAYER)[state.facing], (row, column))
        # the step bar: x < round(64 * steps / budget) without the float (there are no exact halves to round)
        lit = (2 * jnp.arange(SIZE) + 1) * level_data("steps", state.level) < 2 * SIZE * state.steps
        return frame.at[HUD_ROW].set(jnp.where(lit, HUD_LEFT, HUD_SPENT))
