"""Checks the JAX ARC-AGI-3 games against the original Python games (envs/arc/originals/<id>.py, see src/modules/envs/arc/download.py,
run with the arcengine dev dependency), frame by frame: python -m pytest tests/test_arc.py
Per game: the original's class name and a guided policy that reaches what random play never does (locks, level passes)."""

import heapq
import importlib.util
import sys
from collections import deque
from pathlib import Path

import jax
import numpy as np
from arcengine import ActionInput, GameAction, GameState
from jax import numpy as jnp

from modules.envs import make

ORIGINALS = Path(__file__).resolve().parents[1] / "src/modules/envs/arc/originals"
ACTIONS = (GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4, GameAction.ACTION5)
STEPS_PER_LEVEL = 3000


def load_original(name: str, class_name: str):
    path = ORIGINALS / f"{name}.py"
    assert path.exists(), f"{path} is missing: python src/modules/envs/arc/download.py fetches the originals"
    spec = importlib.util.spec_from_file_location(f"{name}_original", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"{name}_original"] = module
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def cell(x: int, y: int) -> tuple[int, int]:
    """The 5x5 cell a sprite at (x, y) belongs to in ls20: the maze's cells start at x = 4, y = 0."""
    return (x - 4) // 5, y // 5


def first_step(walls: set, start: tuple, goal: tuple) -> int | None:
    """The first action of a shortest path from start to goal around the walls (breadth first), None if there is none."""
    moves = ((0, -1), (0, 1), (-1, 0), (1, 0))
    queue, came = [start], {start: None}
    if start == goal:
        return None
    while queue:
        here = queue.pop(0)
        if here == goal:
            while came[here][0] != start:
                here = came[here][0]
            return came[here][1]
        for a, (dx, dy) in enumerate(moves):
            there = (here[0] + dx, here[1] + dy)
            if there not in came and there not in walls and 0 <= there[0] < 12 and 0 <= there[1] < 12:
                came[there] = (here, a)
                queue.append(there)
    return None


def guide_ls20(game, rng) -> int:
    """Mostly a step along a shortest path to what the level needs next (the lock if the symbol fits, else the tile that fixes
    its first wrong attribute, a pickup when the budget runs low, now and then a lock that refuses), sometimes random."""
    if rng.random() < 0.15:
        return int(rng.integers(4))
    level = game.current_level
    locks = [i for i in range(len(game.qqv)) if not game.rzt[i]]
    fits = [i for i in locks if game.qhg(i)]
    if fits:
        goal = game.qqv[fits[0]]
    elif rng.random() < 0.1:
        goal = game.qqv[locks[0]]
    else:
        i = locks[0]
        want, have = (game.gfy[i], game.vxy[i], game.cjl[i]), (game.snw, game.tmx, game.tuv)
        tag = "gsu" if want[0] != have[0] else "gic" if want[1] != have[1] else "bgt"
        goal = next(iter(level.get_sprites_by_tag(tag)), None)
    pickups = level.get_sprites_by_tag("iri")
    if pickups and (goal is None or game.ggk.snw < 12):
        goal = min(pickups, key=lambda s: abs(s.x - game.mgu.x) + abs(s.y - game.mgu.y))
    walls = {cell(s.x, s.y) for s in level.get_sprites_by_tag("jdd")}
    action = None if goal is None else first_step(walls, cell(game.mgu.x, game.mgu.y), cell(goal.x, goal.y))
    return int(rng.integers(4)) if action is None else action


# tr87, per level: the variant digit each cursor position must show, over the answer row, or over every rule side (left,
# right, next rule...) on the levels whose rules are editable; worked out by hand from the original's rule tables
TR87_SOLUTIONS = ([3, 2, 6, 5, 1], [3, 1, 5, 1, 2, 2, 7], [4, 6, 7, 7, 5, 6, 1], [3, 2, 7, 1, 2, 3, 7], [5, 3, 1, 5, 4, 1, 2, 2], [7, 7, 7, 3, 1, 2, 2, 1, 6, 6, 6, 5])


def guide_tr87(game, rng) -> int:
    """Mostly the move or cycle that brings the nearest wrong tile to its solution digit, sometimes random; now and then a
    whole level is played randomly so that the budget runs out."""
    if game.upmkivwyrxz == game.vfpimnmtnta:  # a fresh level
        game.squander = rng.random() < 0.15
    if game.squander or rng.random() < 0.15:
        return int(rng.integers(4))
    if game.current_level.get_data("alter_rules"):
        tiles = [side[0] for rule in game.cifzvbcuwqe for side in rule]
    else:
        tiles = game.ztgmtnnufb
    have, want = [int(s.name[-1]) for s in tiles], TR87_SOLUTIONS[game.level_index]
    wrong = [k for k in range(len(tiles)) if have[k] != want[k]]
    if not wrong:
        return int(rng.integers(4))
    k = min(wrong, key=lambda k: abs(k - game.qvtymdcqear_index))
    if k == game.qvtymdcqear_index:
        return 1 if (want[k] - have[k]) % 7 <= 3 else 0
    return 3 if k > game.qvtymdcqear_index else 2


def route(start: tuple, valid, is_target) -> tuple[list | None, set]:
    """The wa30 bots' search: the breadth-first path from start over valid cells to the first target (None if there is none),
    and every cell it visited."""
    queue, seen = [[start]], {start}
    while queue:
        path = queue.pop(0)
        if is_target(path[-1]):
            return path, seen
        for there in ((path[-1][0] - 4, path[-1][1]), (path[-1][0] + 4, path[-1][1]), (path[-1][0], path[-1][1] - 4), (path[-1][0], path[-1][1] + 4)):
            if there not in seen and valid(there):
                seen.add(there)
                queue.append(path + [there])
    return None, seen


def towards(here: tuple, cell: tuple) -> int:
    """The wa30 move from a cell to an adjacent one."""
    dx, dy = cell[0] - here[0], cell[1] - here[1]
    return 0 if dy < 0 else 1 if dy > 0 else 2 if dx < 0 else 3


def guide_wa30(game, rng) -> int:
    """Mostly: drag the held crate to a goal and drop it there, or, where the player cannot reach one, to where a collector
    bot can take it over and deliver it (drop it where no path is left); else remove a thief that is close (they steal from
    goals), or walk to the nearest loose crate on a side from which it can be delivered (the player cannot turn while
    dragging) and grab it; sometimes random."""
    if rng.random() < 0.05:
        return int(rng.integers(5))
    level = game.current_level
    player = level.get_sprites_by_tag("wbmdvjhthc")[0]
    here, facing = (player.x, player.y), {0: 0, 180: 1, 270: 2, 90: 3}[player.rotation]
    moves = ((-4, 0), (4, 0), (0, -4), (0, 4))
    around = lambda v: tuple((v[0] + dx, v[1] + dy) for dx, dy in moves)
    # cells a crate can be left in for a collector: next to one the collector can walk to, from which the crate's offset
    # also lets it reach a free goal cell (a collector never lets go of a crate it cannot deliver), and only while the
    # collector's region has more free goals than crates already waiting for it or carried in it
    goals = {(x, y) for x in range(0, 64, 4) for y in range(0, 64, 4) if game.shbxbhnhjc((x, y)) and (x, y) not in game.pkbufziase}
    regions = {}
    for bot in level.get_sprites_by_tag("kdweefinfi"):
        regions.setdefault(frozenset(route((bot.x, bot.y), game.kblzhbvysd, lambda v: False)[1]), []).append(bot)
    handoff = set()
    for reach, bots in regions.items():
        pending = [c for c in level.get_sprites_by_tag("geezpjgiyd") if not game.shbxbhnhjc((c.x, c.y)) and (game.zmqreragji.get(c) in bots or (c not in game.zmqreragji and any(v in reach for v in around((c.x, c.y)))))]
        free = [g for g in goals if any(v in reach for v in around(g))]
        for dx, dy in moves:
            if len(free) > len(pending) and any((g[0] - dx, g[1] - dy) in reach for g in free):
                handoff |= {(x + dx, y + dy) for x, y in reach}

    def plan(crate: tuple, starts: list, is_target=None) -> tuple[int, list] | None:
        """The cheapest way to get a crate to a goal, else to a handoff cell, as (cost, [(crate cell, offset to the player)...])
        from one of the start states (cost, cell, offset): a drag move costs 1, letting go to grab from another side 5. The
        player's and the crate's own cells count as free: they will have left them."""
        if is_target is None:
            return plan(crate, starts, game.shbxbhnhjc) or plan(crate, starts, lambda c: c in handoff and not game.ahzqkfjpsc(c))
        free = lambda v: v not in game.pkbufziase or v in (here, crate)
        heap, dist, came = list(starts), {(c, d): cost for cost, c, d in starts}, {}
        heapq.heapify(heap)
        while heap:
            cost, c, d = heapq.heappop(heap)
            if cost > dist[c, d]:
                continue
            if is_target(c):
                path = [(c, d)]
                while path[-1] in came:
                    path.append(came[path[-1]])
                return cost, path[::-1]
            p = (c[0] - d[0], c[1] - d[1])
            steps = [((c[0] + dx, c[1] + dy), d, 1) for dx, dy in moves if (free((p[0] + dx, p[1] + dy)) or (p[0] + dx, p[1] + dy) == c) and (p[0] + dx, p[1] + dy) not in game.qthdiggudy and (free((c[0] + dx, c[1] + dy)) or (c[0] + dx, c[1] + dy) == p)]
            steps += [(c, d2, 5) for d2 in moves if d2 != d and free((c[0] - d2[0], c[1] - d2[1])) and (c[0] - d2[0], c[1] - d2[1]) not in game.qthdiggudy]
            for c2, d2, extra in steps:
                if cost + extra < dist.get((c2, d2), cost + extra + 1):
                    dist[c2, d2], came[c2, d2] = cost + extra, (c, d)
                    heapq.heappush(heap, (cost + extra, c2, d2))
        return None

    if player in game.nsevyuople:
        crate = game.nsevyuople[player]
        crate, offset = (crate.x, crate.y), (crate.x - player.x, crate.y - player.y)
        found = plan(crate, [(0, crate, offset)])
        if found is None or len(found[1]) < 2 or found[1][1][1] != offset:
            return 4
        return towards(here, (here[0] + found[1][1][0][0] - crate[0], here[1] + found[1][1][0][1] - crate[1]))
    thieves = [(t.x, t.y) for t in level.get_sprites_by_tag("ysysltqlke")]
    path, _ = route(here, game.kblzhbvysd, lambda v: any(t in around(v) for t in thieves))
    if path is not None and len(path) <= 6:
        if len(path) == 2:  # any free cell next to a thief does; a fixed choice lets a sidestepping thief dance forever
            options = [v for v in around(here) if game.kblzhbvysd(v) and any(t in around(v) for t in thieves)]
            path = [here, options[rng.integers(len(options))]]
        target = next(t for t in thieves if t in around(path[-1]))
    else:
        # the loose crate with the cheapest plan, counting the walk to the side it starts from
        best = None
        for c in level.get_sprites_by_tag("geezpjgiyd"):
            crate = (c.x, c.y)
            if c in game.zmqreragji or game.shbxbhnhjc(crate):
                continue
            starts = [((abs(crate[0] - d[0] - here[0]) + abs(crate[1] - d[1] - here[1])) // 4, crate, d) for d in moves if game.kblzhbvysd((crate[0] - d[0], crate[1] - d[1])) or (crate[0] - d[0], crate[1] - d[1]) == here]
            found = plan(crate, starts)
            if found is not None and len(found[1]) > 1 and (best is None or found[0] < best[0]):
                best = (found[0], crate, found[1][0][1])
        if best is None:
            return int(rng.integers(5))
        _, target, offset = best
        path, _ = route(here, game.kblzhbvysd, lambda v: v == (target[0] - offset[0], target[1] - offset[1]))
    if path is None:
        return int(rng.integers(5))
    if len(path) > 1:
        return towards(here, path[1])
    action = towards(here, target)
    return 4 if action == facing else action


def guide_ul01(game, rng) -> int:
    """Mostly a step along a shortest path to the key, then to the door, sometimes random."""
    if rng.random() < 0.2:
        return int(rng.integers(4))
    goal = game._door[0] if game._has_key else game._key[0]
    walls = set() if game._has_key else {(d.x, d.y) for d in game._door}
    walls |= {(8, k) for k in range(12)} | {(k, 8) for k in range(12)}  # first_step searches 12x12, the grid is 8x8
    action = first_step(walls, (game._player.x, game._player.y), (goal.x, goal.y))
    return int(rng.integers(4)) if action is None else action


SK01_ROUTES = {}  # (level, player, crates) -> action, searched once per level and kept across the test's restarts


def sokoban_route(game) -> dict:
    """sk01: the action at each (level, player, crates) state of a shortest push sequence that puts every crate on a target
    (breadth first), empty if there is none."""
    level = game.current_level
    width, height = level.grid_size
    walls = {(s.x, s.y) for s in level.get_sprites_by_tag("wall")}
    targets = {(s.x, s.y) for s in game._targets}
    free = lambda v: 0 <= v[0] < width and 0 <= v[1] < height and v not in walls
    start = ((game._player.x, game._player.y), tuple(sorted((b.x, b.y) for b in game._blocks)))
    queue, came = deque([start]), {start: None}
    while queue:
        here = queue.popleft()
        if set(here[1]) <= targets:
            route = {}
            while came[here] is not None:
                here, action = came[here]
                route[(game.level_index,) + here] = action
            return route
        (x, y), crates = here
        for a, (dx, dy) in enumerate(((0, -1), (0, 1), (-1, 0), (1, 0))):
            player, crates_after = (x + dx, y + dy), crates
            if not free(player):
                continue
            if player in crates:
                pushed = (player[0] + dx, player[1] + dy)
                if not free(pushed) or pushed in crates:
                    continue
                crates_after = tuple(sorted(pushed if c == player else c for c in crates))
            there = (player, crates_after)
            if there not in came:
                came[there] = (here, a)
                queue.append(there)
    return {}


def guide_sk01(game, rng) -> int:
    """The next push of a shortest solution; now and then a whole level is played randomly so that the step limit runs out
    (no random moves within a guided level: every detour would cost a new search, seconds each on the large level)."""
    if getattr(game, "fresh", None) is not game.current_level:
        game.fresh, game.squander = game.current_level, rng.random() < 0.25
    if game.squander:
        return int(rng.integers(4))
    here = (game.level_index, (game._player.x, game._player.y), tuple(sorted((b.x, b.y) for b in game._blocks)))
    if here not in SK01_ROUTES:
        SK01_ROUTES.update(sokoban_route(game))
    return SK01_ROUTES.get(here, int(rng.integers(4)))


# g50t, per level: actions (Up, Down, Left, Right, W = rewind) that clear it from its start; each W turns the run so far
# into a ghost that then holds a plate open for the next run; found by hand on the original's maps
G50T_SCRIPTS = (
    "RRRRWDDDDDDDRRRRR",
    "LLWDDDDLLLLUULLWUUULLLLLLLDDRRR",
    "UURRRRDDDDRWUURRRRRRRDDDDDDDLLLLLWUURRRRRRRDDDDDDDLLLLLLLUUURRUU",
    "DDRDWDDRRUURRDDDWLLLDDDDDRRRLLL",
    "UDDRRRDDDWDRRRUURRRRRRDDDWDRRRUURRRDDDDDRLDLLLLLUU",
    "LLUWLLULLWLLDLLLLUULLLDDDDDRRUDLLUUUUURRRDDRRDDRR",
    "DDLRUULLUUWDDRRUUUULUURRRWDDRRUUUURRDDDDLLL",
)


def guide_g50t(game, rng) -> int:
    """The level's script (indexed by the game's own action count, which restarts with the level); now and then a whole
    level is played randomly or the script is left at a random point for random play, so that deaths, the timer running
    out and rewinds in odd states get covered."""
    script = G50T_SCRIPTS[game.level_index]
    if game.ucorwtereb == 0:  # a fresh level
        game.leave_at = len(script) if rng.random() < 0.6 else int(rng.integers(len(script)))
    if game.ucorwtereb >= game.leave_at:
        return int(rng.integers(5))
    return "UDLRW".index(script[game.ucorwtereb])


def grid_step(free, start: tuple, goal: tuple) -> int | None:
    """The first of actions 1-4 (up, down, left, right) on a shortest path from start to goal over free cells, None if none."""
    queue, came = deque([start]), {start: None}
    while queue:
        here = queue.popleft()
        if here == goal:
            action = None
            while came[here] is not None:
                here, action = came[here]
            return action
        for a, (dx, dy) in enumerate(((0, -1), (0, 1), (-1, 0), (1, 0))):
            there = (here[0] + dx, here[1] + dy)
            if there not in came and free(there):
                came[there] = (here, a)
                queue.append(there)
    return None


def guide_ob01(game, rng) -> int:
    """Mostly a step of the active body along a shortest path to its pad, a switch when it is there or blocked, sometimes random."""
    if rng.random() < 0.15:
        return int(rng.integers(5))
    width, height = game.current_level.grid_size
    body, goal = game._players[game._active], game._goals[game._active]
    goals = {(g.x, g.y) for g in game._goals}
    blocked = {(s.x, s.y) for s in game.current_level.get_sprites_by_tag("wall")} - goals
    blocked |= {(p.x, p.y) for p in game._players}
    free = lambda v: 0 <= v[0] < width and 0 <= v[1] < height and v not in blocked
    action = None if (body.x, body.y) == (goal.x, goal.y) else grid_step(free, (body.x, body.y), (goal.x, goal.y))
    return 4 if action is None else action


def hanoi_move(state: tuple, action: int) -> tuple:
    """hn01's step on (stacks, hand, peg): actions 1-3 select a peg, 4 does nothing, 5 picks up or drops."""
    stacks, hand, peg = state
    if action < 3:
        return stacks, hand, action
    stacks = [list(s) for s in stacks]
    if action == 4 and hand is None and stacks[peg]:
        hand = stacks[peg].pop()
    elif action == 4 and hand is not None and (stacks[peg][-1] if stacks[peg] else 99) > hand:
        stacks[peg].append(hand)
        hand = None
    return tuple(tuple(s) for s in stacks), hand, peg


def guide_hn01(game, rng) -> int:
    """Mostly the next action of a shortest solution from the current state (breadth first, cached), sometimes random."""
    if rng.random() < 0.15:
        return int(rng.integers(5))
    start = (tuple(tuple(s) for s in game._st), game._hand, game._peg)
    if start not in getattr(game, "route", {}):
        queue, came = deque([start]), {start: None}
        while queue:
            here = queue.popleft()
            if here[0][2] == (3, 2, 1):
                break
            for a in (0, 1, 2, 4):
                there = hanoi_move(here, a)
                if there not in came:
                    came[there] = (here, a)
                    queue.append(there)
        game.route = {}
        while came[here] is not None:
            here, action = came[here]
            game.route[here] = action
    return game.route[start]


def grid_first_step(free, start: tuple, goals: set) -> int | None:
    """The first action (up, down, left, right) of a shortest path from start to any goal over free cells (breadth first),
    None if there is none or start is a goal."""
    queue, came = deque([start]), {start: None}
    while queue:
        here = queue.popleft()
        if here in goals:
            if here == start:
                return None
            while came[here][0] != start:
                here = came[here][0]
            return came[here][1]
        for a, (dx, dy) in enumerate(((0, -1), (0, 1), (-1, 0), (1, 0))):
            there = (here[0] + dx, here[1] + dy)
            if there not in came and free(there):
                came[there] = (here, a)
                queue.append(there)
    return None


def guide_bp01(game, rng) -> int:
    """Mostly: power the tower underfoot, else a step towards the nearest unpowered tower, else towards the goal; sometimes
    random."""
    if rng.random() < 0.1:
        return int(rng.integers(5))
    level = game.current_level
    here = (game._player.x, game._player.y)
    towers = {(s.x, s.y) for s in level.get_sprites_by_tag("tower") if "powered" not in s.tags}
    if here in towers:
        return 4
    width, height = level.grid_size
    walls = {(s.x, s.y) for s in level.get_sprites_by_tag("wall")}
    free = lambda v: 0 <= v[0] < width and 0 <= v[1] < height and v not in walls
    goal = level.get_sprites_by_tag("goal")[0]
    action = grid_first_step(free, here, towers or {(goal.x, goal.y)})
    return int(rng.integers(5)) if action is None else action


def guide_sv01(game, rng) -> int:
    """Mostly: a step towards the nearest food when hungry, else towards the nearest warm cell, waiting on one; sometimes
    random; now and then a whole level is played randomly so that hunger or cold kills."""
    if game._steps_survived == 0:  # a fresh level
        game.squander = rng.random() < 0.2
    if game.squander or rng.random() < 0.05:
        return int(rng.integers(5))
    width, height = game.current_level.grid_size
    here = (game._player.x, game._player.y)
    free = lambda v: 0 <= v[0] < width and 0 <= v[1] < height
    warm = {(s.x, s.y) for s in game.current_level.get_sprites_by_tag("warm_zone")}
    foods = {(s.x, s.y) for s in game._foods}
    action = grid_first_step(free, here, foods if game._hunger <= 60 and foods else warm)
    return 4 if action is None else action


def guide_dl01(game, rng) -> int:
    """Mostly the next step of a shortest path to the goal from where the queued move will leave the player (a move runs one
    action later), sometimes random. On the last level it drops a queued winning move while the blocked-move mark is still
    counting down: that countdown outlives the original's game restart but not ours (dl01/game.py's shortcut)."""
    if rng.random() < 0.1:
        return int(rng.integers(5))
    width, height = game.current_level.grid_size
    blocked = {(s.x, s.y) for tag in ("wall", "hazard") for s in game.current_level.get_sprites_by_tag(tag)}
    free = lambda v: 0 <= v[0] < width and 0 <= v[1] < height and v not in blocked
    here = (game._player.x, game._player.y)
    if game._q and free((here[0] + game._q[0][0], here[1] + game._q[0][1])):
        here = (here[0] + game._q[0][0], here[1] + game._q[0][1])
    goal = game.current_level.get_sprites_by_tag("goal")[0]
    if here == (goal.x, goal.y) and game.is_last_level() and game._ui._bump > 0:
        return 4
    action = grid_first_step(free, here, {(goal.x, goal.y)})
    return int(rng.integers(5)) if action is None else action


def guide_hs01(game, rng) -> int:
    """Mostly a step along a shortest path to the goal around the walls and the hunter, sometimes random."""
    if rng.random() < 0.2:
        return int(rng.integers(4))
    width, height = game.current_level.grid_size
    blocked = {(s.x, s.y) for s in game.current_level.get_sprites_by_tag("wall")} | {(game._hunter.x, game._hunter.y)}
    free = lambda v: 0 <= v[0] < width and 0 <= v[1] < height and v not in blocked
    action = grid_step(free, (game._player.x, game._player.y), (game._goal.x, game._goal.y))
    return int(rng.integers(4)) if action is None else action


def gravity_step(game, start: tuple, is_target) -> tuple[bool, int | None]:
    """gr01: whether a cell satisfying is_target can be reached from start under the game's own move-then-fall rule, and
    the first action of a shortest way there (breadth first; None if start already does)."""
    queue, came = deque([start]), {start: None}
    while queue:
        here = queue.popleft()
        if is_target(here):
            action = None
            while came[here] is not None:
                here, action = came[here]
            return True, action
        for a, (dx, dy) in enumerate(((0, -1), (0, 1), (-1, 0), (1, 0))):
            there = game._after_action_from(*here, dx, dy)
            if there is not None and there not in came:
                came[there] = (here, a)
                queue.append(there)
    return False, None


def guide_gr01(game, rng) -> int:
    """Mostly a step along a shortest way to the goal, sometimes random; now and then a whole level heads for a dead end
    (no way to the goal left) instead, so that warnings and losses get covered."""
    if getattr(game, "fresh", None) is not game.current_level:
        game.fresh, game.squander = game.current_level, rng.random() < 0.3
    if rng.random() < 0.1:
        return int(rng.integers(4))
    goal = (game._targets[0].x, game._targets[0].y)
    reaches_goal = lambda v: v == goal
    is_target = (lambda v: not gravity_step(game, v, reaches_goal)[0]) if game.squander else reaches_goal
    _, action = gravity_step(game, (game._player.x, game._player.y), is_target)
    return int(rng.integers(4)) if action is None else action


def guide_rv01(game, rng) -> int:
    """Mostly a step along a shortest path to the first target left, around the walls and the sparks, sometimes random."""
    if rng.random() < 0.2:
        return int(rng.integers(4))
    width, height = game.current_level.grid_size
    blocked = {(s.x, s.y) for s in game.current_level.get_sprites_by_tag("wall") + game._sparks}
    free = lambda v: 0 <= v[0] < width and 0 <= v[1] < height and v not in blocked
    action = grid_step(free, (game._player.x, game._player.y), (game._targets[0].x, game._targets[0].y))
    return int(rng.integers(4)) if action is None else action


def guide_zq01(game, rng) -> int:
    """Mostly a step along a shortest path to the target around the hazards that are on now, sometimes random."""
    if rng.random() < 0.2:
        return int(rng.integers(4))
    width, height = game.current_level.grid_size
    blocked = {(s.x, s.y) for s in game._hazard_sprites} if game._hazard_on else set()
    free = lambda v: 0 <= v[0] < width and 0 <= v[1] < height and v not in blocked
    action = grid_step(free, (game._player.x, game._player.y), (game._targets[0].x, game._targets[0].y))
    return int(rng.integers(4)) if action is None else action


def guide_fs01(game, rng) -> int:
    """Mostly a step towards the nearest inactive switch, then (door open) towards the target, sometimes random."""
    if rng.random() < 0.2:
        return int(rng.integers(4))
    level = game.current_level
    width, height = level.grid_size
    walls = {(s.x, s.y) for s in level.get_sprites_by_tag("wall")} | {(d.x, d.y) for d in game._door}
    free = lambda v: 0 <= v[0] < width and 0 <= v[1] < height and v not in walls
    goals = set(game._switch_positions - game._activated) or {(t.x, t.y) for t in game._targets}
    action = grid_first_step(free, (game._player.x, game._player.y), goals)
    return int(rng.integers(4)) if action is None else action


def first_action(start, successor, is_goal) -> int | None:
    """The first of actions 1-4 on a shortest path from start to a goal state, successor(state, action) giving the next
    state (breadth first), None if there is none or start is a goal."""
    queue, came = deque([start]), {start: None}
    while queue:
        here = queue.popleft()
        if is_goal(here):
            action = None
            while came[here] is not None:
                here, action = came[here]
            return action
        for a in range(4):
            there = successor(here, a)
            if there not in came:
                came[there] = (here, a)
                queue.append(there)
    return None


def guide_tp01(game, rng) -> int:
    """Mostly a step along a shortest path (through the portals) to the target, sometimes random."""
    if rng.random() < 0.2:
        return int(rng.integers(4))
    level = game.current_level
    width, height = level.grid_size
    walls = {(s.x, s.y) for s in level.get_sprites_by_tag("wall")}
    moves = ((0, -1), (0, 1), (-1, 0), (1, 0))

    def successor(here, a):
        there = (here[0] + moves[a][0], here[1] + moves[a][1])
        if not (0 <= there[0] < width and 0 <= there[1] < height) or there in walls:
            return here
        return game._portal_to_partner.get(there, there)

    target = game._targets[0]
    action = first_action((game._player.x, game._player.y), successor, lambda v: v == (target.x, target.y))
    return int(rng.integers(4)) if action is None else action


def guide_ic01(game, rng) -> int:
    """Mostly the next slide of a shortest slide sequence to the target, sometimes random."""
    if rng.random() < 0.2:
        return int(rng.integers(4))
    width, height = game.current_level.grid_size
    moves = ((0, -1), (0, 1), (-1, 0), (1, 0))

    def successor(here, a):
        while 0 <= here[0] + moves[a][0] < width and 0 <= here[1] + moves[a][1] < height and not game._blocked(here[0] + moves[a][0], here[1] + moves[a][1]):
            here = (here[0] + moves[a][0], here[1] + moves[a][1])
        return here

    target = game._targets[0]
    action = first_action((game._player.x, game._player.y), successor, lambda v: v == (target.x, target.y))
    return int(rng.integers(4)) if action is None else action


def guide_nw01(game, rng) -> int:
    """Mostly a step of a shortest path to the target over (cell, queued arrow) states, sometimes random."""
    if rng.random() < 0.2:
        return int(rng.integers(4))
    width, height = game.current_level.grid_size
    moves = ((0, -1), (0, 1), (-1, 0), (1, 0))

    def successor(here, a):
        (x, y), forced = here
        dx, dy = forced or moves[a]
        if not (0 <= x + dx < width and 0 <= y + dy < height):  # no level has walls
            return (x, y), None
        return (x + dx, y + dy), game._arrow_map.get((x + dx, y + dy))

    target = game._targets[0]
    action = first_action(((game._player.x, game._player.y), game._forced), successor, lambda v: v[0] == (target.x, target.y))
    return int(rng.integers(4)) if action is None else action


def guide_rs01(game, rng) -> int:
    """Mostly a step along a shortest path (around the other targets) to the nearest target still safe when reached (any
    target once the colors have cycled), else a random move that counts and touches no target; sometimes random, so that
    wrong colors get taken."""
    if game._win_hold or rng.random() < 0.05:
        return int(rng.integers(4))
    width, height = game.current_level.grid_size
    walls = {(s.x, s.y) for s in game.current_level.get_sprites_by_tag("wall")}
    targets = {(t.x, t.y): t.pixels[0][0] for t in game._targets}
    free = lambda v: 0 <= v[0] < width and 0 <= v[1] < height and v not in walls
    left = game._cycle_interval - game._steps % game._cycle_interval  # counted moves before the safe color changes
    moves = ((0, -1), (0, 1), (-1, 0), (1, 0))
    start = (game._player.x, game._player.y)
    queue, came = deque([(start, 0)]), {start: None}
    while queue:
        here, depth = queue.popleft()
        for a, (dx, dy) in enumerate(moves):
            there = (here[0] + dx, here[1] + dy)
            if there in came or not free(there):
                continue
            came[there] = (here, a)
            if there not in targets:
                queue.append((there, depth + 1))
            elif game._all_cycled or (targets[there] == game._safe_color and depth < left):
                while came[there][0] != start:
                    there = came[there][0]
                return came[there][1]
    options = [a for a, (dx, dy) in enumerate(moves) if free((start[0] + dx, start[1] + dy)) and (start[0] + dx, start[1] + dy) not in targets]
    return int(rng.choice(options)) if options else int(rng.integers(4))


def guide_va01(game, rng) -> int:
    """Mostly a step towards the nearest unvisited open cell, sometimes random."""
    if rng.random() < 0.1:
        return int(rng.integers(4))
    width, height = game.current_level.grid_size
    free = lambda v: 0 <= v[0] < width and 0 <= v[1] < height and v in game._open_cells
    action = grid_first_step(free, (game._player.x, game._player.y), game._open_cells - game._visited)
    return int(rng.integers(4)) if action is None else action


def guide_bd01(game, rng) -> int:
    """Mostly a step along a shortest path to the target over cells not yet visited, sometimes random (which loses when it
    steps back)."""
    if rng.random() < 0.1:
        return int(rng.integers(4))
    width, height = game.current_level.grid_size
    walls = {(s.x, s.y) for s in game.current_level.get_sprites_by_tag("wall")}
    free = lambda v: 0 <= v[0] < width and 0 <= v[1] < height and v not in walls and v not in game._visited
    action = grid_first_step(free, (game._player.x, game._player.y), {(t.x, t.y) for t in game._targets})
    return int(rng.integers(4)) if action is None else action


def guide_wk01(game, rng) -> int:
    """Mostly a step along a shortest path to the target around walls and holes, sometimes random; now and then a whole
    level is played randomly so that holes and the move budget get it."""
    if getattr(game, "fresh", None) is not game.current_level:
        game.fresh, game.squander = game.current_level, rng.random() < 0.15
    if game.squander or rng.random() < 0.05:
        return int(rng.integers(4))
    width, height = game.current_level.grid_size
    blocked = {(s.x, s.y) for tag in ("wall", "hole") for s in game.current_level.get_sprites_by_tag(tag)}
    free = lambda v: 0 <= v[0] < width and 0 <= v[1] < height and v not in blocked
    action = grid_first_step(free, (game._player.x, game._player.y), {(t.x, t.y) for t in game._targets})
    return int(rng.integers(4)) if action is None else action


def guide_ms01(game, rng) -> int:
    """Mostly a step along a shortest path to the goal around the (hidden) mines, sometimes random; now and then a whole
    level is played randomly so that it steps on a mine."""
    if getattr(game, "fresh", None) is not game.current_level:
        game.fresh, game.squander = game.current_level, rng.random() < 0.15
    if game.squander or rng.random() < 0.05:
        return int(rng.integers(4))
    width, height = game.current_level.grid_size
    free = lambda v: 0 <= v[0] < width and 0 <= v[1] < height and v not in game._minefield
    goal = game.current_level.get_sprites_by_tag("goal")[0]
    action = grid_first_step(free, (game._player.x, game._player.y), {(goal.x, goal.y)})
    return int(rng.integers(4)) if action is None else action


def guide_co01(game, rng) -> int:
    """Mostly the first move of a shortest path to the goal over (cell, active hue), sometimes random."""
    if rng.random() < 0.15:
        return int(rng.integers(4))
    level = game.current_level
    width, height = level.grid_size
    first = {}  # cell -> the first non-player sprite there in list order, as get_sprite_at finds it
    for s in level.get_sprites():
        if "player" not in s.tags:
            first.setdefault((s.x, s.y), s)
    start = ((game._player.x, game._player.y), game._active)
    queue, came = deque([start]), {start: None}
    while queue:
        here = queue.popleft()
        if here[0] == (game._goal.x, game._goal.y):
            while came[here][0] != start:
                here = came[here][0]
            return came[here][1]
        for a, (dx, dy) in enumerate(((0, -1), (0, 1), (-1, 0), (1, 0))):
            cell, active = (here[0][0] + dx, here[0][1] + dy), here[1]
            s = first.get(cell)
            if not (0 <= cell[0] < width and 0 <= cell[1] < height) or (s and "wall" in s.tags):
                continue
            hue = s and (8 if "c8" in s.tags else 11)
            if s and "door" in s.tags and hue != active:
                continue
            there = (cell, hue if s and "recolor" in s.tags else active)
            if there not in came:
                came[there] = (here, a)
                queue.append(there)
    return int(rng.integers(4))


def guide_sn01(game, rng) -> int:
    """Mostly a step along a shortest path to the nearest food around the walls and the body (its tail moves away),
    sometimes random, so that the snake bites itself."""
    if rng.random() < 0.1:
        return int(rng.integers(4))
    width, height = game.current_level.grid_size
    blocked = {(s.x, s.y) for s in game.current_level.get_sprites_by_tag("wall")} | set(list(game._body)[1:])
    free = lambda v: 0 <= v[0] < width and 0 <= v[1] < height and v not in blocked
    action = grid_first_step(free, game._body[-1], {(f.x, f.y) for f in game._foods})
    return int(rng.integers(4)) if action is None else action


def guide_or01(game, rng) -> int:
    """Mostly the first move of a shortest plan (breadth first over the cell and the keys' ring slots, simulating the
    rotation and the pickups), sometimes random; never the winning move while the pickup flash would outlive it: the
    original's flash carries over the reset that follows a win, the port's does not."""
    level = game.current_level
    width, height = level.grid_size
    blocked = {(s.x, s.y) for s in level.get_sprites() if "wall" in s.tags or "pillar" in s.tags}
    ring, moves = sys.modules[type(game).__module__].ring_offsets(), ((0, -1), (0, 1), (-1, 0), (1, 0))

    def after(here, a):
        (x, y), keys = here
        keys = tuple((r + 1) % len(ring) for r in keys)
        cell = (x + moves[a][0], y + moves[a][1])
        if cell in blocked or not (0 <= cell[0] < width and 0 <= cell[1] < height):
            return (x, y), keys
        for i, r in enumerate(keys):
            if abs(game._cx + ring[r][0] - cell[0]) + abs(game._cy + ring[r][1] - cell[1]) == 1:
                return cell, keys[:i] + keys[i + 1 :]
        return cell, keys

    goal = (game._goal.x, game._goal.y)
    start = ((game._player.x, game._player.y), tuple(game._ring_index))
    action = int(rng.integers(4))
    queue, came = deque([start] if rng.random() >= 0.1 else []), {start: None}
    while queue:
        here = queue.popleft()
        if here[0] == goal and not here[1] and here != start:
            while came[here][0] != start:
                here = came[here][0]
            action = came[here][1]
            break
        for a in range(4):
            there = after(here, a)
            if there not in came:
                came[there] = (here, a)
                queue.append(there)
    there = after(start, action)
    if game.is_last_level() and there[0] == goal and not there[1] and (game._ui._pickup_flash > 1 or start[1]):
        return action ^ 1  # the opposite move, which cannot land on the goal
    return action


def magnet_move(here: tuple, action: int, walls: set) -> tuple:
    """mb01: the (player, crates) after an action: the player steps onto a free cell, then each crate in turn slides one
    cell toward it along the longer axis if that cell is free."""
    (x, y), crates = here
    dx, dy = ((0, -1), (0, 1), (-1, 0), (1, 0))[action]
    blocked = lambda v, crates: not (0 <= v[0] < 10 and 0 <= v[1] < 10) or v in walls or v in crates
    player = (x + dx, y + dy)
    if blocked(player, crates):
        return here
    crates = list(crates)
    for i, (cx, cy) in enumerate(crates):
        gx, gy = player[0] - cx, player[1] - cy
        step = (np.sign(gx), 0) if abs(gx) >= abs(gy) else (0, np.sign(gy))
        pulled = (cx + int(step[0]), cy + int(step[1]))
        if not blocked(pulled, crates) and pulled != player:
            crates[i] = pulled
    return player, tuple(crates)


def guide_mb01(game, rng) -> int:
    """Mostly the next action of a shortest path to the goal over (player, crates) states (breadth first, cached), sometimes
    random."""
    if rng.random() < 0.15:
        return int(rng.integers(4))
    walls = {(s.x, s.y) for s in game.current_level.get_sprites_by_tag("wall")}
    goal = (game._goal.x, game._goal.y)
    start = ((game._player.x, game._player.y), tuple((c.x, c.y) for c in game._crates))
    if start not in getattr(game, "route", {}):
        queue, came, here = deque([start]), {start: None}, None
        while queue:
            here = queue.popleft()
            if here[0] == goal:
                break
            for a in range(4):
                there = magnet_move(here, a, walls)
                if there not in came:
                    came[there] = (here, a)
                    queue.append(there)
        game.route = {}
        while came[here] is not None:
            here, action = came[here]
            game.route[here] = action
    return game.route.get(start, int(rng.integers(4)))


# game -> (class name in its original.py, guided policy)
GAMES = {"bd01": ("Bd01", guide_bd01), "bp01": ("Bp01", guide_bp01), "co01": ("Co01", guide_co01), "dl01": ("Dl01", guide_dl01), "fs01": ("Fs01", guide_fs01), "g50t": ("G50t", guide_g50t), "gr01": ("Gr01", guide_gr01), "hn01": ("Hn01", guide_hn01), "hs01": ("Hs01", guide_hs01), "ic01": ("Ic01", guide_ic01), "ls20": ("Ls20", guide_ls20), "mb01": ("Mb01", guide_mb01), "ms01": ("Ms01", guide_ms01), "nw01": ("Nw01", guide_nw01), "ob01": ("Ob01", guide_ob01), "or01": ("Or01", guide_or01), "rs01": ("Rs01", guide_rs01), "rv01": ("Rv01", guide_rv01), "sk01": ("Sk01", guide_sk01), "sn01": ("Sn01", guide_sn01), "sv01": ("Sv01", guide_sv01), "tp01": ("Tp01", guide_tp01), "tr87": ("Tr87", guide_tr87), "ul01": ("Ul01", guide_ul01), "va01": ("Va01", guide_va01), "wa30": ("Wa30", guide_wa30), "wk01": ("Wk01", guide_wk01), "zq01": ("Zq01", guide_zq01)}


def check_against_original(name: str):
    """Original and port get the same actions from every level's start; every frame and score must agree."""
    class_name, guide = GAMES[name]
    original_class = load_original(name, class_name)
    env, _ = make(f"Arc-{name}")
    step = jax.jit(env.step)
    rng = np.random.default_rng(0)
    cleared = [0] * env.num_levels
    for level in range(env.num_levels):
        original = original_class()
        original.set_level(level)
        _, state = env.reset(jax.random.key(0))
        state = state.replace(game=env.init_level(jnp.int32(level)))
        assert (np.asarray(env.observe(state))[..., 0] == original.camera.render(original.current_level.get_sprites())).all(), f"{name} level {level}: first frame differs"
        score = original._score
        for t in range(STEPS_PER_LEVEL):
            action = guide(original, rng)
            frame = original.perform_action(ActionInput(id=ACTIONS[action]), raw=True)
            _, state, reward, done, info = step(jax.random.key(t), state, jnp.int32(action))
            ours = np.asarray(info["final_observation"])[..., 0]
            assert (ours == frame.frame[-1]).all(), f"{name} level {level} step {t} action {action}: frame differs at {np.argwhere(ours != frame.frame[-1])[:5].tolist()}"
            assert float(reward) == frame.levels_completed - score, f"{name} level {level} step {t}: reward {reward} vs {frame.levels_completed - score}"
            cleared[level] += frame.levels_completed - score
            score = frame.levels_completed
            over = frame.state in (GameState.GAME_OVER, GameState.WIN)
            assert bool(done) == (over or bool(info["truncated"])), f"{name} level {level} step {t}: done {done} vs state {frame.state}"
            if done:  # ours restarted (the game ended, or our step cap hit): restart the original the same way
                original.full_reset()
                original.set_level(level)
                score = original._score
                state = state.replace(game=env.init_level(jnp.int32(level)))
    print(f"{name}: clears per level during the check: {cleared}")
    assert all(cleared), f"{name}: the guided play left a level never cleared ({cleared}), so its locks went unchecked"


def test_games_match_originals():
    for name in GAMES:
        check_against_original(name)


def test_arc_env_runs_batched():
    env, _ = make("Arc-ls20")
    keys = jax.random.split(jax.random.key(0), 8)
    obs, state = jax.vmap(env.reset)(keys)
    assert obs.shape == (8, 64, 64, 1) and obs.dtype == jnp.uint8
    step = jax.jit(jax.vmap(env.step))
    for t in range(5):
        obs, state, reward, done, info = step(keys, state, jnp.full(8, t % 4))
    assert obs.shape == (8, 64, 64, 1) and reward.shape == (8,) and done.shape == (8,)


if __name__ == "__main__":
    test_arc_env_runs_batched()
    test_games_match_originals()
    print("ok")
