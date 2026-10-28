"""Checks the JAX ARC-AGI-3 games against the original Python games (envs/arc/originals/<id>.py, see envs/arc/download.py,
run with the arcengine dev dependency), frame by frame: python src/test_arc.py
Per game: the original's class name and a guided policy that reaches what random play never does (locks, level passes)."""

import importlib.util
import sys
from pathlib import Path

import jax
import numpy as np
from arcengine import ActionInput, GameAction, GameState
from jax import numpy as jnp

from envs import make

ORIGINALS = Path(__file__).parent / "envs/arc/originals"
ACTIONS = (GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4)
STEPS_PER_LEVEL = 3000


def load_original(name: str, class_name: str):
    path = ORIGINALS / f"{name}.py"
    assert path.exists(), f"{path} is missing: python src/envs/arc/download.py fetches the originals"
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


# game -> (class name in its original.py, guided policy)
GAMES = {"ls20": ("Ls20", guide_ls20)}


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
