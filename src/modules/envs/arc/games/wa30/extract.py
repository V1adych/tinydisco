"""Turns the original ARC-AGI-3 game wa30 (originals/wa30.py, run with the arcengine package) into the arrays game.py plays on:
python src/modules/envs/arc/games/wa30/extract.py  -> wa30.npz next to it.
Everything sits on a 4-pixel cell grid; a cell grid here is 18x18 (the 16 cells plus a blocked ring around them, cell = pixel // 4 + 1).
Per level: the static layer-0 sprites pre-rendered into one 64x64 background, cell grids of blocked / checkered / goal / sink cells,
the start cells of the player, crates, collectors and thieves (in the game's own list order) and the step budget, padded to the
largest level."""

import importlib.util
import sys
from pathlib import Path

import numpy as np

SOURCE = Path(__file__).parents[2] / "originals/wa30.py"
OUT = Path(__file__).with_name("wa30.npz")
CELL, CELLS = 4, 18
# what a sprite is to the game, by its (obfuscated) tag
TAG_ROLE = {"wbmdvjhthc": "player", "geezpjgiyd": "crate", "kdweefinfi": "collector", "ysysltqlke": "thief",
            "fsjjayjoeg": "goal", "zqxwgacnue": "sink", "bnzklblgdk": "checkered"}
MOVERS = ("player", "crate", "collector", "thief")


def load(source: Path):
    spec = importlib.util.spec_from_file_location("wa30_original", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules["wa30_original"] = module
    spec.loader.exec_module(module)
    return module


def role_of(sprite) -> str:
    for tag in sprite.tags:
        if tag in TAG_ROLE:
            return TAG_ROLE[tag]
    return "other"


def cells(sprites) -> np.ndarray:
    return np.array([(s.x // CELL + 1, s.y // CELL + 1) for s in sprites], np.int32).reshape(-1, 2)


def area(sprites) -> np.ndarray:
    """The cells whose top-left pixel lies inside one of the sprites (the game's zone test is on that pixel)."""
    out = np.zeros((CELLS, CELLS), bool)
    for s in sprites:
        out[s.y // CELL + 1 : (s.y + s.height) // CELL + 1, s.x // CELL + 1 : (s.x + s.width) // CELL + 1] = True
    return out


def extract(game, module) -> dict:
    levels = []
    for i in range(len(module.levels)):
        game.set_level(i)
        sprites = game.current_level.get_sprites()
        by_role = {}
        for s in sprites:
            by_role.setdefault(role_of(s), []).append(s)
        blocked = np.zeros((CELLS, CELLS), bool)
        blocked[[0, -1], :] = blocked[:, [0, -1]] = True
        for s in sprites:  # only a collidable sprite's top-left cell blocks, whatever its size (the game keeps a set of positions)
            if s.is_collidable and role_of(s) not in MOVERS:
                blocked[s.y // CELL + 1, s.x // CELL + 1] = True
        level = dict(
            background=game.camera._raw_render([s for s in sprites if role_of(s) not in MOVERS]),
            blocked=blocked,
            checkered=area(by_role.get("checkered", [])),
            goal=area(by_role.get("goal", [])),
            sink=area(by_role.get("sink", [])),
            player=cells(by_role["player"]).squeeze(0),
            steps=np.int32(game.current_level.get_data("StepCounter")),
        )
        for role, name in (("crate", "crates"), ("collector", "collectors"), ("thief", "thieves")):
            level[name] = cells(by_role.get(role, []))
            level["num_" + name] = np.int32(len(by_role.get(role, [])))
        levels.append(level)
    out = {}
    for key in levels[0]:
        arrays = [lv[key] for lv in levels]
        if key in ("crates", "collectors", "thieves"):  # padded with the ring cell (0, 0), which the game never uses
            n = max(len(a) for a in arrays)
            arrays = [np.concatenate([a, np.zeros((n - len(a), 2), np.int32)]) for a in arrays]
        out[key] = np.stack(arrays)
    out["player_pixels"] = np.array(by_role["player"][0].pixels, np.int8)
    return out


if __name__ == "__main__":
    module = load(SOURCE)
    data = extract(module.Wa30(), module)
    np.savez_compressed(OUT, **data)
    print(f"{OUT}: {len(data['background'])} levels, {data['crates'].shape[1]} crate slots, {OUT.stat().st_size // 1024} KB")
