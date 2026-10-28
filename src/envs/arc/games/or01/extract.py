"""Turns the community ARC-AGI-3 game or01 (originals/or01.py, run with the arcengine package) into the arrays game.py plays
on:  python src/envs/arc/games/or01/extract.py  -> or01.npz next to it.
Per level (all 10x10): the player's start, the pillar, the goal, the keys' starting slots on the ring around the pillar
(padded with real = False), the wall mask, and the camera's grid -> frame mapping (row * GRID + column, -1 on the
letterbox); plus the ring's offsets, clockwise from east."""

import importlib.util
import sys
from pathlib import Path

import numpy as np

SOURCE = Path(__file__).parents[2] / "originals/or01.py"
OUT = Path(__file__).with_name("or01.npz")
SIZE = 64


def load(source: Path):
    spec = importlib.util.spec_from_file_location("or01_original", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules["or01_original"] = module
    spec.loader.exec_module(module)
    return module


def pixel_cells(camera, grid: int) -> np.ndarray:
    """The grid cell (row * grid + column) each frame pixel shows, -1 = letterbox; grid is the padded grid's side."""
    scale, x0, y0 = camera._calculate_scale_and_offset()
    rows, cols = np.mgrid[:SIZE, :SIZE]
    gy, gx = (rows - y0) // scale, (cols - x0) // scale
    inside = (rows >= y0) & (cols >= x0) & (gy < camera.height) & (gx < camera.width)
    return np.where(inside, gy * grid + gx, -1).astype(np.int32)


def extract(game, module) -> dict:
    grid = max(max(level.grid_size) for level in module.levels)
    count = max(len(level.get_sprites_by_tag("key")) for level in module.levels)
    ring = module.ring_offsets()
    levels = []
    for i in range(len(module.levels)):
        game.set_level(i)
        level = game.current_level
        # game.py places keys by ring slot only, so they must start on the ring
        assert all((k.x - game._cx, k.y - game._cy) == ring[r] for k, r in zip(game._keys, game._ring_index))
        walls = np.zeros((grid, grid), bool)
        for w in level.get_sprites_by_tag("wall"):
            walls[w.y, w.x] = True
        pad = count - len(game._keys)
        levels.append(dict(
            grid_size=np.array(level.grid_size, np.int32),
            start=np.array([game._player.x, game._player.y], np.int32),
            pillar=np.array([game._cx, game._cy], np.int32),
            goal=np.array([game._goal.x, game._goal.y], np.int32),
            slots=np.array(game._ring_index + [0] * pad, np.int32),
            real=np.arange(count) < len(game._keys),
            walls=walls,
            cells=pixel_cells(game.camera, grid),
        ))
    out = {k: np.stack([lv[k] for lv in levels]) for k in levels[0]}
    out["ring"] = np.array(ring, np.int32)
    out["colors"] = np.array([int(module.sprites[name].pixels[0, 0]) for name in ("player", "pillar", "key", "goal", "wall")], np.int32)
    out["background"], out["padding"] = np.int32(module.BACKGROUND_COLOR), np.int32(module.PADDING_COLOR)
    return out


if __name__ == "__main__":
    module = load(SOURCE)
    data = extract(module.Or01(), module)
    np.savez_compressed(OUT, **data)
    print(f"{OUT}: {len(data['start'])} levels, {OUT.stat().st_size} bytes")
