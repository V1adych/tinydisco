"""Turns the community ARC-AGI-3 game nw01 (originals/nw01.py, run with the arcengine package) into the arrays game.py plays
on:  python src/envs/arc/games/nw01/extract.py  -> nw01.npz next to it.
Per level, on a grid padded to the largest level (GRID x GRID): the player's start, the arrows' cells and directions
(padded with GRID, which every lookup drops), the target's cell, the wall mask, and the camera's grid -> frame mapping: each
of the 64x64 frame pixels as the index of the grid cell it shows (row * GRID + column), -1 on the letterbox padding."""

import importlib.util
import sys
from pathlib import Path

import numpy as np

SOURCE = Path(__file__).parents[2] / "originals/nw01.py"
OUT = Path(__file__).with_name("nw01.npz")
SIZE = 64


def load(source: Path):
    spec = importlib.util.spec_from_file_location("nw01_original", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules["nw01_original"] = module
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
    slots = max(len(level.get_sprites_by_tag("arrow")) for level in module.levels)
    levels = []
    for i in range(len(module.levels)):
        game.set_level(i)
        level = game.current_level
        at = lambda tag: [[s.x, s.y] for s in level.get_sprites_by_tag(tag)][0]  # one player and one target per level
        arrows = list(game._arrow_map.items())
        pad = [[grid, grid]] * (slots - len(arrows))
        walls = np.zeros((grid, grid), bool)
        for s in level.get_sprites_by_tag("wall"):
            walls[s.y, s.x] = True
        levels.append(dict(
            grid_size=np.array(level.grid_size, np.int32),
            start=np.array(at("player"), np.int32),
            arrows=np.array([list(cell) for cell, _ in arrows] + pad, np.int32),
            vectors=np.array([list(vec) for _, vec in arrows] + [[0, 0]] * (slots - len(arrows)), np.int32),
            target=np.array(at("target"), np.int32),
            walls=walls,
            cells=pixel_cells(game.camera, grid),
        ))
    out = {k: np.stack([lv[k] for lv in levels]) for k in levels[0]}
    out["colors"] = np.array([int(module.sprites[name].pixels[0, 0]) for name in ("player", "arrow", "target", "wall")], np.int32)
    out["background"], out["padding"] = np.int32(module.BACKGROUND_COLOR), np.int32(module.PADDING_COLOR)
    return out


if __name__ == "__main__":
    module = load(SOURCE)
    data = extract(module.Nw01(), module)
    np.savez_compressed(OUT, **data)
    print(f"{OUT}: {len(data['start'])} levels, {OUT.stat().st_size} bytes")
