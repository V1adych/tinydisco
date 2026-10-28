"""Turns the community ARC-AGI-3 game hn01 (originals/hn01.py, run with the arcengine package) into the arrays game.py plays
on:  python src/envs/arc/games/hn01/extract.py  -> hn01.npz next to it.
Per level: the grid size and the camera's grid -> frame mapping: each of the 64x64 frame pixels as the index of the grid
cell it shows (row * GRID + column), -1 on the letterbox padding (the 10x12 grid is scaled by 5 and centered); plus the
pegs' columns and base row and the disks' and pegs' colors."""

import importlib.util
import sys
from pathlib import Path

import numpy as np

SOURCE = Path(__file__).parents[2] / "originals/hn01.py"
OUT = Path(__file__).with_name("hn01.npz")
SIZE = 64


def load(source: Path):
    spec = importlib.util.spec_from_file_location("hn01_original", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules["hn01_original"] = module
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
    levels = []
    for i in range(len(module.levels)):
        game.set_level(i)
        levels.append(dict(grid_size=np.array(game.current_level.grid_size, np.int32), cells=pixel_cells(game.camera, grid)))
    out = {k: np.stack([lv[k] for lv in levels]) for k in levels[0]}
    out["peg_x"], out["base_y"] = np.array(module.PEGX, np.int32), np.int32(module.BASEY)
    out["colors"] = np.array([int(module.s[name].pixels[0, 0]) for name in ("d1", "d2", "d3", "peg")], np.int32)
    out["background"], out["padding"] = np.int32(module.BG), np.int32(module.PAD)
    return out


if __name__ == "__main__":
    module = load(SOURCE)
    data = extract(module.Hn01(), module)
    np.savez_compressed(OUT, **data)
    print(f"{OUT}: {len(data['cells'])} levels, {OUT.stat().st_size} bytes")
