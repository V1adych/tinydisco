from pathlib import Path

import orbax.checkpoint as ocp
from flax import nnx


def save_checkpoint(directory: str, step: int, model: nnx.Module) -> Path:
    # orbax requires absolute paths
    path = (Path(directory) / str(step)).absolute()
    with ocp.StandardCheckpointer() as ckptr:
        ckptr.save(path, nnx.state(model))
    return path


def load_checkpoint(path: str, model: nnx.Module):
    """Restores params saved by save_checkpoint into an already constructed model."""
    with ocp.StandardCheckpointer() as ckptr:
        state = ckptr.restore(Path(path).absolute(), nnx.state(model))
    nnx.update(model, state)
