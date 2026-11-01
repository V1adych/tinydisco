from pathlib import Path

import orbax.checkpoint as ocp
import wandb
from flax import nnx


def save_checkpoint(directory: str, step: int, model: nnx.Module) -> Path:
    # orbax requires absolute paths
    path = (Path(directory) / str(step)).absolute()
    with ocp.StandardCheckpointer() as ckptr:
        ckptr.save(path, nnx.state(model))
    return path


def fetch_checkpoint(name: str) -> tuple[Path, int]:
    """A local checkpoint directory, or a wandb artifact such as 'v1adych-team/meta-rl/meta_baseline:v2' downloaded, with the step it was saved at:
    the directory name for a local one, the step-N alias for an artifact."""
    if Path(name).exists():
        return Path(name), int(Path(name).name)
    artifact = wandb.Api().artifact(name, type="model")
    step = next(int(alias.removeprefix("step-")) for alias in artifact.aliases if alias.startswith("step-"))
    return Path(artifact.download()), step


def load_checkpoint(path: str, model: nnx.Module):
    """Restores params saved by save_checkpoint into an already constructed model."""
    with ocp.StandardCheckpointer() as ckptr:
        state = ckptr.restore(Path(path).absolute(), nnx.state(model))
    nnx.update(model, state)
