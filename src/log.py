import time
from dataclasses import asdict

import numpy as np
import wandb


class Logger:
    """Prints a one-line summary and, if `track`, sends everything to wandb."""

    def __init__(self, cfg, project: str, name: str, track: bool):
        self.run = None
        if track:
            self.run = wandb.init(project=project, name=name, config=asdict(cfg))
        self.start = time.perf_counter()

    def log(self, step: int, metrics: dict[str, float], console: list[str] | None = None):
        """console: the keys to print; by default all but the achievements."""
        metrics = {k: float(np.asarray(v)) for k, v in metrics.items()}
        if self.run is not None:
            self.run.log(metrics, step=step)
        shown = console if console is not None else [k for k in metrics if not k.startswith("achievements/")]
        print(f"[{time.perf_counter() - self.start:7.0f}s] step {step:,}  " + "  ".join(f"{k}={metrics[k]:.4g}" for k in shown), flush=True)

    def upload_checkpoint(self, path: str, step: int):
        """Uploads a saved checkpoint as a new version of the run's wandb artifact; the upload runs in the background."""
        if self.run is not None:
            artifact = wandb.Artifact(self.run.name, type="model")
            artifact.add_dir(path)
            self.run.log_artifact(artifact, aliases=[f"step-{step}"])

    def close(self):
        if self.run is not None:
            self.run.finish()
