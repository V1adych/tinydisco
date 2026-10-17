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
        self.start = time.time()

    def log(self, step: int, metrics: dict[str, float]):
        metrics = {k: float(np.asarray(v)) for k, v in metrics.items()}
        if self.run is not None:
            self.run.log(metrics, step=step)
        brief = {k: v for k, v in metrics.items() if not k.startswith("achievements/")}
        print(f"[{time.time() - self.start:7.0f}s] step {step:,}  " + "  ".join(f"{k}={v:.4g}" for k, v in brief.items()), flush=True)

    def close(self):
        if self.run is not None:
            self.run.finish()
