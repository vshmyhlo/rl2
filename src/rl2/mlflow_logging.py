"""MLflow runs, metrics, and playback artifacts shared by PPO trainers."""

import os
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from time import time
from types import TracebackType
from typing import Self

import mlflow
import numpy as np
from mlflow import MlflowClient
from mlflow.entities import Metric
from moviepy import ImageSequenceClip
from numpy.typing import NDArray


class MLflowLogger:
    def __init__(self, *, experiment: str, run_name: str | None, run_dir: str, resume: bool) -> None:
        self.client = MlflowClient()
        experiment_id = os.environ.get("MLFLOW_EXPERIMENT_ID")
        selected = mlflow.set_experiment(
            experiment_id=experiment_id,
            experiment_name=None if experiment_id else os.environ.get("MLFLOW_EXPERIMENT_NAME", experiment),
        )
        directory = run_dir.rstrip("/") if "://" in run_dir else str(Path(run_dir).resolve())
        # Hash the directory so arbitrary paths never become search-filter syntax.
        key = sha256(f"{experiment}:{directory}".encode()).hexdigest()
        runs = (
            self.client.search_runs(
                [selected.experiment_id],
                filter_string=f"tags.`rl2.run_key` = '{key}'",
                order_by=["attributes.start_time DESC"],
                max_results=1,
            )
            if resume
            else []
        )
        if runs:
            self.run_id = runs[0].info.run_id
            self.client.update_run(self.run_id, status="RUNNING")
        else:
            self.run_id = self.client.create_run(
                selected.experiment_id,
                run_name=run_name,
                tags={"rl2.run_key": key, "rl2.run_dir": directory, "rl2.trainer": experiment},
            ).info.run_id

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> None:
        self.client.set_terminated(self.run_id, status="FINISHED" if exc_type is None else "FAILED")

    def log_metrics(self, metrics: dict[str, float | int], step: int) -> None:
        timestamp = int(time() * 1000)
        self.client.log_batch(
            self.run_id,
            metrics=[
                Metric(key=key, value=float(value), timestamp=timestamp, step=step) for key, value in metrics.items()
            ],
            synchronous=True,
        )

    def log_text(self, text: str, artifact_file: str) -> None:
        self.client.log_text(self.run_id, text, artifact_file)

    def log_video(self, frames: list[NDArray[np.uint8]], artifact_file: str, *, fps: float) -> None:  # each (H, W, 3)
        """Encode (H, W, 3) RGB frames as an MP4 artifact."""
        with TemporaryDirectory() as directory:
            path = Path(directory) / Path(artifact_file).name
            with ImageSequenceClip(frames, fps=fps) as clip:
                clip.write_videofile(
                    str(path),
                    codec="libx264",
                    audio=False,
                    logger=None,
                    pixel_format="yuv420p",
                    # Browser-compatible H.264 needs even dimensions.
                    ffmpeg_params=["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"],
                )
            self.client.log_artifact(self.run_id, str(path), artifact_path=str(Path(artifact_file).parent))
