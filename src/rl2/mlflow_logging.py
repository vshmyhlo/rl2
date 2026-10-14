"""MLflow runs, metrics, and playback artifacts shared by PPO trainers."""

import os
import posixpath
from hashlib import sha256
from math import inf, isfinite, nextafter
from pathlib import Path
from tempfile import TemporaryDirectory
from time import time
from types import TracebackType
from typing import Self

import numpy as np
from mlflow import MlflowClient
from mlflow.entities import Metric
from mlflow.exceptions import MlflowException
from moviepy import ImageSequenceClip
from numpy.typing import NDArray


class MLflowLogger:
    def __init__(self, *, experiment: str, run_name: str | None, run_dir: str, resume: bool) -> None:
        self.client = MlflowClient()
        experiment_id = os.environ.get("MLFLOW_EXPERIMENT_ID")
        # Resolve through the client without changing MLflow's global experiment or environment.
        if not experiment_id:
            experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", experiment)
            selected = self.client.get_experiment_by_name(experiment_name)
            if selected is None:
                try:
                    experiment_id = self.client.create_experiment(experiment_name)
                except MlflowException as error:
                    # Another trainer may have created the experiment concurrently.
                    if error.error_code != "RESOURCE_ALREADY_EXISTS":
                        raise
                    selected = self.client.get_experiment_by_name(experiment_name)
                    if selected is None:
                        raise
                    experiment_id = selected.experiment_id
            else:
                experiment_id = selected.experiment_id
        directory = run_dir.rstrip("/") if "://" in run_dir else str(Path(run_dir).resolve())
        # Hash the directory so arbitrary paths never become search-filter syntax.
        key = sha256(f"{experiment}:{directory}".encode()).hexdigest()
        runs = (
            self.client.search_runs(
                [experiment_id],
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
                experiment_id,
                run_name=run_name,
                tags={"rl2.run_key": key, "rl2.run_dir": directory, "rl2.trainer": experiment},
            ).info.run_id

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> None:
        try:
            self.client.set_terminated(self.run_id, status="FINISHED" if exc_type is None else "FAILED")
        except Exception as error:
            if exc is None:
                raise
            exc.add_note(f"MLflow run {self.run_id} termination also failed: {error}")

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
        if not frames:
            raise ValueError("Video must contain at least one frame")
        if not isfinite(fps) or fps <= 0:
            raise ValueError("Video fps must be finite and positive")
        artifact_path, filename = posixpath.split(artifact_file)
        with TemporaryDirectory() as directory:
            path = Path(directory) / filename
            with ImageSequenceClip(frames, fps=fps) as clip:
                # MoviePy truncates duration * fps; round upward to preserve the final frame.
                clip.duration = nextafter(len(frames) / fps, inf)
                clip.write_videofile(
                    str(path),
                    codec="libx264",
                    audio=False,
                    logger=None,
                    # Set the output format explicitly; MoviePy's pixel_format controls the input.
                    ffmpeg_params=["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p"],
                )
            self.client.log_artifact(self.run_id, str(path), artifact_path=artifact_path or None)
