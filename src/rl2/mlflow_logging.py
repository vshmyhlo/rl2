"""MLflow runs, console logs, metrics, and playback artifacts shared by PPO trainers."""

import os
import posixpath
import sys
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from hashlib import sha256
from io import TextIOBase
from math import inf, isfinite, nextafter
from pathlib import Path
from tempfile import TemporaryDirectory
from time import time
from traceback import print_exception
from types import TracebackType
from typing import Any, Self, TextIO
from uuid import uuid4

import numpy as np
from mlflow import MlflowClient
from mlflow.entities import Metric, Param
from mlflow.exceptions import MlflowException
from moviepy import ImageSequenceClip
from numpy.typing import NDArray


class _Tee(TextIOBase):
    def __init__(self, stream: TextIO, log: TextIO) -> None:
        self.stream = stream
        self.log = log

    def write(self, text: str) -> int:
        self.log.write(text)
        return self.stream.write(text)

    def flush(self) -> None:
        self.stream.flush()
        if not self.log.closed:
            self.log.flush()

    def isatty(self) -> bool:
        return self.stream.isatty()

    def fileno(self) -> int:
        return self.stream.fileno()

    @property
    def encoding(self) -> str:
        return self.stream.encoding


class MLflowLogger:
    def __init__(self, *, experiment: str, run_name: str | None, resume: bool) -> None:
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
        # Hash the name so arbitrary names never become search-filter syntax.
        key = sha256(f"{experiment}:{run_name}".encode()).hexdigest()
        runs = (
            self.client.search_runs(
                [experiment_id],
                filter_string=f"tags.`rl2.run_key` = '{key}'",
                order_by=["attributes.start_time DESC"],
                max_results=1,
            )
            if resume and run_name is not None
            else []
        )
        if runs:
            self.run_id = runs[0].info.run_id
            self.client.update_run(self.run_id, status="RUNNING")
        else:
            self.run_id = self.client.create_run(
                experiment_id,
                run_name=run_name,
                tags={"rl2.run_key": key, "rl2.trainer": experiment},
            ).info.run_id
        self._console_resources = ExitStack()
        self._console_file: TextIO | None = None
        self._uploaded_size = 0

    def __enter__(self) -> Self:
        with ExitStack() as resources:
            directory = resources.enter_context(TemporaryDirectory())
            self._console_path = Path(directory) / f"{uuid4().hex}.log"
            self._console_file = resources.enter_context(self._console_path.open("w", encoding="utf-8"))
            resources.enter_context(redirect_stdout(_Tee(stream=sys.stdout, log=self._console_file)))
            resources.enter_context(redirect_stderr(_Tee(stream=sys.stderr, log=self._console_file)))
            self._console_resources = resources.pop_all()
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> None:
        failure = exc
        try:
            with self._console_resources:
                if exc is not None:
                    print_exception(exc_type, exc, traceback, file=self._console_file)
                self._upload_console()
        except Exception as error:  # noqa: BLE001 -- Terminate the run before propagating the failure.
            if failure is None:
                failure = error
            else:
                failure.add_note(f"MLflow console upload also failed: {error}")
        try:
            self.client.set_terminated(self.run_id, status="FINISHED" if failure is None else "FAILED")
        except Exception as error:
            if failure is None:
                raise
            failure.add_note(f"MLflow run {self.run_id} termination also failed: {error}")
        if exc is None and failure is not None:
            raise failure

    def _upload_console(self) -> None:
        if self._console_file is None or self._console_file.closed:
            return
        self._console_file.flush()
        size = self._console_file.tell()
        if size != self._uploaded_size:
            self.client.log_artifact(self.run_id, str(self._console_path), artifact_path="logs")
            self._uploaded_size = size

    def log_metrics(self, metrics: dict[str, float | int], step: int) -> None:
        timestamp = int(time() * 1000)
        self.client.log_batch(
            self.run_id,
            metrics=[
                Metric(key=key, value=float(value), timestamp=timestamp, step=step) for key, value in metrics.items()
            ],
            synchronous=True,
        )
        self._upload_console()

    def log_text(self, text: str, artifact_file: str) -> None:
        self.client.log_text(self.run_id, text, artifact_file)

    def log_params(self, config: dict[str, Any]) -> None:
        """Log nested config with dotted keys, preserving existing parameters on resume."""
        existing = self.client.get_run(self.run_id).data.params
        params: list[Param] = []

        def collect(values: dict[str, Any], prefix: str = "") -> None:
            for name, value in values.items():
                key = f"{prefix}.{name}" if prefix else name
                if isinstance(value, dict):
                    collect(value, key)
                elif key not in existing:
                    params.append(Param(key=key, value=str(value)))

        collect(config)
        if params:
            self.client.log_batch(self.run_id, params=params, synchronous=True)

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
