from pathlib import Path
from unittest.mock import patch

import mlflow
import numpy as np
import pytest
from mlflow import MlflowClient
from moviepy import VideoFileClip

from rl2.mlflow_logging import MLflowLogger


@pytest.fixture(scope="module")
def tracking_uri(tmp_path_factory: pytest.TempPathFactory) -> str:
    directory = tmp_path_factory.mktemp("mlflow")
    uri = f"sqlite:///{directory / 'tracking.db'}"
    client = MlflowClient(tracking_uri=uri)
    client.create_experiment("logging-tests", artifact_location=(directory / "artifacts").as_uri())
    return uri


@pytest.fixture
def client(tracking_uri: str, monkeypatch: pytest.MonkeyPatch) -> MlflowClient:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "logging-tests")
    monkeypatch.delenv("MLFLOW_EXPERIMENT_ID", raising=False)
    return MlflowClient()


def test_metrics_artifacts_and_resume(client: MlflowClient, tmp_path: Path) -> None:
    # Resuming a pre-MLflow checkpoint must also create a run when none exists.
    with MLflowLogger(experiment="ppo", run_name="training", run_dir=str(tmp_path), resume=True) as logger:
        run_id = logger.run_id
        assert client.get_run(run_id).info.status == "RUNNING"
        assert client.get_run(run_id).info.run_name == "training"
        logger.log_metrics({"charts/return_mean": 2.5, "charts/total_episodes": 1}, step=4)
        logger.log_text("seed: 0\n", "config/4.yaml")
        logger.log_text('{"return_mean": 2.5}', "eval/report/4.json")
        # A single frame with odd dimensions exercises the MP4 encoder's boundaries.
        frames = [np.zeros((3, 5, 3), dtype=np.uint8)]  # entries: (H=3, W=5, C=3)
        logger.log_video(frames, "eval/gameplay/4.mp4", fps=12.5)
    assert client.get_run(run_id).info.status == "FINISHED"
    assert client.get_run(run_id).data.metrics["charts/total_episodes"] == 1
    config = Path(client.download_artifacts(run_id, "config/4.yaml"))
    assert config.read_text() == "seed: 0\n"
    report = Path(client.download_artifacts(run_id, "eval/report/4.json"))
    assert report.read_text() == '{"return_mean": 2.5}'
    with VideoFileClip(client.download_artifacts(run_id, "eval/gameplay/4.mp4")) as clip:
        assert clip.size == [6, 4]
        assert clip.fps == 12.5
        assert clip.duration == pytest.approx(1 / 12.5)
        assert clip.get_frame(0).shape == (4, 6, 3)

    with (
        pytest.raises(RuntimeError, match="interrupted"),
        MLflowLogger(experiment="ppo", run_name="training", run_dir=str(tmp_path), resume=True) as resumed,
    ):
        assert resumed.run_id == run_id
        assert client.get_run(run_id).info.status == "RUNNING"
        resumed.log_metrics({"charts/return_mean": 3.5}, step=8)
        raise RuntimeError("interrupted")
    assert client.get_run(run_id).info.status == "FAILED"
    assert [(metric.value, metric.step) for metric in client.get_metric_history(run_id, "charts/return_mean")] == [
        (2.5, 4),
        (3.5, 8),
    ]
    with MLflowLogger(experiment="ppo", run_name="training", run_dir=str(tmp_path), resume=True) as resumed:
        assert resumed.run_id == run_id
    assert client.get_run(run_id).info.status == "FINISHED"
    with MLflowLogger(experiment="ppo", run_name="training", run_dir=str(tmp_path), resume=False) as fresh:
        assert fresh.run_id != run_id


def test_experiment_id_and_cloud_run_directory(client: MlflowClient, monkeypatch: pytest.MonkeyPatch) -> None:
    experiment = client.get_experiment_by_name("logging-tests")
    assert experiment is not None
    monkeypatch.setenv("MLFLOW_EXPERIMENT_ID", experiment.experiment_id)
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "ignored")
    directory = "gs://bucket/run's-directory"
    with MLflowLogger(experiment="mappo", run_name=None, run_dir=directory, resume=False) as logger:
        run_id = logger.run_id
        assert client.get_run(run_id).data.tags["rl2.run_dir"] == directory
    with MLflowLogger(experiment="mappo", run_name=None, run_dir=directory + "/", resume=True) as resumed:
        assert resumed.run_id == run_id
    with MLflowLogger(experiment="ppo", run_name=None, run_dir=directory, resume=True) as separate:
        assert separate.run_id != run_id


def test_default_experiment_and_artifact_failure(
    client: MlflowClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME")
    # Keep even the automatically created experiment's artifacts inside the test directory.
    monkeypatch.chdir(tmp_path)
    with (
        patch.object(mlflow, "set_experiment", wraps=mlflow.set_experiment) as select,
        pytest.raises(RuntimeError, match="upload"),
        MLflowLogger(experiment="ppo", run_name=None, run_dir=str(tmp_path), resume=False) as logger,
        patch.object(logger.client, "log_artifact", side_effect=RuntimeError("upload")) as upload,
    ):
        frames = [np.zeros((2, 2, 3), dtype=np.uint8)]  # entries: (H=2, W=2, C=3)
        logger.log_video(frames, "gameplay/0.mp4", fps=10)
    select.assert_called_once_with(experiment_id=None, experiment_name="ppo")
    assert client.get_run(logger.run_id).info.status == "FAILED"
    assert not Path(upload.call_args.args[1]).exists()
