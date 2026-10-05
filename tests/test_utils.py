from pathlib import Path

import pytest

from rl2 import utils


def test_checkpoint_io_preserves_previous_file_and_propagates_storage_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "checkpoint.msgpack"
    destination.write_bytes(b"old")

    def failed_replace(source: Path, target: Path) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(utils.os, "replace", failed_replace)
    with pytest.raises(OSError, match="replace failure"):
        utils.write_bytes(str(destination), b"new")
    assert destination.read_bytes() == b"old"
    assert list(tmp_path.iterdir()) == [destination]
    assert utils.read_optional(str(tmp_path / "missing")) is None

    def denied(path: str) -> bytes:
        raise PermissionError("storage access denied")

    monkeypatch.setattr(utils, "read_bytes", denied)
    with pytest.raises(PermissionError, match="denied"):
        utils.read_optional("gs://bucket/run/checkpoint.msgpack")
