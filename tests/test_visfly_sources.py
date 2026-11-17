"""Checksummed simulator source extraction."""

import hashlib
import io
import runpy
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

sources = SimpleNamespace(
    **runpy.run_path(str(Path(__file__).resolve().parents[1] / "infra/k3s/training/visfly/fetch_sources.py"))
)


def test_source_archive_and_checksum(tmp_path: Path) -> None:
    archive = tmp_path / "source.tar.gz"
    contents = b"pinned source\n"
    with tarfile.open(archive, "w:gz") as output:
        entry = tarfile.TarInfo("project-revision/source.txt")
        entry.size = len(contents)
        output.addfile(entry, io.BytesIO(contents))
    checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
    destination = tmp_path / "sources" / "project"
    sources.extract_source(archive.as_uri(), checksum, destination)
    assert (destination / "source.txt").read_bytes() == contents
    invalid_destination = tmp_path / "invalid"
    with pytest.raises(ValueError, match="checksum mismatch"):
        sources.extract_source(archive.as_uri(), "0" * 64, invalid_destination)
    assert not invalid_destination.exists()
