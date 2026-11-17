"""Download checksummed source archives without creating Git checkouts."""

import hashlib
import json
import shutil
import sys
import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory, TemporaryFile
from urllib.request import urlopen


def extract_source(url: str, sha256: str, destination: Path) -> None:
    with TemporaryFile() as archive:
        with urlopen(url, timeout=120) as response:
            shutil.copyfileobj(response, archive)
        archive.seek(0)
        if hashlib.file_digest(archive, "sha256").hexdigest() != sha256:
            raise ValueError(f"Source checksum mismatch: {url}")
        archive.seek(0)
        with TemporaryDirectory() as directory, tarfile.open(fileobj=archive, mode="r:gz") as source:
            source.extractall(directory, filter="data")
            (root,) = Path(directory).iterdir()
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(root, destination, dirs_exist_ok=True)


def main() -> None:
    manifest, destination = map(Path, sys.argv[1:])
    for source in json.loads(manifest.read_text()):
        print(f"Downloading {source['path']}", flush=True)
        extract_source(source["url"], source["sha256"], destination / source["path"])


if __name__ == "__main__":
    main()
