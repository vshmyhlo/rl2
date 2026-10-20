"""Read-only HTTP access to Ray timeline snapshots."""

import argparse
import errno
import os
import re
import shutil
import stat
from contextlib import ExitStack
from email.utils import formatdate
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


class TimelineHandler(BaseHTTPRequestHandler):
    def __init__(self, *args: Any, root: Path, **kwargs: Any) -> None:
        self.root = root
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/health":
            self.respond(b"ok\n", "text/plain")
        elif path == "/":
            self.respond(Path(__file__).with_name("index.html").read_bytes(), "text/html; charset=utf-8")
        elif re.fullmatch(r"/traces/[A-Za-z0-9_-]+\.json", path):
            self.serve_trace(path.rsplit("/", 1)[1])
        else:
            self.send_error(404)

    def respond(self, body: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        super().end_headers()

    def serve_trace(self, name: str) -> None:
        with ExitStack() as resources:
            try:
                descriptor = os.open(self.root / name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            except OSError as error:
                self.send_error(404 if error.errno == errno.ENOENT else 403)
                return
            resources.callback(os.close, descriptor)
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode):
                self.send_error(403)
                return
            trace = resources.enter_context(os.fdopen(descriptor, "rb", closefd=False))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(info.st_size))
            self.send_header("Last-Modified", formatdate(info.st_mtime, usegmt=True))
            self.end_headers()
            shutil.copyfileobj(trace, self.wfile)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/runs/ray-timelines"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8085)
    args = parser.parse_args()
    with ThreadingHTTPServer((args.host, args.port), partial(TimelineHandler, root=args.root)) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
