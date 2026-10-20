"""Training jobs, live Ray dashboards, and saved timeline snapshots."""

import argparse
import errno
import json
import os
import re
import shutil
import ssl
import stat
from contextlib import ExitStack, closing
from email.utils import formatdate
from functools import partial
from http.client import HTTPConnection, HTTPException
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from ipaddress import ip_address
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

SERVICE_ACCOUNT = Path("/var/run/secrets/kubernetes.io/serviceaccount")
HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


class Cluster:
    def __init__(self, namespace: str = "training") -> None:
        self.namespace = namespace

    def list_resources(self, resource: str) -> list[dict[str, Any]]:
        api = "apis/batch/v1" if resource == "jobs" else "api/v1"
        selector = "kueue.x-k8s.io/queue-name" if resource == "jobs" else "batch.kubernetes.io/job-name"
        query = urlencode({"labelSelector": selector})
        request = Request(
            f"https://kubernetes.default.svc/{api}/namespaces/{self.namespace}/{resource}?{query}",
            headers={"Authorization": f"Bearer {SERVICE_ACCOUNT.joinpath('token').read_text().strip()}"},
        )
        context = ssl.create_default_context(cafile=str(SERVICE_ACCOUNT / "ca.crt"))
        with urlopen(request, context=context, timeout=10) as response:
            return json.load(response)["items"]

    def jobs(self) -> list[dict[str, Any]]:
        pods = self.list_resources("pods")
        jobs = []
        for job in self.list_resources("jobs"):
            metadata, spec, status = job["metadata"], job["spec"], job.get("status", {})
            if any(
                condition["type"] in {"Complete", "Failed"} and condition["status"] == "True"
                for condition in status.get("conditions", [])
            ):
                continue
            running = [
                pod
                for pod in pods
                if pod.get("status", {}).get("phase") == "Running"
                and not pod["metadata"].get("deletionTimestamp")
                and any(
                    owner["uid"] == metadata["uid"] and owner.get("kind") == "Job"
                    for owner in pod["metadata"].get("ownerReferences", [])
                )
            ]
            running.sort(key=lambda pod: pod["metadata"].get("creationTimestamp", ""), reverse=True)
            pod = running[0] if running else None
            jobs.append(
                {
                    "name": metadata["name"],
                    "run_id": metadata.get("annotations", {}).get("rl2.local/run-id", ""),
                    "created": metadata.get("creationTimestamp", ""),
                    "status": "Running" if pod else "Queued" if spec.get("suspend") else "Starting",
                    "pod_ip": pod["status"].get("podIP") if pod else None,
                }
            )
        return sorted(jobs, key=lambda job: job["created"], reverse=True)


class TimelineHandler(BaseHTTPRequestHandler):
    def __init__(self, *args: Any, root: Path, cluster: Cluster | None = None, **kwargs: Any) -> None:
        self.root = root
        self.cluster = cluster
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/health":
            self.respond(b"ok\n", "text/plain")
        elif path == "/":
            self.respond(Path(__file__).with_name("index.html").read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/jobs":
            jobs = self.cluster_jobs()
            if jobs is not None:
                self.respond(
                    json.dumps(
                        [{key: value for key, value in job.items() if key != "pod_ip"} for job in jobs]
                    ).encode(),
                    "application/json",
                )
        elif path.startswith("/dashboard/"):
            self.proxy_dashboard()
        elif re.fullmatch(r"/traces/[A-Za-z0-9_-]+\.json", path):
            self.serve_trace(path.rsplit("/", 1)[1])
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        if re.fullmatch(r"/dashboard/[a-z0-9-]+/api/authenticate", urlsplit(self.path).path):
            self.proxy_dashboard()
        else:
            self.send_error(405)

    def do_HEAD(self) -> None:
        if urlsplit(self.path).path.startswith("/dashboard/"):
            self.proxy_dashboard()
        else:
            self.send_error(404)

    def cluster_jobs(self) -> list[dict[str, Any]] | None:
        if self.cluster is None:
            self.send_error(503, "Job discovery is available when the viewer runs inside Kubernetes.")
            return None
        try:
            return self.cluster.jobs()
        except (OSError, URLError, ValueError, KeyError):
            self.send_error(503, "Could not list training jobs. Check the viewer's Kubernetes permissions.")
            return None

    def proxy_dashboard(self) -> None:
        url = urlsplit(self.path)
        match = re.fullmatch(r"/dashboard/([a-z0-9-]+)(/.*)?", url.path)
        if match is None:
            self.send_error(404)
            return
        name, path = match.groups()
        prefix = f"/dashboard/{name}/"
        if path is None:
            self.send_response(307)
            self.send_header("Location", prefix + ("?" + url.query if url.query else ""))
            self.end_headers()
            return
        jobs = self.cluster_jobs()
        if jobs is None:
            return
        job = next((job for job in jobs if job["name"] == name), None)
        if job is None:
            self.send_error(404, "This job is finished or no longer exists. Saved timelines remain available.")
            return
        if not job["pod_ip"]:
            self.send_error(503, "This job is queued or starting. Refresh after Ray starts.")
            return
        try:
            host = str(ip_address(job["pod_ip"]))
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_error(400)
            return
        if self.headers.get("Transfer-Encoding") or not 0 <= length <= 65536:
            self.send_error(413)
            return
        blocked = (
            HOP_HEADERS
            | {"host"}
            | {header.strip().lower() for header in self.headers.get("Connection", "").split(",")}
        )
        headers = {key: value for key, value in self.headers.items() if key.lower() not in blocked}
        body = self.rfile.read(length) if length else None
        started = False
        try:
            with closing(HTTPConnection(host, 8265, timeout=30)) as upstream:
                upstream.request(
                    self.command, path + ("?" + url.query if url.query else ""), body=body, headers=headers
                )
                with upstream.getresponse() as response:
                    self.send_response(response.status)
                    blocked = (
                        HOP_HEADERS
                        | {"server", "date", "cache-control", "x-content-type-options"}
                        | {header.strip().lower() for header in response.getheader("Connection", "").split(",")}
                    )
                    for key, value in response.getheaders():
                        if key.lower() in blocked:
                            continue
                        if key.lower() == "set-cookie":
                            cookies = SimpleCookie()
                            cookies.load(value)
                            for cookie in cookies.values():
                                cookie["path"] = prefix
                                cookie["domain"] = ""
                                self.send_header(key, cookie.OutputString())
                        elif key.lower() == "location" and value.startswith("/"):
                            self.send_header(key, prefix + value.lstrip("/"))
                        else:
                            self.send_header(key, value)
                    self.end_headers()
                    started = True
                    while chunk := response.read1(65536):
                        self.wfile.write(chunk)
                        self.wfile.flush()
        except (OSError, HTTPException):
            if not started:
                self.send_error(502, "Ray dashboard is not ready or has stopped. Refresh to retry.")
            self.close_connection = True

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
    parser.add_argument("--namespace", default="training")
    args = parser.parse_args()
    cluster = Cluster(namespace=args.namespace) if SERVICE_ACCOUNT.joinpath("token").exists() else None
    with ThreadingHTTPServer(
        (args.host, args.port), partial(TimelineHandler, root=args.root, cluster=cluster)
    ) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
