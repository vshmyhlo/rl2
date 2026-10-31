# Stateful Python sandbox

`PythonSession` owns one gVisor pod and one persistent Python interpreter. Use
one session per rollout; variables, imports, functions, and `/tmp` files persist
between calls. Exiting the context deletes the pod.

```python
from rl2.sandbox.kubernetes import PythonSession

async with PythonSession() as session:
    await session.execute("import math; numbers = [12, 18, 24]")
    result = await session.execute("print(math.gcd(*numbers))")
    print(result.output)  # 6
```

`execute()` returns `PythonResult(output, exit_code, output_truncated)` with
combined stdout/stderr, including native writes and child-process output.
Print values explicitly. Python exceptions return `exit_code=1` and preserve
state; successful calls return zero. Timeout, cancellation of an executing call,
process exit, or invalid protocol closes the session. Calls are serialized;
their execution deadline starts after acquiring the session lock. `aclose()` is
also available and is idempotent.
Execution failures raise `PythonExecutionError`; execution deadlines raise its
subclass `PythonExecutionTimeout` (also a `TimeoutError`). Startup, cleanup,
and cancellation errors propagate separately so callers cannot mistake failed
pod deletion for a normal execution failure.

## Prerequisites

- A dedicated `sandbox` namespace and a **trusted** `runsc-no-network`
  RuntimeClass backed by `io.containerd.runsc.v1` with `[runsc_config] network = "none"`.
  See [gVisor configuration](https://gvisor.dev/docs/user_guide/containerd/configuration/)
  and [k3s containerd templates](https://docs.k3s.io/advanced#configuring-containerd).
  The class does not install runtimes or restart k3s. It fails if the runtime is
  missing or the interpreter sees any interface other than loopback.
- `kubectl` and credentials in the **trusted caller**, with namespaced
  create/get/list/watch/delete permissions on pods and create/get on pods/exec.
  For local k3s administration, pass `kubectl=("sudo", "-n", "k3s", "kubectl")`.
- A trusted image containing `python` and `sleep`; default: `python:3.12-slim-bookworm`.
  Preload it on the nodes for faster startup; pin a digest for reproducibility.
  Never use an image containing training data, credentials, or scoring code.

No Docker daemon/socket is used. One persistent `kubectl exec` connection carries
code and responses; the sandbox does not listen on a network port. The caller
owns Kubernetes credentials; sandbox pods have no service-account token, host
mounts, or GPU access. Add a namespace deny-all NetworkPolicy as defense in depth.

Defaults: 90-second startup, 5-second calls, 30-second pod cleanup, 30-minute pod
lifetime, 1 CPU/256 MiB limits, 16 MiB memory-backed `/tmp`, and 16 KiB code/output
limits. The root filesystem is read-only and Python runs as a non-root user.
Cleanup also allows up to 10 seconds to stop the local exec client. The pod's
lifetime deadline bounds orphan execution if the caller crashes; terminated pod
objects still need deletion. API/node failures can prevent cleanup and are
reported. This class does not provide a pool, global concurrency limits, or
automatic recovery of a failed session.

The worker executes arbitrary code inside gVisor; its response is untrusted tool
output, never a score or proof of correct execution. Background processes that
keep a call's output pipe open can cause that call to time out and close the session.

Run focused tests with `uv run pytest tests/test_python_session.py`.
Set `RL2_TEST_K3S_SANDBOX=1` to include the live test after configuring the cluster.
