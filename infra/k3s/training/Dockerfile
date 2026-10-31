# syntax=docker/dockerfile:1.19
FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git ffmpeg libegl1 libgl1 libopengl0 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.12.21 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never PYTHONUNBUFFERED=1
# Request host rendering drivers at container creation, before uv loads nvida.env.
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
# Keep the large dependency layer independent of source and documentation edits.
COPY pyproject.toml uv.lock ./
ARG UV_EXTRA=""
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project ${UV_EXTRA:+--extra=$UV_EXTRA}
# FlashInfer expects a conventional toolkit layout; CUDA wheels use nvidia/cu13.
ENV CUDA_HOME=/app/.venv/lib/python3.12/site-packages/nvidia/cu13
RUN test -x "$CUDA_HOME/bin/nvcc" \
    && ln -s lib "$CUDA_HOME/lib64" \
    && ln -s libcudart.so.13 "$CUDA_HOME/lib/libcudart.so"
RUN printf '#include <cuda_runtime.h>\nstatic_assert(CUDART_VERSION / 10 == __CUDACC_VER_MAJOR__ * 100 + __CUDACC_VER_MINOR__);\n' \
    | "$CUDA_HOME/bin/nvcc" -x cu -c -o /tmp/cuda-check.o - \
    && rm /tmp/cuda-check.o
# Bundle Panda assets so Playground can start without runtime downloads.
RUN --mount=type=bind,source=src/rl2/envs/playground_assets.py,target=/tmp/playground_assets.py \
    /app/.venv/bin/python /tmp/playground_assets.py
# Keep local datasets out of the layer rebuilt for each source edit.
COPY data/ ./data/
# Install editable metadata without copying source into an intermediate layer.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=README.md,target=/app/README.md \
    --mount=type=bind,source=src,target=/app/src \
    uv sync --locked --no-dev ${UV_EXTRA:+--extra=$UV_EXTRA}
COPY --from=registry.k8s.io/kubectl:v1.36.5 /bin/kubectl /usr/local/bin/kubectl
RUN kubectl version --client
COPY --exclude=data . .
# Preserve the built environment when a command uses uv run.
ENV UV_NO_SYNC=1 UV_ENV_FILE=/app/nvida.env PATH="/app/.venv/bin:${CUDA_HOME}/bin:$PATH"
# Ray workers share the installed image environment, including optional extras.
ENV RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 RAY_DEDUP_LOGS=0
