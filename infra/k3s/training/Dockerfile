# syntax=docker/dockerfile:1.19
FROM python:3.12-slim-bookworm AS visfly-sources
COPY infra/k3s/training/visfly/fetch_sources.py infra/k3s/training/visfly/sources.json /tmp/visfly/
RUN python /tmp/visfly/fetch_sources.py /tmp/visfly/sources.json /opt

FROM python:3.12-slim-bookworm AS habitat-build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build libegl1-mesa-dev libgl1-mesa-dev \
    libcgal-dev libeigen3-dev libassimp-dev rapidjson-dev libzstd-dev \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir setuptools==75.8.0 wheel==0.45.1 pybind11==2.13.6
COPY --from=visfly-sources /opt/habitat-sim /opt/habitat-sim
WORKDIR /opt/habitat-sim
# Pybind11 2.13 supplies NumPy 2 support; the fork retains an obsolete NumPy pin.
RUN sed -i 's/numpy>=1.20.0,<1.24.0/numpy>=2,<3/' requirements.txt
COPY infra/k3s/training/visfly/route_collision.h /opt/habitat-sim/src/esp/bindings/route_collision.h
COPY infra/k3s/training/visfly/route_collision.patch /tmp/visfly/route_collision.patch
RUN patch --strip=1 < /tmp/visfly/route_collision.patch
COPY infra/k3s/training/visfly/rgbd.patch /tmp/visfly/rgbd.patch
RUN patch --strip=1 < /tmp/visfly/rgbd.patch
RUN --mount=type=cache,target=/opt/habitat-sim/build,sharing=locked \
    python setup.py --headless --bullet --no-update-submodules --skip-install-magnum \
    --build-type Release \
    --cmake-args="-DUSE_SYSTEM_EIGEN=ON -DUSE_SYSTEM_ASSIMP=ON -DUSE_SYSTEM_OPENEXR=ON -DUSE_SYSTEM_RAPIDJSON=OFF -DUSE_SYSTEM_ZSTD=ON -DUSE_SYSTEM_PYBIND11=ON -Dpybind11_DIR=/usr/local/lib/python3.12/site-packages/pybind11/share/cmake/pybind11" \
    build_ext --parallel 8 build_py bdist_wheel --skip-build --dist-dir /wheels \
    && pip wheel --no-deps --no-build-isolation build/deps/magnum-bindings/src/python -w /wheels

# Scene downloads do not depend on native builds or VisFly source patches.
FROM python:3.12-slim-bookworm AS visfly-assets
RUN pip install --no-cache-dir huggingface-hub==1.33.0
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('LiFanxing/visfly-beta', repo_type='dataset', revision='04ad03b9a2a079891c0ae10fd99e3e1b35bc5824', local_dir='/opt/VisFly/datasets/visfly-beta', ignore_patterns=['.git*', '.vscode/*'])"
COPY infra/k3s/training/visfly/prepare_assets.py /tmp/visfly/prepare_assets.py
RUN python /tmp/visfly/prepare_assets.py /opt/VisFly/datasets/visfly-beta

FROM python:3.12-slim-bookworm AS training-base
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git ffmpeg libegl1 libgl1 libopengl0 libglib2.0-0 libgomp1 libibverbs1 \
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
# Bundle replacement sprites without an interactive ROM ownership confirmation.
RUN --mount=type=bind,source=src/rl2/envs/jaxtari_assets.py,target=/tmp/jaxtari_assets.py \
    JAX_PLATFORMS=cpu /app/.venv/bin/python /tmp/jaxtari_assets.py
# FlashInfer expects a conventional toolkit layout; CUDA wheels use nvidia/cu13.
ENV CUDA_HOME=/app/.venv/lib/python3.12/site-packages/nvidia/cu13
# cuDNN attention loads NVRTC and its builtins through the dynamic loader.
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib"
RUN test -x "$CUDA_HOME/bin/nvcc" \
    && ln -s lib "$CUDA_HOME/lib64" \
    && ln -s libcudart.so.13 "$CUDA_HOME/lib/libcudart.so"
RUN printf '#include <cuda_runtime.h>\nstatic_assert(CUDART_VERSION / 10 == __CUDACC_VER_MAJOR__ * 100 + __CUDACC_VER_MINOR__);\n' \
    | "$CUDA_HOME/bin/nvcc" -x cu -c -o /tmp/cuda-check.o - \
    && rm /tmp/cuda-check.o
# Bundle Panda assets so Playground can start without runtime downloads.
RUN --mount=type=bind,source=src/rl2/envs/playground_assets.py,target=/tmp/playground_assets.py \
    /app/.venv/bin/python /tmp/playground_assets.py
COPY --from=registry.k8s.io/kubectl:v1.36.5 /bin/kubectl /usr/local/bin/kubectl
RUN kubectl version --client
# Preserve the built environment when a command uses uv run.
ENV UV_NO_SYNC=1 UV_ENV_FILE=/app/nvida.env PATH="/app/.venv/bin:${CUDA_HOME}/bin:$PATH"
# Ray workers share the installed image environment, including optional extras.
ENV RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 RAY_DEDUP_LOGS=0

FROM training-base AS visfly-base
# Leave GPU memory for Habitat scenes and evaluation; overrides uv's nvida.env default.
ENV XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_CLIENT_MEM_FRACTION=0.50
RUN apt-get update && apt-get install -y --no-install-recommends libassimp5 libzstd1 libgmp10 libmpfr6 patch \
    && rm -rf /var/lib/apt/lists/*
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,from=habitat-build,source=/wheels,target=/tmp/visfly-wheels \
    uv pip install --no-deps /tmp/visfly-wheels/*.whl \
    && uv pip install numpy-quaternion==2024.0.13 stable-baselines3==2.9.2
COPY --from=visfly-sources /opt/VisFly /opt/VisFly
COPY infra/k3s/training/visfly/scene_lifecycle.patch /tmp/scene_lifecycle.patch
RUN patch --directory=/opt/VisFly --strip=1 < /tmp/scene_lifecycle.patch \
    && rm /tmp/scene_lifecycle.patch
# Remove unused example imports that load torchvision/Triton after TensorFlow.
RUN sed -i '/^from \.\.utils\.tools\.train_encoder import model as encoder$/d' /opt/VisFly/envs/NavigationEnv.py \
    && sed -i '/^from torchvision import datasets, transforms$/d' /opt/VisFly/utils/dataloader.py
# Articulated scene furniture requires Bullet; drone dynamics remain in VisFly.
RUN sed -i 's/sim_cfg.enable_physics = False/sim_cfg.enable_physics = True/' /opt/VisFly/utils/SceneManager.py
# Scene JSON contains paths relative to the parent of VisFly.
RUN ln -s /opt/VisFly /app/VisFly
ENV PYTHONPATH=/opt VISFLY_SCENE_PATH=/opt/VisFly/datasets/visfly-beta/configs/scenes
# Use TensorBoard's no-TensorFlow mode for PyTorch logging. Loading TensorFlow
# before Triton introduces conflicting native compiler libraries.
RUN touch /app/.venv/lib/python3.12/site-packages/tensorboard/compat/notf.py
COPY --link --from=visfly-assets /opt/VisFly/datasets /opt/VisFly/datasets

# Install only editable project metadata after all heavyweight runtime layers.
# --no-deps preserves the separately installed Habitat/VisFly packages.
FROM training-base AS training
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=README.md,target=/app/README.md \
    --mount=type=bind,source=src,target=/app/src \
    uv pip install --no-deps --editable .
COPY --link data/ ./data/
COPY --link --exclude=data . .

FROM visfly-base AS visfly
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=README.md,target=/app/README.md \
    --mount=type=bind,source=src,target=/app/src \
    uv pip install --no-deps --editable .
COPY --link data/ ./data/
COPY --link --exclude=data . .
RUN PYTHONDONTWRITEBYTECODE=1 python -X faulthandler -c "from rl2 import train_ppo_gym; from rl2.envs.visfly.navigation import RGBNavigationEnv; import habitat_sim; assert hasattr(habitat_sim.Simulator, 'route_motion_is_valid'); assert hasattr(habitat_sim.gfx.Renderer.Flags, 'RGB_DEPTH')"

# Keep the standard image as the default build target.
FROM training AS default
