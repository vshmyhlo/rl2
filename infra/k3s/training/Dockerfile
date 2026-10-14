FROM python:3.12-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.12.21 /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git ffmpeg libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
ENV UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never PYTHONUNBUFFERED=1
ARG UV_EXTRAS=""
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project --extra cuda12 $UV_EXTRAS
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --extra cuda12 $UV_EXTRAS
# Preserve the built environment, including extras, when a command uses uv run.
ENV UV_NO_SYNC=1 PATH="/app/.venv/bin:$PATH"
