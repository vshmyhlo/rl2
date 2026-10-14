FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git ffmpeg libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.12.21 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never PYTHONUNBUFFERED=1
# Keep the large dependency layer independent of source and documentation edits.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project
# Install editable metadata without copying source into an intermediate layer.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=README.md,target=/app/README.md \
    --mount=type=bind,source=src,target=/app/src \
    uv sync --locked --no-dev
COPY . .
# Preserve the built environment when a command uses uv run.
ENV UV_NO_SYNC=1 UV_ENV_FILE=/app/nvida.env PATH="/app/.venv/bin:$PATH"
