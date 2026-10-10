# One image for both Cloud Run services (deploy/cloudrun.sh), picked by NAV_APP: map-api (recordings, map builds,
# review, the debugging frontend) or nav-api (the live API: maps, routes, localization). `docker build .` gives the
# same image locally. No torch: the models run on the inference host (NAV_INFER_URL). Data is the team bucket,
# mounted at /data by Cloud Run (read-only for nav-api).
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.11.16 /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never UV_NO_CACHE=1
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-default-groups --group map --no-install-project
COPY nav ./nav
COPY frontend ./frontend
RUN uv sync --frozen --no-dev --no-default-groups --group map

ENV PATH=/app/.venv/bin:$PATH NAV_DATA_DIR=/data PYTHONUNBUFFERED=1
CMD exec nav serve --app "${NAV_APP:-map-api}" --host 0.0.0.0 --port "${PORT:-8080}"
