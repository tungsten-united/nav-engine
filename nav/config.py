"""Settings from environment variables (or a .env file in the working directory)."""

import os
from pathlib import Path


def _load_dotenv(path: Path = Path(".env")) -> None:
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()

DATA_DIR = Path(os.environ.get("NAV_DATA_DIR", "data")).resolve()
RECORDINGS_DIR = DATA_DIR / "recordings"
HOST = os.environ.get("NAV_HOST", "127.0.0.1")
PORT = int(os.environ.get("NAV_PORT", "8450"))
MAX_UPLOAD_BYTES = int(os.environ.get("NAV_MAX_UPLOAD_MB", "64")) * 1024 * 1024
# Shared copy of data/recordings for the team (`nav rec push` / `nav rec pull`); same layout, one prefix per rec_id.
GCS_URL = os.environ.get("NAV_GCS_URL", "gs://tungsten-united-nav-recordings/recordings").rstrip("/")

# Map pipeline (`nav map build`): one folder per place under MAPS_DIR, per-recording stage caches under CACHE_DIR.
MAPS_DIR = DATA_DIR / "maps"
CACHE_DIR = DATA_DIR / "cache"
# Maps go next to the recordings in the team bucket, under their own prefix.
GCS_MAPS_URL = os.environ.get("NAV_GCS_MAPS_URL", "gs://tungsten-united-nav-recordings/maps").rstrip("/")
EMBED_MODEL = os.environ.get("NAV_EMBED_MODEL", "facebook/dinov2-base")
# Same-place retrieval across walks: torch.hub "owner/repo:commit" (MegaLoc, MIT).
PLACE_MODEL = os.environ.get("NAV_PLACE_MODEL", "gmberton/MegaLoc:5fe0dd697c4a70ba3e23607f6716ab3c606b16db")
TRANSCRIBE_MODEL = os.environ.get("NAV_TRANSCRIBE_MODEL", "openai/whisper-large-v3-turbo")
MAP_LLM_MODEL = os.environ.get("NAV_MAP_LLM_MODEL", "opus")  # Claude Code --model (alias or full id)
MAP_LLM_EFFORT = os.environ.get("NAV_MAP_LLM_EFFORT", "medium")  # low, medium, high, xhigh, max
# Cheap quality check of every built map (nav/mapping/quality.py); skipped without TYPESAFE_API_KEY.
QUALITY_MODEL = os.environ.get("NAV_QUALITY_MODEL", "jev-latest")
MAP_LANGUAGE = os.environ.get("NAV_MAP_LANGUAGE", "en")
MAP_LLM_BUDGET_USD = float(os.environ.get("NAV_MAP_LLM_BUDGET_USD", "5"))
