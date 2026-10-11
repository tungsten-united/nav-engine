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
# Static apps (phone recorder, /dash/, /viz/): the repo's frontend/ folder; "" serves no frontend.
_web = os.environ.get("NAV_WEB_DIR", str(Path(__file__).resolve().parent.parent / "frontend"))
WEB_DIR = Path(_web).resolve() if _web else None
# Public deployment (Cloud Run). Unset, the API is open, as behind Cloudflare Access on helium.
API_TOKEN = os.environ.get("NAV_API_TOKEN") or None  # required on /api/* (Bearer header or ?token=)
CORS_ORIGINS = [o.strip() for o in os.environ.get("NAV_CORS_ORIGINS", "").split(",") if o.strip()]
CORS_ORIGIN_REGEX = os.environ.get("NAV_CORS_ORIGIN_REGEX") or None  # e.g. http://localhost(:\d+)?
ALLOW_DELETE = os.environ.get("NAV_ALLOW_DELETE", "1") != "0"  # 0 where the data is the team's shared copy
# Shared copy of data/recordings for the team (`nav rec push` / `nav rec pull`); same layout, one prefix per rec_id.
GCS_URL = os.environ.get("NAV_GCS_URL", "gs://tungsten-united-nav-recordings/recordings").rstrip("/")

# Map pipeline (`nav map build`): one folder per place under MAPS_DIR, per-recording stage caches under CACHE_DIR.
MAPS_DIR = DATA_DIR / "maps"
CACHE_DIR = Path(os.environ.get("NAV_CACHE_DIR", DATA_DIR / "cache")).resolve()
# Maps go next to the recordings in the team bucket, under their own prefix.
GCS_MAPS_URL = os.environ.get("NAV_GCS_MAPS_URL", "gs://tungsten-united-nav-recordings/maps").rstrip("/")
EMBED_MODEL = os.environ.get("NAV_EMBED_MODEL", "facebook/dinov2-base")
# Same-place retrieval across walks: torch.hub "owner/repo:commit" (MegaLoc, MIT).
PLACE_MODEL = os.environ.get("NAV_PLACE_MODEL", "gmberton/MegaLoc:5fe0dd697c4a70ba3e23607f6716ab3c606b16db")
TRANSCRIBE_MODEL = os.environ.get("NAV_TRANSCRIBE_MODEL", "openai/whisper-large-v3-turbo")
# Language of the voice notes (a Whisper code, e.g. "en"); unset: detected once per recording from all its clips.
TRANSCRIBE_LANGUAGE = os.environ.get("NAV_TRANSCRIBE_LANGUAGE") or None
MAP_LLM_MODEL = os.environ.get("NAV_MAP_LLM_MODEL", "opus")  # Claude Code --model (alias or full id)
MAP_LLM_EFFORT = os.environ.get("NAV_MAP_LLM_EFFORT", "medium")  # low, medium, high, xhigh, max
# Cheap quality check of every built map (nav/mapping/quality.py); skipped without TYPESAFE_API_KEY.
QUALITY_MODEL = os.environ.get("NAV_QUALITY_MODEL", "jev-latest")
MAP_LANGUAGE = os.environ.get("NAV_MAP_LANGUAGE", "en")
MAP_LLM_BUDGET_USD = float(os.environ.get("NAV_MAP_LLM_BUDGET_USD", "5"))
# map-api rebuilds a place's map when one of its walks finalizes (same map, improved: ids and review carry over).
AUTO_BUILD = os.environ.get("NAV_AUTO_BUILD", "1") != "0"
# The published map keeps only what the walks back up (nav/mapping/clean.py); 0 publishes the reviewed build as is.
MAP_CLEAN = os.environ.get("NAV_MAP_CLEAN", "1") != "0"

# Inference host (nav infer serve): the GPU machine serving the local models and the Claude Code map job over HTTP.
# Set NAV_INFER_URL on a worker without GPU to use it instead of local models; it goes through Cloudflare Access with
# the service token in CF_ACCESS_CLIENT_ID / CF_ACCESS_CLIENT_SECRET.
INFER_URL = (os.environ.get("NAV_INFER_URL") or "").rstrip("/") or None
INFER_HOST = os.environ.get("NAV_INFER_HOST", "127.0.0.1")
INFER_PORT = int(os.environ.get("NAV_INFER_PORT", "8451"))
INFER_JOBS_DIR = Path(os.environ.get("NAV_INFER_JOBS_DIR", DATA_DIR / "infer-jobs")).resolve()
INFER_QUEUE = int(os.environ.get("NAV_INFER_QUEUE", "4"))  # model requests waiting for the GPU before 503
INFER_JOB_QUEUE = int(os.environ.get("NAV_INFER_JOB_QUEUE", "2"))  # map jobs waiting behind the running one
INFER_FEATURE_CACHE = int(os.environ.get("NAV_INFER_FEATURE_CACHE", "3000"))  # memory frames' local features kept

# Which service `nav serve` runs: map-api (recordings, map builds, review, the debugging frontend) or nav-api (the
# live side for Orient's orchestrator: published maps, routes, localization; read-only on the data).
APP = os.environ.get("NAV_APP", "map-api")
NAV_API_PORT = int(os.environ.get("NAV_NAV_API_PORT", "8452"))  # local default of `nav serve --app nav-api`
# nav-api: where the debugging frontend lives (old bookmarks of /, /dash/ ... on nav-api's URL are sent there).
FRONTEND_URL = (os.environ.get("NAV_FRONTEND_URL") or "").rstrip("/") or None
LIVE_INFER_DEADLINE_S = float(os.environ.get("NAV_LIVE_INFER_DEADLINE_S", "8"))  # a live frame fails fast
MAP_RELOAD_S = float(os.environ.get("NAV_MAP_RELOAD_S", "30"))  # how often nav-api checks for a newly published map
COMMIT = os.environ.get("NAV_COMMIT") or None  # the build, in nav-api's log lines (deploy/cloudrun.sh sets it)
# nav-api records every call it answers, frames included, under here (nav/nav_api/capture.py); unset: nothing is kept.
# On Cloud Run a read-write mount of the bucket's live/ prefix; `nav live pull` copies it to LIVE_DIR.
CAPTURE_DIR = (lambda v: Path(v).resolve() if v else None)(os.environ.get("NAV_CAPTURE_DIR"))
LIVE_DIR = DATA_DIR / "live"
GCS_LIVE_URL = os.environ.get("NAV_GCS_LIVE_URL", "gs://tungsten-united-nav-recordings/live").rstrip("/")
# Localization status (nav/navigation/localize.py): placeholders until tuned on venue images.
LOC_CONFIRM_SCORE = float(os.environ.get("NAV_LOC_CONFIRM_SCORE", "0.45"))
LOC_CONFIRM_MARGIN = float(os.environ.get("NAV_LOC_CONFIRM_MARGIN", "0.05"))
LOC_LOST_SCORE = float(os.environ.get("NAV_LOC_LOST_SCORE", "0.35"))
# With the phone's motion (nav/navigation/localize.py): leave out frames pointed at the floor/ceiling, and require the
# walked distance from the last confirmed node (tuned with `nav map orchestrator-replay`: on Itnig's 1-7 m edges a
# 2 m slack let every next node through after a step or two; 1 m stopped 3 of 9 early confirmations, lost no node).
LOC_MAX_ABS_PITCH_DEG = float(os.environ.get("NAV_LOC_MAX_ABS_PITCH_DEG", "55"))
LOC_GATE_SLACK_M = float(os.environ.get("NAV_LOC_GATE_SLACK_M", "1.0"))
LOC_GATE_SLACK_FRAC = float(os.environ.get("NAV_LOC_GATE_SLACK_FRAC", "0.35"))
# Visual memory (nav/mapping/memory.py): a frame belongs to the nearest node within this many metres of where it was
# taken (Nati House: nodes 1-4 m apart; Itnig 3-7 m).
LOC_MEMORY_RADIUS_M = float(os.environ.get("NAV_LOC_MEMORY_RADIUS_M", "5.0"))
# Live verification (localize.py VerifyParams): LightGlue on the inference host between the camera frames and the
# NAV_LOC_VERIFY_K best memory frames (0: off); inliers from LOW to HIGH add up to GAIN to that frame's score.
# Itnig, cross-walk: >= 300 inliers put the camera within 3 m of the matched frame 80% of the time. Defaults from
# `nav map loc-bench` on Itnig and Nati House with degraded frames: every walk located (memory alone: 3 of 4 Itnig
# walks, Nati House after 31 s), 95-100% of the belief's commits within 3 m of the truth; a stronger gain commits
# faster and strays more.
LOC_VERIFY_K = int(os.environ.get("NAV_LOC_VERIFY_K", "6"))
LOC_VERIFY_LOW = int(os.environ.get("NAV_LOC_VERIFY_LOW", "150"))
LOC_VERIFY_HIGH = int(os.environ.get("NAV_LOC_VERIFY_HIGH", "500"))
LOC_VERIFY_GAIN = float(os.environ.get("NAV_LOC_VERIFY_GAIN", "0.3"))
LIVE_VERIFY_DEADLINE_S = float(os.environ.get("NAV_LIVE_VERIFY_DEADLINE_S", "3"))  # then the answer goes unverified


def verify_params():
    """nav-api's live verification settings from the environment (NAV_LOC_VERIFY_*)."""
    from nav.navigation.localize import VerifyParams

    return VerifyParams(k=LOC_VERIFY_K, low=LOC_VERIFY_LOW, high=LOC_VERIFY_HIGH, gain=LOC_VERIFY_GAIN)


def loc_thresholds():
    """nav-api's localization thresholds from the environment (NAV_LOC_*)."""
    from nav.schemas.navigation import Thresholds

    return Thresholds(confirm_score=LOC_CONFIRM_SCORE, confirm_margin=LOC_CONFIRM_MARGIN, lost_score=LOC_LOST_SCORE,
                      max_abs_pitch_deg=LOC_MAX_ABS_PITCH_DEG, gate_slack_m=LOC_GATE_SLACK_M,
                      gate_slack_frac=LOC_GATE_SLACK_FRAC)
