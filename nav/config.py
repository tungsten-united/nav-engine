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
