"""Live capture: every call nav-api answers, frames included, kept for benchmarks (NAV_CAPTURE_DIR; unset: off).

    <root>/<map_id>/<YYYY-MM-DD>/<session>/<HHMMSSmmm>-<id>-localize.json     request, motion, answer (or error)
    <root>/<map_id>/<YYYY-MM-DD>/<session>/<HHMMSSmmm>-<id>-localize-f0.jpg   the frames as received
    <root>/<map_id>/<YYYY-MM-DD>/<session>/<HHMMSSmmm>-<id>-route.json

`session` is the orchestrator's X-Session-Id (one walk of one user), else X-Client-Id, else "anonymous"; dates and
times are UTC; `id` is X-Request-Id, else random. A session folder replays as the walk it was
(nav/navigation/captured.py), and Cloud Logging has the orchestrator's side of it under the same sessionId
(`nav live logs`).

On Cloud Run the root is a read-write mount of the bucket's `live/` prefix and nothing else (deploy/cloudrun.sh):
nav-api still only reads the maps. Writes happen on one background thread, so being recorded never makes a call
slower; if the bucket stalls, a full queue drops captures with a warning instead of holding memory. Cloud Run only
gives CPU while requests run, so the last captures of a quiet instance are written with its next request (or on
shutdown, which flushes).
"""

import json
import logging
import queue
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

SAFE = re.compile(r"[^A-Za-z0-9._-]+")
MAX_QUEUE = 256  # calls waiting to be written (each a few frames of ~50-200 KB)


def safe(part: str | None, default: str) -> str:
    s = SAFE.sub("-", part or "").strip("-.")[:64]
    return s or default


class Capture:
    def __init__(self, root: Path, clock=lambda: datetime.now(timezone.utc)):
        self.root = Path(root)
        self._clock = clock
        self._q: queue.Queue = queue.Queue(MAX_QUEUE)
        self.dropped = 0
        threading.Thread(target=self._worker, name="live-capture", daemon=True).start()

    def localize(self, map_id: str, who: dict, frames: list[bytes], request: dict, *, response: dict | None = None,
                 error: dict | None = None, build_id: str | None = None) -> None:
        self._put("localize", map_id, who, frames, request, response, error, build_id)

    def route(self, map_id: str, who: dict, request: dict, *, response: dict | None = None,
              error: dict | None = None, build_id: str | None = None) -> None:
        self._put("route", map_id, who, [], request, response, error, build_id)

    def _put(self, kind, map_id, who, frames, request, response, error, build_id) -> None:
        from nav import config

        now = self._clock()
        folder = (self.root / safe(map_id, "unknown") / f"{now:%Y-%m-%d}"
                  / safe(who.get("sessionId") or who.get("clientId"), "anonymous"))
        stem = f"{now:%H%M%S}{now.microsecond // 1000:03d}-{safe(who.get('requestId'), uuid.uuid4().hex[:8])}-{kind}"
        names = [f"{stem}-f{i}.jpg" for i in range(len(frames))]
        record = {"kind": kind, "t_utc": now.isoformat(), "map_id": map_id, "build_id": build_id, "who": who,
                  "request": request, "frames": names, "response": response, "error": error,
                  "commit": config.COMMIT}
        try:
            self._q.put_nowait((folder, stem, frames, names, record))
        except queue.Full:
            self.dropped += 1
            log.warning("live capture: queue full, dropped %s %s (%d dropped so far)", kind, map_id, self.dropped)

    def _worker(self) -> None:
        while True:
            folder, stem, frames, names, record = self._q.get()
            try:
                folder.mkdir(parents=True, exist_ok=True)
                for name, data in zip(names, frames):
                    (folder / name).write_bytes(data)
                # the record last: a session folder only lists calls whose frames are there
                (folder / f"{stem}.json").write_text(json.dumps(record, ensure_ascii=False))
            except OSError as e:
                log.warning("live capture: %s/%s not written: %s", folder, stem, e)
            finally:
                self._q.task_done()

    def flush(self, timeout_s: float = 10) -> bool:
        """Wait until everything queued is written (tests, shutdown). False if it took longer than timeout_s."""
        done = threading.Event()

        def wait():
            self._q.join()
            done.set()

        threading.Thread(target=wait, daemon=True).start()
        return done.wait(timeout_s)
