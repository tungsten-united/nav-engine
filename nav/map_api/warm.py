"""Live poses without a cold start: map-api keeps the inference host's feature cache holding every posed memory frame
of the published maps, so nav-api's live poses send only keys.

A live pose matches the camera frames against the map's best memory frames, whose features the host keeps by key
(memory.pose_key). A frame it lacks travels from nav-api first (409, then its image and 3D observations): 1.2-1.9 s a
call instead of ~0.3 s (Itnig, 2026-10-13), each time someone reaches a spot the host hasn't seen since it started.
nav-api has CPU only during requests, so map-api (one instance, always on) does it: after each build and every
NAV_POSE_WARM_S (the host forgets everything when it restarts), it asks the host which frames it lacks and sends only
those, in small batch calls that wait behind live frames. Maps go newest first, up to the cache's capacity: beyond it
they would only push each other out.
"""

import logging
import threading
from pathlib import Path

from nav.infer.errors import InferenceError
from nav.mapping.inference import remote
from nav.mapping.memory import pose_refs
from nav.mapping.store import MapStore
from nav.schemas.map import PlaceMap

log = logging.getLogger(__name__)


class PoseWarmer:
    def __init__(self, maps: MapStore, interval_s: float, client_fn=None):
        self.maps, self.interval_s = maps, interval_s
        self.client_fn = client_fn or remote
        self._kick = threading.Event()
        self._refs: dict[str, tuple[float, list]] = {}  # map id -> (map.json mtime, pose refs)
        self._skipped: set[tuple[str, int]] = set()  # (map id, frames) already reported as over capacity

    def start(self) -> "PoseWarmer":
        threading.Thread(target=self._loop, name="pose-warmer", daemon=True).start()
        return self

    def kick(self) -> None:
        """Check now (a build was published)."""
        self._kick.set()

    def _loop(self) -> None:
        while True:
            try:
                self.warm_once()
            except InferenceError as e:
                log.info("pose warm-up: inference host unavailable (%s)", e)
            except Exception:
                log.exception("pose warm-up failed")
            self._kick.wait(self.interval_s)
            self._kick.clear()

    def refs(self, map_id: str, published: Path) -> list:
        mtime = published.stat().st_mtime
        if (got := self._refs.get(map_id)) and got[0] == mtime:
            return got[1]
        pmap = PlaceMap.model_validate_json(published.read_text())
        refs = pose_refs(map_id, self.maps.dir(map_id), pmap)
        self._refs[map_id] = (mtime, refs)
        return refs

    def warm_once(self) -> dict[str, int]:
        """One round: map id -> frames sent to the host."""
        client = self.client_fn()
        if client is None or not self.maps.root.is_dir():
            return {}
        found = []
        for d in self.maps.root.iterdir():
            if (published := d / "map.json").is_file():
                found.append((published.stat().st_mtime, d.name, published))
        sent: dict[str, int] = {}
        used, capacity = 0, None
        for _, map_id, published in sorted(found, reverse=True):
            refs = self.refs(map_id, published)
            if not refs:
                continue
            if capacity is not None and used + len(refs) > capacity:
                if (map_id, len(refs)) not in self._skipped:
                    self._skipped.add((map_id, len(refs)))
                    log.warning("pose warm-up: %s (%d frames) left out: the host caches %d and newer maps hold %d",
                                map_id, len(refs), capacity, used)
                continue
            got = client.warm_poses([(k, (lambda p=path: p.read_bytes()), obs) for k, path, obs in refs])
            used, capacity = used + len(refs), got["capacity"]
            sent[map_id] = got["sent"]
            if got["sent"]:
                log.info("pose warm-up: %s: %d of %d frames sent (host cache %d of %d)", map_id, got["sent"],
                         len(refs), got["cached"], got["capacity"])
        return sent
