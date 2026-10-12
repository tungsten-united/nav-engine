"""Published maps for nav-api, read from the maps folder (the team bucket, mounted read-only on Cloud Run) and kept in
memory with their reference embeddings and visual memory (nav/mapping/memory.py). A map is reloaded when its map.json changes; that is checked at most every
`reload_s` per map, since a stat on the bucket mount is a network call."""

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nav.mapping.memory import load_geometry_obs, load_memory
from nav.mapping.store import MapStore
from nav.navigation.localize import RefIndex
from nav.recorder.store import NotFound
from nav.schemas.map import MapSummary, PlaceMap

log = logging.getLogger(__name__)


@dataclass
class _Entry:
    mtime_ns: int
    checked: float
    index: RefIndex


class MapCache:
    def __init__(self, root: Path, reload_s: float = 30, clock=time.monotonic, memory_radius_m: float | None = None):
        from nav import config

        self.store = MapStore(root)
        self.reload_s = reload_s
        self.memory_radius_m = config.LOC_MEMORY_RADIUS_M if memory_radius_m is None else memory_radius_m
        self._clock = clock
        self._lock = threading.Lock()
        self._maps: dict[str, _Entry] = {}

    def index(self, map_id: str) -> RefIndex:
        path = self.store.dir(map_id) / "map.json"
        now = self._clock()
        with self._lock:
            e = self._maps.get(map_id)
            if e and now - e.checked < self.reload_s:
                return e.index
        try:
            mtime = path.stat().st_mtime_ns
        except FileNotFoundError:
            with self._lock:
                self._maps.pop(map_id, None)
            raise NotFound(map_id)
        if e and e.mtime_ns == mtime:
            e.checked = now
            return e.index
        index = self._load(map_id, path)
        with self._lock:
            self._maps[map_id] = _Entry(mtime, now, index)
        return index

    def _load(self, map_id: str, path: Path) -> RefIndex:
        pmap = PlaceMap.model_validate_json(path.read_text())
        vectors = None
        if pmap.embedding:
            try:
                vectors = np.load(self.store.file(map_id, pmap.embedding.file))
            except (NotFound, ValueError, OSError) as e:
                log.warning("map %s: no reference embeddings (%s)", map_id, e)
        memory = geometry = None
        if pmap.memory:
            try:
                memory = load_memory(self.store.dir(map_id), pmap.memory)
            except (ValueError, OSError) as e:
                log.warning("map %s: no visual memory (%s)", map_id, e)
            try:
                geometry = load_geometry_obs(self.store.dir(map_id), pmap.memory)
            except (ValueError, OSError, KeyError) as e:
                log.warning("map %s: no 3D model for live poses (%s)", map_id, e)
        index = RefIndex.build(pmap, vectors, memory=memory, memory_radius_m=self.memory_radius_m, geometry=geometry)
        log.info("map %s: build %s, %d nodes, %d reference images, %d memory frames (%d in a node's region, %d posed%s)",
                 map_id, pmap.build.build_id, len(pmap.nodes), len(index.refs),
                 len(index.memory.frames) if index.memory else 0,
                 sum(n is not None for n in index.memory.nodes) if index.memory else 0,
                 sum(f.posed for f in index.memory.frames) if index.memory else 0,
                 ", 3D model loaded" if geometry is not None else "")
        return index

    def get(self, map_id: str) -> PlaceMap:
        return self.index(map_id).pmap

    def summaries(self) -> list[MapSummary]:
        return self.store.summaries(get=self.get)

    def file(self, map_id: str, rel: str) -> Path:
        """Only what nav-api's callers need from the published build: its reference images, candidate frames, memory
        frames and embeddings. Builds' job files, candidates and older builds stay on map-api."""
        pmap = self.get(map_id)
        build = f"builds/{pmap.build.build_id}/"
        ok = (rel.startswith(build) and rel.count("/") == 3 and rel.split("/")[2] in ("refs", "frames", "memory")
              and rel.endswith(".jpg")) or (pmap.embedding is not None and rel == pmap.embedding.file)
        if not ok:
            raise NotFound(rel)
        return self.store.file(map_id, rel)
