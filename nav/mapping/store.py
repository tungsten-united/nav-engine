"""Maps on the local filesystem (layout in nav/schemas/map.py): one folder per place, immutable builds inside."""

from pathlib import Path

from nav.recorder.store import NotFound
from nav.schemas.map import MapSummary, PlaceMap, slugify

MAP_ID_CHARS = set("abcdefghijklmnopqrstuvwxyz0123456789-")


class MapStore:
    def __init__(self, root: Path):
        self.root = root

    def dir(self, map_id: str) -> Path:
        if not map_id or not set(map_id) <= MAP_ID_CHARS:
            raise NotFound(map_id)
        return self.root / map_id

    def dir_for_place(self, place: str) -> Path:
        return self.root / slugify(place)

    def get(self, map_id: str) -> PlaceMap:
        path = self.dir(map_id) / "map.json"
        if not path.is_file():
            raise NotFound(map_id)
        return PlaceMap.model_validate_json(path.read_text())

    def summaries(self, get=None) -> list[MapSummary]:
        """`get`: how to read a map (default: parse its map.json; nav-api passes its cache)."""
        out = []
        for d in sorted(self.root.glob("*/map.json")) if self.root.is_dir() else []:
            m = (get or self.get)(d.parent.name)
            out.append(MapSummary(map_id=m.map_id, place=m.place, built_at_utc=m.build.built_at_utc,
                                  recordings=len(m.recordings), nodes=len(m.nodes), edges=len(m.edges)))
        return sorted(out, key=lambda s: s.built_at_utc, reverse=True)

    def builds(self, map_id: str) -> list[str]:
        d = self.dir(map_id) / "builds"
        return sorted((b.name for b in d.iterdir() if (b / "map.json").is_file()), reverse=True) if d.is_dir() else []

    def file(self, map_id: str, rel: str) -> Path:
        d = self.dir(map_id).resolve()
        p = (d / rel).resolve()
        if not p.is_relative_to(d) or not p.is_file():
            raise NotFound(rel)
        return p
