"""Captured live sessions (nav/nav_api/capture.py) read back: each session folder is one walk of one user, its calls in
time order, with the frames as nav-api received them, so a benchmark can replay them through today's localizer.

    for s in sessions(config.LIVE_DIR, "nati-house"):
        for call in s.localizations():
            call.images(), call.motions(), call.request["previous"], call.response["best"] ...
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from nav.schemas.navigation import LiveMotion


@dataclass
class CapturedCall:
    path: Path  # the call's .json
    kind: str  # localize | route
    t_utc: datetime
    map_id: str
    build_id: str | None
    who: dict
    request: dict
    response: dict | None
    error: dict | None
    frames: list[Path] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> "CapturedCall":
        r = json.loads(path.read_text())
        return cls(path=path, kind=r["kind"], t_utc=datetime.fromisoformat(r["t_utc"]), map_id=r["map_id"],
                   build_id=r.get("build_id"), who=r.get("who") or {}, request=r.get("request") or {},
                   response=r.get("response"), error=r.get("error"), frames=[path.parent / n for n in r["frames"]])

    def images(self) -> list[bytes]:
        return [p.read_bytes() for p in self.frames]

    def motions(self) -> list[LiveMotion | None]:
        out = []
        for m in self.request.get("motion") or []:
            try:
                out.append(LiveMotion.model_validate(m) if isinstance(m, dict) else None)
            except ValueError:
                out.append(None)
        return out

    @property
    def step_count(self) -> int | None:
        """The phone's step count with the newest frame that has one."""
        return next((m.step_count for m in reversed(self.motions()) if m and m.step_count is not None), None)


@dataclass
class Session:
    map_id: str
    date: str
    session: str
    dir: Path
    calls: list[CapturedCall]

    def localizations(self) -> list[CapturedCall]:
        return [c for c in self.calls if c.kind == "localize" and c.frames]

    def summary(self) -> dict:
        locs = self.localizations()
        statuses: dict[str, int] = {}
        for c in locs:
            s = (c.response or {}).get("status") or f"error {(c.error or {}).get('status')}"
            statuses[s] = statuses.get(s, 0) + 1
        span = (self.calls[-1].t_utc - self.calls[0].t_utc).total_seconds() if self.calls else 0.0
        builds = sorted({c.build_id for c in self.calls if c.build_id})
        return {"map_id": self.map_id, "date": self.date, "session": self.session, "calls": len(self.calls),
                "localize": len(locs), "routes": sum(c.kind == "route" for c in self.calls),
                "frames": sum(len(c.frames) for c in locs), "minutes": round(span / 60, 1), "statuses": statuses,
                "builds": builds}


def sessions(root: Path, map_id: str | None = None, since: str | None = None) -> list[Session]:
    """Every captured session under root (root/<map_id>/<date>/<session>/), oldest first; `since`: YYYY-MM-DD."""
    out = []
    for map_dir in sorted(p for p in Path(root).glob("*") if p.is_dir()):
        if map_id and map_dir.name != map_id:
            continue
        for day in sorted(p for p in map_dir.glob("*") if p.is_dir()):
            if since and day.name < since:
                continue
            for folder in sorted(p for p in day.glob("*") if p.is_dir()):
                calls = sorted((CapturedCall.load(f) for f in folder.glob("*.json")), key=lambda c: c.t_utc)
                if calls:
                    out.append(Session(map_dir.name, day.name, folder.name, folder, calls))
    return sorted(out, key=lambda s: s.calls[0].t_utc)
