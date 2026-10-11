"""Localization benchmark (`nav map loc-bench`): how often, how fast and how right the live loop localizes, replayed
on the map's own walks and on captured live sessions (nav/navigation/captured.py), for each way of localizing.

Walks: every usable 1 fps frame of every walk is a query, matched against the map without that walk (its references
and memory frames left out); a map walked once instead leaves out only the frames within `exclude_s` of the query
(optimistic: what is left is the same walk a few seconds away). The truth is where the walk was: its placed track
(the review's scale applied) and the node whose region that is (MemoryIndex's rule). Live sessions have no truth:
they say how often and how fast the loop commits, and how often it changes its mind.

Each sequence of answers also goes through the orchestrator's belief (belief.NodeBelief, the live temperature),
the thing that has to commit: when it first gets a node to `reach`, and how often that node was the true one.

    method   refs: the map's reference images only (as before the memory); memory: + every frame of every walk;
             verify: + LightGlue on the best memory frames (nav-api's verifier, when the map has a memory)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from nav.navigation.belief import BeliefParams, NodeBelief
from nav.navigation.localize import MEMORY_RADIUS_M, MemoryIndex, RefIndex, VerifyParams, localize
from nav.schemas.map import PlaceMap
from nav.schemas.navigation import Localization, Thresholds

LIVE_TEMPERATURE = 0.18  # the orchestrator's NAV_BELIEF_TEMPERATURE (CLAUDE.md, 2026-10-11)
DECISIVE_LEAD = 0.08  # a lead the belief acts on within a few calls


@dataclass
class Query:
    rec_id: str  # the walk, or the live session
    t_ms: float
    q: np.ndarray  # [F, D] place embeddings of the burst
    image: Path | list[Path]  # the frame(s), for the verifier
    walked_m: float  # since the walk started (dead reckoning / floor odometry)
    x_m: float | None = None  # truth, when known (walks)
    y_m: float | None = None
    node: str | None = None


@dataclass
class Row:
    query: Query
    loc: Localization
    ms: float


@dataclass
class Score:
    method: str
    queries: int = 0
    with_truth: int = 0
    node_right: int = 0
    decisive: int = 0
    decisive_known: int = 0  # ... with a truth
    decisive_right: int = 0
    pos_err: list[float] = field(default_factory=list)
    located_s: list[float | None] = field(default_factory=list)  # per sequence: first commit, seconds in
    commits: int = 0  # belief entering `reach` on a node
    commits_known: int = 0
    commits_right: int = 0
    commit_err: list[float] = field(default_factory=list)  # metres from the true position to the committed node
    ms: list[float] = field(default_factory=list)

    def line(self) -> str:
        def pct(a, b):
            return f"{100 * a / b:3.0f}%" if b else "  - "
        err = np.array(self.pos_err)
        loc = [x for x in self.located_s if x is not None]
        return (f"{self.method:15s} queries {self.queries:4d} | node right {pct(self.node_right, self.with_truth)} | "
                f"decisive (lead>={DECISIVE_LEAD}) {pct(self.decisive, self.queries)}, right "
                f"{pct(self.decisive_right, self.decisive_known)} | position "
                + (f"median {np.median(err):.1f} m, <=3 m {pct(int((err <= 3).sum()), len(err))}" if len(err) else "-")
                + f" | belief located {len(loc)}/{len(self.located_s)}"
                + (f" (median {np.median(loc):.0f} s)" if loc else "")
                + f", commits {self.commits} right {pct(self.commits_right, self.commits_known)}"
                + (f" (node <=3 m away {pct(int((np.array(self.commit_err) <= 3).sum()), len(self.commit_err))})"
                   if self.commit_err else "")
                + f" | {np.median(self.ms):.0f} ms" if self.ms else "")


def region(pmap: PlaceMap, x: float | None, y: float | None, radius_m: float = MEMORY_RADIUS_M) -> str | None:
    if x is None:
        return None
    best = min(((np.hypot(n.position.x_m - x, n.position.y_m - y), n.id) for n in pmap.nodes if n.position),
               default=None)
    return best[1] if best and best[0] <= radius_m else None


def walk_queries(pmap: PlaceMap, recordings_root: Path, cache_root: Path, every_s: float = 1.0) -> list[list[Query]]:
    """Per walk of the map, its usable frames as queries with the truth (placed track, published scale)."""
    from nav.mapping.prepare import load_walk, place_embeddings
    from nav.mapping.references import _usable
    from nav.mapping.select import SelectParams

    p = SelectParams()
    k = pmap.frame.distance_scale if pmap.frame else 1.0
    tracks = {t.rec_id: np.array([pt[:3] for pt in t.points], float) for t in pmap.tracks}
    out = []
    for rec in pmap.recordings:
        rec_dir = recordings_root / rec
        if not (rec_dir / "manifest.json").is_file():
            continue
        walk = load_walk(rec_dir, cache_root, p)
        place = place_embeddings(walk, cache_root, pmap.memory.model if pmap.memory else pmap.embedding.model)
        row_of = {r["file"]: i for i, r in enumerate(walk.rows)}
        tr = tracks.get(rec)
        qs, last = [], -1e18
        for f in _usable(walk.rows, walk.track, p):
            if f["t_ms"] - last < every_s * 1000 - 1:
                continue
            last = f["t_ms"]
            x = y = None
            if tr is not None:
                x = float(np.interp(f["t_ms"], tr[:, 0], tr[:, 1])) * k
                y = float(np.interp(f["t_ms"], tr[:, 0], tr[:, 2])) * k
            qs.append(Query(rec, f["t_ms"], place[row_of[f["file"]]][None], walk.frames_dir / f["file"],
                            float(walk.track.at(f["t_ms"])["walked_m"]) * k, x, y, region(pmap, x, y)))
        out.append(qs)
    return out


def _without_window(index: RefIndex, rec: str, t_ms: float, exclude_s: float) -> RefIndex:
    """The index without the references and memory frames of `rec` within exclude_s of t_ms."""
    keep = [i for i, r in enumerate(index.refs) if not (r.rec_id == rec and abs(r.t_ms - t_ms) <= exclude_s * 1000)]
    mem = index.memory
    if mem:
        mk = [i for i, f in enumerate(mem.frames) if not (f.rec_id == rec and abs(f.t_ms - t_ms) <= exclude_s * 1000)]
        mem = MemoryIndex([mem.frames[i] for i in mk], mem.vectors[mk], [mem.nodes[i] for i in mk])
    return RefIndex(index.pmap, index.vectors[keep], [index.refs[i] for i in keep], mem)


def local_verifier(map_dir: Path, index: RefIndex, image: Path):
    """localize's `verify` for a replayed frame: LightGlue here (nav/mapping/inference.verify_live, the inference
    host's code), memory frames cached by path."""
    from nav.mapping.inference import verify_live

    images = image if isinstance(image, list) else [image]

    def verify(rows: list[int], frames: list[int]) -> dict[int, int]:
        refs = [(str(map_dir / index.memory.frames[i].image), map_dir / index.memory.frames[i].image) for i in rows]
        got = verify_live([images[k].read_bytes() for k in frames if k < len(images)] or [images[-1].read_bytes()],
                          refs, deadline_s=30)
        return {i: max(int(row[c]) for row in got) for c, i in enumerate(rows)}
    return verify


def run_sequence(index_for, queries: list[Query], thresholds: Thresholds, map_dir: Path | None = None,
                 verify: VerifyParams | None = None) -> list[Row]:
    rows = []
    for q in queries:
        index = index_for(q)
        kw = {}
        if verify is not None and index.memory:
            kw |= {"verify": local_verifier(map_dir, index, q.image), "verify_params": verify}
        t = time.perf_counter()
        loc = localize(index, q.q, thresholds, top_k=20, **kw)
        rows.append(Row(q, loc, (time.perf_counter() - t) * 1000))
    return rows


def score(method: str, pmap: PlaceMap, sequences: list[list[Row]], reach: float | None = None) -> Score:
    bp = BeliefParams(temperature=LIVE_TEMPERATURE)
    reach = reach or bp.reach
    s = Score(method)
    positions = {n.id: (n.position.x_m, n.position.y_m) for n in pmap.nodes if n.position}
    for rows in sequences:
        belief = NodeBelief(pmap, bp)
        located, inside, last_walked = None, None, None
        t0 = rows[0].query.t_ms if rows else 0
        for r in rows:
            q, loc = r.query, r.loc
            s.queries += 1
            s.ms.append(r.ms)
            if q.node is not None:
                s.with_truth += 1
                s.node_right += loc.best == q.node
            if (loc.margin or 0) >= DECISIVE_LEAD and loc.best:
                s.decisive += 1
                if q.node is not None:
                    s.decisive_known += 1
                    s.decisive_right += loc.best == q.node
            if loc.position and q.x_m is not None:
                s.pos_err.append(float(np.hypot(loc.position.x_m - q.x_m, loc.position.y_m - q.y_m)))
            walked = 0.0 if last_walked is None else max(q.walked_m - last_walked, 0.0)
            last_walked = q.walked_m
            belief.predict(walked * bp.walk_scale)
            belief.update(loc)
            node, prob = belief.best()
            if prob >= reach and node != inside:
                s.commits += 1
                if q.node is not None:
                    s.commits_known += 1
                    s.commits_right += node == q.node
                pos = positions.get(node)
                if pos is not None and q.x_m is not None:
                    s.commit_err.append(float(np.hypot(pos[0] - q.x_m, pos[1] - q.y_m)))
                if located is None:
                    located = (q.t_ms - t0) / 1000
            inside = node if prob >= reach else None
        s.located_s.append(located)
    return s


def bench_walks(pmap: PlaceMap, map_dir: Path, recordings_root: Path, cache_root: Path, thresholds: Thresholds,
                methods: list[str], exclude_s: float = 10.0, verify_params: VerifyParams | None = None) -> list[Score]:
    from nav.navigation.replay import ref_indexes

    walks = walk_queries(pmap, recordings_root, cache_root)
    holdout = len([w for w in walks if w]) >= 2
    out = []
    for method in methods:
        use_memory = method != "refs"
        indexes = ref_indexes(pmap, map_dir, holdout=holdout, memory=use_memory)
        if holdout:
            index_for = lambda q: indexes[q.rec_id]  # noqa: E731
        else:
            index_for = lambda q: _without_window(indexes, q.rec_id, q.t_ms, exclude_s)  # noqa: E731
        verify = (verify_params or VerifyParams()) if method.startswith("verify") else None
        out.append(score(method, pmap, [run_sequence(index_for, w, thresholds, map_dir, verify)
                                        for w in walks if w]))
    return out


def session_queries(pmap: PlaceMap, session) -> list[Query]:
    """A captured live session (navigation/captured.Session) as queries: each localize call's frames, embedded now
    with the map's place model; walked metres from the phone's steps at the map's live step length. No truth."""
    from nav.mapping.inference import place_files

    model = pmap.memory.model if pmap.memory else pmap.embedding.model
    step = (pmap.frame.live_step_length_m or pmap.frame.step_length_m) if pmap.frame else 0.7
    calls = session.localizations()
    if not calls:
        return []
    paths = [f for c in calls for f in c.frames]
    v = place_files(paths, model)
    out, row, t0, last_steps, walked = [], 0, calls[0].t_utc, None, 0.0
    for c in calls:
        q = v[row:row + len(c.frames)]
        row += len(c.frames)
        steps = c.step_count
        if steps is not None and last_steps is not None and steps >= last_steps:
            walked += (steps - last_steps) * step
        last_steps = steps if steps is not None else last_steps
        out.append(Query(session.session, (c.t_utc - t0).total_seconds() * 1000, q, list(c.frames), walked))
    return out


def bench_sessions(pmap: PlaceMap, map_dir: Path, sessions, thresholds: Thresholds, methods: list[str],
                   verify_params: VerifyParams | None = None) -> list[Score]:
    """Captured live sessions of the map replayed through today's localizer, method by method (no truth: how often
    it is decisive, how soon the belief commits and how often it changes its mind)."""
    from nav.navigation.replay import ref_indexes

    queries = [qs for s in sessions if (qs := session_queries(pmap, s))]
    out = []
    for method in methods:
        index = ref_indexes(pmap, map_dir, holdout=False, memory=method != "refs")
        verify = (verify_params or VerifyParams()) if method.startswith("verify") else None
        out.append(score(method, pmap, [run_sequence(lambda q: index, qs, thresholds, map_dir, verify)
                                        for qs in queries]))
    return out
