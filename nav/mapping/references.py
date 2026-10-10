"""Reference images for live localization (docs/navigation_stack.md 2.4 and 4): for every node, frames taken while
approaching it, standing at it, and leaving it along each edge, with the camera heading of each, and their place
embeddings (the same model as same-place retrieval, MegaLoc) so the app can match live camera frames against them.

Frames come from the per-recording cache (1 fps, the sharpest of each second), so they span every walk that passed
the node, not just the candidate shown to the LLM. Picked per role: the sharpest usable frames, spread over
recordings and time. Relation viewpoints are always kept, since relations are measured from them.

Adapted from tungsten-united/demo-repository@c5dc26f (navigation, mapping/src/mapper/reference_images.py) by LPA.
"""

import json
import logging
import shutil
from dataclasses import fields
from pathlib import Path

import numpy as np

from nav.mapping.select import SelectParams
from nav.schemas.map import EmbeddingInfo, PlaceMap, ReferenceImage

log = logging.getLogger(__name__)

PER_ROLE = 3  # images per node and role (outgoing: per edge)
APPROACH_MS = 3000
OUTGOING_MS = 3000
AT_MARGIN_MS = 500
SPREAD_MS = 1500  # two picks from one recording are at least this far apart
VISIT_MS = 15_000  # views of a node closer than this are one visit (as build.VISIT_S)


def _visits(times: list[float]) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for t in sorted(times):
        if out and t - out[-1][-1] <= VISIT_MS:
            out[-1].append(t)
        else:
            out.append([t])
    return [(v[0], v[-1]) for v in out]


def _pick(pool: list[dict], k: int) -> list[dict]:
    """Up to k of the sharpest frames, a new recording first, never two within SPREAD_MS in one recording."""
    chosen: list[dict] = []
    ranked = sorted(pool, key=lambda f: -f["sharpness"])
    for prefer_new in (True, False):
        for f in ranked:
            if len(chosen) >= k:
                return chosen
            if f in chosen or (prefer_new and any(c["rec_id"] == f["rec_id"] for c in chosen)):
                continue
            if any(c["rec_id"] == f["rec_id"] and abs(c["t_ms"] - f["t_ms"]) < SPREAD_MS for c in chosen):
                continue
            chosen.append(f)
    return chosen


def _usable(rows: list[dict], track, p: SelectParams) -> list[dict]:
    if not rows:
        return []
    floor = p.min_sharpness_rel * float(np.median([r["sharpness"] for r in rows]))
    out = []
    for r in rows:
        motion = track.at(r["t_ms"])
        pitch = motion["pitch_deg"]
        if (r["sharpness"] >= floor and p.min_brightness <= r["brightness"] <= p.max_brightness
                and (pitch is None or abs(pitch) <= p.max_abs_pitch_deg)):
            out.append({**r, "heading_deg": motion["heading_deg"]})
    return out


def select_references(pmap: PlaceMap, params: dict, build_dir: Path, recordings_root: Path, cache_root: Path,
                      warnings: list[str]) -> list[ReferenceImage]:
    """Copy the chosen frames to build_dir/refs/ and link them to nodes and edges (in place on pmap)."""
    from nav.mapping.prepare import load_walk

    p = SelectParams(**{f.name: params[f.name] for f in fields(SelectParams) if f.name in params})
    build_rel = f"builds/{build_dir.name}"
    edge_of = {}
    for e in pmap.edges:
        edge_of[(e.source, e.target)] = edge_of[(e.target, e.source)] = e.id
    # every node visit per recording, in time order: [(t0, t1, node id)]
    visits: dict[str, list[tuple[float, float, str]]] = {}
    for n in pmap.nodes:
        per_rec: dict[str, list[float]] = {}
        for ev in n.evidence:
            per_rec.setdefault(ev.rec_id, []).append(ev.t_ms)
        for rec, times in per_rec.items():
            visits.setdefault(rec, []).extend((t0, t1, n.id) for t0, t1 in _visits(times))

    pools: dict[tuple[str, str, str | None], list[dict]] = {}  # (node, role, toward) -> frames
    folders: dict[str, Path] = {}
    for rec, vs in visits.items():
        rec_dir = recordings_root / rec
        if not (rec_dir / "manifest.json").is_file():
            warnings.append(f"references: recording {rec} not found, its frames are not used")
            continue
        try:
            walk = load_walk(rec_dir, cache_root, p)
        except Exception as e:  # a broken recording must not lose the map
            warnings.append(f"references: recording {rec}: {type(e).__name__}: {e}")
            continue
        folders[rec] = walk.frames_dir
        frames = [{**f, "rec_id": rec} for f in _usable(walk.rows, walk.track, p)]
        vs.sort()
        for i, (t0, t1, nid) in enumerate(vs):
            nxt = next(((s, node) for s, _, node in vs[i + 1:] if node != nid), None)
            prev_end = max((e for _, e, node in vs[:i] if node != nid), default=-1e12)
            pools.setdefault((nid, "at", None), []).extend(
                f for f in frames if t0 - AT_MARGIN_MS <= f["t_ms"] <= t1 + AT_MARGIN_MS)
            pools.setdefault((nid, "approach", None), []).extend(
                f for f in frames if max(t0 - APPROACH_MS, prev_end) <= f["t_ms"] < t0 - AT_MARGIN_MS)
            if nxt and (nid, nxt[1]) in edge_of:
                pools.setdefault((nid, "outgoing", nxt[1]), []).extend(
                    f for f in frames if t1 + AT_MARGIN_MS < f["t_ms"] <= min(t1 + OUTGOING_MS, nxt[0]))

    refs_dir = build_dir / "refs"
    if refs_dir.exists():
        shutil.rmtree(refs_dir)  # this build's own derived output, rewritten by every finish
    refs: list[ReferenceImage] = []
    nodes = {n.id: n for n in pmap.nodes}
    edges = {e.id: e for e in pmap.edges}
    for n in nodes.values():
        n.references = []
    for e in edges.values():
        e.reference_views = []
    for (nid, role, toward), pool in sorted(pools.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2] or "")):
        for f in _pick(pool, PER_ROLE):
            rid = f"r{len(refs) + 1:03d}"
            refs_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(folders[f["rec_id"]] / f["file"], refs_dir / f"{rid}.jpg")
            eid = edge_of.get((nid, toward)) if toward else None
            refs.append(ReferenceImage(id=rid, node=nid, edge=eid, toward=toward, role=role,
                                       image=f"{build_rel}/refs/{rid}.jpg", rec_id=f["rec_id"], t_ms=f["t_ms"],
                                       heading_deg=f["heading_deg"]))
            nodes[nid].references.append(rid)
            if eid:
                edges[eid].reference_views.append(rid)
    for n in nodes.values():  # relations are measured from these photos: matchable views of the node
        for r in n.relations:
            if not any(x.rec_id == r.rec_id and abs(x.t_ms - r.t_ms) < AT_MARGIN_MS and x.node == n.id for x in refs):
                rid = f"r{len(refs) + 1:03d}"
                refs.append(ReferenceImage(id=rid, node=n.id, role="at", image=r.viewpoint_image, rec_id=r.rec_id,
                                           t_ms=r.t_ms, heading_deg=r.viewpoint_heading_deg))
                n.references.append(rid)
    return refs


def embed_references(pmap: PlaceMap, map_dir: Path, build_dir: Path, model: str, warnings: list[str]) -> None:
    """Place embeddings of the reference images to build_dir/embeddings.npy (reused while the images and the model
    stay the same); sets embedding_row and pmap.embedding. Without the model (inference host down), the map keeps
    its references without embeddings."""
    from nav.mapping.inference import place_files

    pmap.embedding = None
    if not pmap.references:
        return
    key = {"model": model, "images": [[r.rec_id, r.t_ms, r.image.rsplit("/", 1)[-1]] for r in pmap.references]}
    npy, meta = build_dir / "embeddings.npy", build_dir / "embeddings.json"
    try:
        if not (npy.is_file() and meta.is_file() and json.loads(meta.read_text()) == key):
            vectors = place_files([map_dir / r.image for r in pmap.references], model)
            np.save(npy, np.asarray(vectors, np.float32))
            meta.write_text(json.dumps(key))
        vectors = np.load(npy)
    except Exception as e:
        log.warning("reference embeddings: %s", e)
        warnings.append(f"reference embeddings not computed ({type(e).__name__}: {e}); rerun `nav map refinish`")
        return
    for i, r in enumerate(pmap.references):
        r.embedding_row = i
    pmap.embedding = EmbeddingInfo(model=model, dim=int(vectors.shape[1]), file=f"builds/{build_dir.name}/embeddings.npy")


def add_references(pmap: PlaceMap, params: dict, build_dir: Path, recordings_root: Path, cache_root: Path,
                   model: str) -> PlaceMap:
    warnings: list[str] = []
    pmap.references = select_references(pmap, params, build_dir, recordings_root, cache_root, warnings)
    embed_references(pmap, build_dir.parent.parent, build_dir, model, warnings)
    pmap.warnings += warnings
    log.info("references: %d images for %d nodes%s", len(pmap.references), len(pmap.nodes),
             f", embedded with {pmap.embedding.model}" if pmap.embedding else "")
    return pmap
