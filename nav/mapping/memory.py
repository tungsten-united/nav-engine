"""The visual memory of a place: where every camera frame of every walk was taken, for live localization.

It splits "where" from "what". The semantic map (nodes, edges, instructions: the LLM's) says what is at a place; the
memory says where a picture was taken. Localization against it answers "which frame of the walks does this look
like, so where am I", which a picture can answer, instead of "which node", which it often can't: a node is an
area, and in an open hall the same view belongs to several of them (Itnig: one walk's "Stage" frames are another's
"Stone wall tables", 78 of 84 confident cross-walk matches that "disagreed" with the node were neighbours).

With a 3D model (geometry.py, every build since metric-1), the memory is the model's frames (2 fps, 1024 px): each
posed frame with the camera pose the model gives it, and the usable unposed ones with their dead-reckoned position
moved into the map frame. Live localization retrieves memory frames by their MegaLoc embedding, then solves the
camera pose against the 3D points the posed ones see (`row`: the frame's index in geometry.npz).

Builds without a model keep the older memory: every usable frame of the per-recording cache (1 fps, the sharpest of
each second; the reference images' quality filter), placed by the walk's track as the build placed it (aligned
through nodes, bent by the loop closure).

Each frame has its MegaLoc place embedding (memory.npy, row i = frame i), its position and camera heading in the
build's map frame, and its image (memory/mNNNN.jpg): what live matching uses and what the debugging pages show.

Frames don't name a node. nav-api gives each the node nearest to its position (navigation/localize.py MemoryIndex),
so a review that moves, merges, hides or rescales nodes moves the regions with it, without rebuilding the memory.
"""

import json
import logging
import shutil
from collections.abc import Callable
from dataclasses import fields
from pathlib import Path

import numpy as np

from nav.mapping.select import SelectParams
from nav.schemas.map import MemoryFrame, MemoryInfo, PlaceMap

log = logging.getLogger(__name__)


def _track_xy(points: list) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if not points:
        return None
    a = np.array([p[:3] for p in points], float)
    return a[:, 0], a[:, 1], a[:, 2]


def build_memory(pmap: PlaceMap, params: dict, build_dir: Path, recordings_root: Path, cache_root: Path,
                 place_model: str, warnings: list[str]) -> MemoryInfo | None:
    """Write build_dir/memory/ (images), memory.json and memory.npy; None when no walk has usable frames."""
    from nav.mapping.prepare import load_walk, place_embeddings
    from nav.mapping.references import _usable

    p = SelectParams(**{f.name: params[f.name] for f in fields(SelectParams) if f.name in params})
    build_rel = f"builds/{build_dir.name}"
    tracks = {t.rec_id: _track_xy(t.points) for t in pmap.tracks}
    out_dir = build_dir / "memory"
    if out_dir.exists():
        shutil.rmtree(out_dir)  # this build's own derived output
    frames: list[MemoryFrame] = []
    vectors: list[np.ndarray] = []
    for rec in pmap.recordings:
        rec_dir = recordings_root / rec
        if not (rec_dir / "manifest.json").is_file():
            warnings.append(f"memory: recording {rec} not found, its frames are not used")
            continue
        try:
            walk = load_walk(rec_dir, cache_root, p)
            place = place_embeddings(walk, cache_root, place_model)
        except Exception as e:  # a broken recording must not lose the map
            warnings.append(f"memory: recording {rec}: {type(e).__name__}: {e}")
            continue
        row_of = {r["file"]: i for i, r in enumerate(walk.rows)}
        tr = tracks.get(rec)
        if tr is None:
            warnings.append(f"memory: walk {rec} is not placed in the map frame: its frames have no position")
        for f in _usable(walk.rows, walk.track, p):
            x = y = None
            if tr is not None:
                x = round(float(np.interp(f["t_ms"], tr[0], tr[1])), 2)
                y = round(float(np.interp(f["t_ms"], tr[0], tr[2])), 2)
            fid = f"m{len(frames) + 1:04d}"
            out_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(walk.frames_dir / f["file"], out_dir / f"{fid}.jpg")
            heading = f.get("heading_deg")
            frames.append(MemoryFrame(id=fid, rec_id=rec, t_ms=f["t_ms"], image=f"{build_rel}/memory/{fid}.jpg",
                                      x_m=x, y_m=y, heading_deg=None if heading is None else round(heading, 1)))
            vectors.append(place[row_of[f["file"]]])
    if not frames:
        return None
    v = np.asarray(vectors, np.float32)
    np.save(build_dir / "memory.npy", v)
    (build_dir / "memory.json").write_text(json.dumps({"frames": [f.model_dump() for f in frames]}))
    log.info("memory: %d frames of %d walks (%d placed)", len(frames), len({f.rec_id for f in frames}),
             sum(f.x_m is not None for f in frames))
    return MemoryInfo(model=place_model, dim=int(v.shape[1]), frames=len(frames), file=f"{build_rel}/memory.json",
                      embeddings=f"{build_rel}/memory.npy")


def build_memory_from_geometry(geo, params: dict, build_dir: Path, recordings_root: Path, cache_root: Path,
                               place_model: str, warnings: list[str]) -> MemoryInfo | None:
    """The model's frames as the memory (see the module doc): posed frames, and unposed ones that pass the quality
    filter (sharp enough for their walk, not too dark or bright, not at the floor or ceiling)."""
    from nav.mapping.geometry import sample_walk
    from nav.mapping.prepare import load_walk

    p = SelectParams(**{f.name: params[f.name] for f in fields(SelectParams) if f.name in params})
    build_rel = f"builds/{build_dir.name}"
    out_dir = build_dir / "memory"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    by_rec: dict[str, list[int]] = {}
    for i, f in enumerate(geo.frames):
        by_rec.setdefault(f.rec_id, []).append(i)
    frames: list[MemoryFrame] = []
    rows: list[int] = []
    for rec, idx in by_rec.items():
        rec_dir = recordings_root / rec
        try:
            fdir = sample_walk(rec_dir, cache_root, geo.info.get("fps", 2.0))[0]
            track = load_walk(rec_dir, cache_root, p).track
        except Exception as e:  # a broken recording must not lose the map
            warnings.append(f"memory: recording {rec}: {type(e).__name__}: {e}")
            continue
        floor = p.min_sharpness_rel * float(np.median([geo.frames[i].sharpness for i in idx]))
        for i in sorted(idx, key=lambda i: geo.frames[i].t_ms):
            f = geo.frames[i]
            pitch = track.at(f.t_ms)["pitch_deg"]
            usable = (f.sharpness >= floor and p.min_brightness <= f.brightness <= p.max_brightness
                      and (pitch is None or abs(pitch) <= p.max_abs_pitch_deg))
            if not (geo.posed[i] or usable) or np.isnan(geo.xyz[i, 0]):
                continue
            fid = f"m{len(frames) + 1:04d}"
            shutil.copyfile(fdir / f.name.split("/", 1)[1], out_dir / f"{fid}.jpg")
            h = geo.heading_deg[i]
            frames.append(MemoryFrame(id=fid, rec_id=rec, t_ms=f.t_ms, image=f"{build_rel}/memory/{fid}.jpg",
                                      x_m=round(float(geo.xyz[i, 0]), 2), y_m=round(float(geo.xyz[i, 1]), 2),
                                      heading_deg=None if np.isnan(h) else round(float(h), 1),
                                      posed=bool(geo.posed[i]), row=i))
            rows.append(i)
    if not frames:
        return None
    v = geo.place[rows].astype(np.float32)
    np.save(build_dir / "memory.npy", v)
    (build_dir / "memory.json").write_text(json.dumps({"frames": [f.model_dump() for f in frames]}))
    log.info("memory: %d frames of %d walks (%d posed by the 3D model)", len(frames), len(by_rec),
             sum(f.posed for f in frames))
    return MemoryInfo(model=place_model, dim=int(v.shape[1]), frames=len(frames), file=f"{build_rel}/memory.json",
                      embeddings=f"{build_rel}/memory.npy", geometry=f"{build_rel}/geometry.npz",
                      posed=sum(f.posed for f in frames))


def add_memory(pmap: PlaceMap, params: dict, build_dir: Path, recordings_root: Path, cache_root: Path,
               place_model: str) -> PlaceMap:
    from nav.mapping import geometry

    warnings: list[str] = []
    geo = geometry.load(build_dir)
    if geo is not None:
        pmap.memory = build_memory_from_geometry(geo, params, build_dir, recordings_root, cache_root, place_model,
                                                 warnings)
    else:
        pmap.memory = build_memory(pmap, params, build_dir, recordings_root, cache_root, place_model, warnings)
    pmap.warnings += warnings
    return pmap


def load_memory(map_dir: Path, info: MemoryInfo) -> tuple[list[MemoryFrame], np.ndarray]:
    """The frames and their embeddings, as written by build_memory (paths relative to the place's folder)."""
    data = json.loads((map_dir / info.file).read_text())
    frames = [MemoryFrame.model_validate(f) for f in data["frames"]]
    vectors = np.load(map_dir / info.embeddings).astype(np.float32)
    if len(vectors) != len(frames):
        raise ValueError(f"memory: {len(frames)} frames but {len(vectors)} embeddings")
    return frames, vectors


class GeometryObs:
    """The 3D observations of the build's posed frames (geometry.npz), by frame row: what a live pose matches."""

    def __init__(self, path: Path):
        z = np.load(path)
        order = np.argsort(z["obs_frame"], kind="stable")
        self.frame = z["obs_frame"][order]
        self.xy = z["obs_xy"][order].astype(np.float32)
        self.xyz = z["points"][z["obs_point"][order]].astype(np.float32)

    def obs(self, row: int) -> np.ndarray:
        """[K, 5]: pixel x, y (at the model frames' size) and map-frame x, y, z of the frame's observations."""
        a, b = np.searchsorted(self.frame, [row, row + 1])
        return np.concatenate([self.xy[a:b], self.xyz[a:b]], axis=1)


def load_geometry_obs(map_dir: Path, info: MemoryInfo) -> GeometryObs | None:
    return GeometryObs(map_dir / info.geometry) if info.geometry else None


def pose_key(map_id: str, build_id: str, frame_id: str) -> str:
    """The inference host's cache key for a memory frame's features: nav-api's live poses, map-api's warm-up."""
    return f"{map_id}/{build_id}/{frame_id}"


PoseRefSource = tuple[str, Path, Callable[[], np.ndarray]]  # (cache key, image file, loader of its 3D observations)


def pose_refs(map_id: str, map_dir: Path, pmap: PlaceMap) -> list[PoseRefSource]:
    """Every posed memory frame of a published map, as live poses send it: (cache key, image file, loader of its 3D
    observations). [] without a 3D model."""
    if not pmap.memory or not pmap.memory.geometry:
        return []
    frames = [MemoryFrame.model_validate(f) for f in json.loads((map_dir / pmap.memory.file).read_text())["frames"]]
    geo = load_geometry_obs(map_dir, pmap.memory)
    return [(pose_key(map_id, pmap.build.build_id, f.id), map_dir / f.image, (lambda r=f.row: geo.obs(r)))
            for f in frames if f.posed and f.row is not None]
