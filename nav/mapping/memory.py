"""The visual memory of a place: where every camera frame of every walk was taken, for live localization.

It splits "where" from "what". The semantic map (nodes, edges, instructions: the LLM's) says what is at a place; the
memory says where a picture was taken. Localization against it answers "which frame of the walks does this look
like, so where am I", which a picture can answer, instead of "which node", which it often can't: a node is an
area, and in an open hall the same view belongs to several of them (Itnig: one walk's "Stage" frames are another's
"Stone wall tables", 78 of 84 confident cross-walk matches that "disagreed" with the node were neighbours).

Every usable frame (the per-recording cache: 1 fps, the sharpest of each second; the reference images' quality
filter) of every walk of the build, with
- its MegaLoc place embedding (the cache's, prepare.place_embeddings) -> memory.npy, row i = frame i;
- the camera position in the build's map frame: the walk's track as the build placed it (aligned through nodes,
  bent by the loop closure), and the camera heading;
- the image, memory/mNNNN.jpg: what the live verifier (LightGlue on the inference host) matches against, and
  what the debugging pages show.

Frames don't name a node. nav-api gives each the node nearest to its position (navigation/localize.py MemoryIndex),
so a review that moves, merges, hides or rescales nodes moves the regions with it, without rebuilding the memory.
"""

import json
import logging
import shutil
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


def add_memory(pmap: PlaceMap, params: dict, build_dir: Path, recordings_root: Path, cache_root: Path,
               place_model: str) -> PlaceMap:
    warnings: list[str] = []
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
