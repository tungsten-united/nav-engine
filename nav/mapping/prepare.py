"""Per-recording preprocessing, cached under CACHE_DIR/<rec_id>/ (recordings don't change once finalized).

    cache/<rec_id>/
      frames-<key>/NNNNNNN.jpg  sampled frames, named by t_ms
      frames-<key>.json         [{t_ms, sharpness, brightness, file}]
      emb-<key>.npy             frame embeddings (DINOv2), same order
      place-<key>.npy           place embeddings (MegaLoc), same order
      voice-<key>.json          {clip: text}

Each key is a short hash of the stage's inputs (parameters, model name, video size), so changing a parameter
or a model recomputes only what depends on it and old entries can simply be deleted.
"""

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from nav.mapping.motion import MotionTrack, load_motion
from nav.mapping.select import Frame, SelectParams
from nav.schemas.recording import Event, Manifest

log = logging.getLogger(__name__)
FRAMES_VERSION = 1


def _key(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:10]


@dataclass
class Prepared:
    manifest: Manifest
    track: MotionTrack
    frames: list[Frame]
    marks: list[dict]  # [{t_ms, tag}]
    voice: list[dict]  # [{t_ms, t_end_ms, clip, text}]


def load_events(rec_dir: Path) -> list[Event]:
    path = rec_dir / "events.jsonl"
    return [Event.model_validate_json(line) for line in path.read_text().splitlines() if line] if path.is_file() else []


def prepare(rec_dir: Path, cache_root: Path, p: SelectParams, embed_model: str, place_model: str,
            transcribe_model: str | None) -> Prepared:
    m = Manifest.model_validate_json((rec_dir / "manifest.json").read_text())
    cache = cache_root / m.rec_id
    cache.mkdir(parents=True, exist_ok=True)
    events = load_events(rec_dir)
    video_start = next((e.t_ms for e in events if e.type == "video_start"), 0.0)
    track = load_motion(rec_dir, m.step_length_m, m.duration_ms, m.device.user_agent)

    # frames
    video = rec_dir / (m.video.file if m.video else "video.mp4")
    fkey = _key(FRAMES_VERSION, p.fps, p.image_long_side, video_start, video.stat().st_size if video.is_file() else 0)
    fdir, fmeta = cache / f"frames-{fkey}", cache / f"frames-{fkey}.json"
    if not fmeta.is_file():
        rows = []
        if video.is_file():
            from nav.mapping.frames import sample_frames

            log.info("%s: sampling frames", m.rec_id)
            fdir.mkdir(exist_ok=True)
            for sf in sample_frames(video, video_start, p.fps, p.image_long_side):
                name = f"{round(sf.t_ms):07d}.jpg"
                Image.fromarray(sf.image).save(fdir / name, quality=88)
                rows.append({"t_ms": sf.t_ms, "sharpness": sf.sharpness, "brightness": sf.brightness, "file": name})
        fmeta.write_text(json.dumps(rows))
    rows = json.loads(fmeta.read_text())

    # embeddings
    ekey = _key(fkey, embed_model)
    epath = cache / f"emb-{ekey}.npy"
    if not epath.is_file():
        from nav.mapping.models import embed_images

        log.info("%s: embedding %d frames with %s", m.rec_id, len(rows), embed_model)
        emb = embed_images([np.asarray(Image.open(fdir / r["file"]).convert("RGB")) for r in rows], embed_model)
        np.save(epath, emb)
    emb = np.load(epath)
    pkey = _key(fkey, place_model)
    ppath = cache / f"place-{pkey}.npy"
    if not ppath.is_file():
        from nav.mapping.models import embed_places

        log.info("%s: place embeddings for %d frames with %s", m.rec_id, len(rows), place_model)
        np.save(ppath, embed_places([np.asarray(Image.open(fdir / r["file"]).convert("RGB")) for r in rows], place_model))
    place = np.load(ppath)

    # voice
    texts: dict[str, str] = {}
    if transcribe_model and m.voice:
        vkey = _key(transcribe_model, [(v.clip, v.bytes) for v in m.voice])
        vpath = cache / f"voice-{vkey}.json"
        if not vpath.is_file():
            from nav.mapping.models import transcribe

            log.info("%s: transcribing %d voice notes with %s", m.rec_id, len(m.voice), transcribe_model)
            out = {}
            for v in m.voice:
                try:
                    out[str(v.clip)] = transcribe(rec_dir / v.file, transcribe_model)
                except Exception as e:  # a broken clip shouldn't sink the map
                    log.warning("%s: voice clip %d: %s", m.rec_id, v.clip, e)
            vpath.write_text(json.dumps(out, ensure_ascii=False, indent=1))
        texts = json.loads(vpath.read_text())

    frames = [Frame(rec_id=m.rec_id, t_ms=r["t_ms"], sharpness=r["sharpness"], brightness=r["brightness"],
                    image_path=str(fdir / r["file"]), emb=emb[i], place=place[i], motion=track.at(r["t_ms"]))
              for i, r in enumerate(rows)]
    marks = [{"t_ms": e.t_ms, "tag": e.data.get("tag", "?")} for e in events if e.type == "mark"]
    voice = [{"t_ms": v.t_start_ms, "t_end_ms": v.t_end_ms, "clip": v.clip, "text": texts.get(str(v.clip), "")}
             for v in m.voice]
    return Prepared(m, track, frames, marks, voice)


def load_motion_only(rec_id: str, recordings_root: Path | None = None) -> MotionTrack:
    from nav import config

    rec_dir = (recordings_root or config.RECORDINGS_DIR) / rec_id
    m = Manifest.model_validate_json((rec_dir / "manifest.json").read_text())
    return load_motion(rec_dir, m.step_length_m, m.duration_ms, m.device.user_agent)
