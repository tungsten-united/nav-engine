"""Per-recording preprocessing, cached under CACHE_DIR/<rec_id>/ (recordings don't change once finalized).

    cache/<rec_id>/
      frames-<key>/NNNNNNN.jpg  sampled frames, named by t_ms
      frames-<key>.json         [{t_ms, sharpness, brightness, file}]  (t_ms: video clock, before calibration)
      calib-<key>.json          sensor-to-video offset, field of view, and the video yaw (calibrate.py)
      odo-<key>.json            floor odometry: metres walked every half second (odometry.py)
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

from nav.infer.errors import InferenceUnavailable
from nav.mapping.calibrate import CALIB_VERSION, Calibration, calibrate, default_fov, video_yaw
from nav.mapping.inference import embed_files, place_files, transcribe_file
from nav.mapping.motion import MotionTrack, load_motion, video_motion
from nav.mapping.odometry import (CAMERA_HEIGHT_M, ODOMETRY_VERSION, FloorSegment, floor_odometry, fuse)
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
    calibration: Calibration


@dataclass
class Walk:
    """A recording's motion and sampled frames, on the sensor clock (frame times corrected by the calibration)."""

    manifest: Manifest
    events: list[Event]
    track: MotionTrack
    frames_dir: Path
    rows: list[dict]  # [{t_ms, sharpness, brightness, file}], t_ms corrected
    calibration: Calibration


def load_events(rec_dir: Path) -> list[Event]:
    path = rec_dir / "events.jsonl"
    return [Event.model_validate_json(line) for line in path.read_text().splitlines() if line] if path.is_file() else []


def _video(rec_dir: Path, m: Manifest) -> Path:
    return rec_dir / (m.video.file if m.video else "video.mp4")


def _calibration(rec_dir: Path, m: Manifest, cache: Path, video_start: float, fkey: str, track: MotionTrack,
                 enabled: bool) -> tuple[Calibration, dict | None]:
    """Calibration (and the video yaw, which a recording without sensors needs as its heading), cached."""
    video = _video(rec_dir, m)
    needs_yaw = track.source == "none"
    if not video.is_file() or not (enabled or needs_yaw):
        return Calibration(offset_ms=0, fov_deg=default_fov(m.holding)), None
    path = cache / f"calib-{_key(CALIB_VERSION, fkey, m.holding, track.source)}.json"
    if not path.is_file():
        log.info("%s: video yaw for calibration", m.rec_id)
        t_v, yaw = video_yaw(video, video_start)
        cal = (calibrate(track.t_ms, track.heading_deg, t_v, yaw, m.holding) if track.source == "imu"
               else Calibration(offset_ms=0, fov_deg=default_fov(m.holding)))
        path.write_text(json.dumps({"calibration": cal.as_dict(),
                                    "yaw": {"t_ms": [round(float(t), 1) for t in t_v],
                                            "units": [round(float(u), 5) for u in yaw]}}))
    data = json.loads(path.read_text())
    cal = Calibration(**data["calibration"])
    if not enabled:
        cal = Calibration(offset_ms=0, fov_deg=default_fov(m.holding))
    return cal, data["yaw"]


def _odometry(rec_dir: Path, m: Manifest, cache: Path, video_start: float, fkey: str, track: MotionTrack,
              cal: Calibration, enabled: bool) -> MotionTrack:
    """The track with walked metres from the floor odometry where it is valid (cached), else unchanged."""
    video = _video(rec_dir, m)
    if not enabled or track.source != "imu" or not video.is_file():
        return track
    path = cache / f"odo-{_key(ODOMETRY_VERSION, fkey, m.holding, cal.offset_ms, cal.fov_deg, CAMERA_HEIGHT_M)}.json"
    if not path.is_file():
        log.info("%s: floor odometry", m.rec_id)
        segs = floor_odometry(video, video_start, cal.offset_ms, cal.fov_deg, track, m.holding)
        path.write_text(json.dumps(None if segs is None else [s.as_dict() for s in segs]))
    data = json.loads(path.read_text())
    fused = fuse(track, None if data is None else [FloorSegment(**d) for d in data], m.step_length_m)
    if data is not None:
        log.info("%s: walked %.1f m (%.0f%% floor odometry; steps x %.2f m said %.1f m)", m.rec_id,
                 fused.track.walked_m[-1], 100 * fused.video_share, m.step_length_m, track.walked_m[-1])
    return fused.track


def load_walk(rec_dir: Path, cache_root: Path, p: SelectParams) -> Walk:
    """Frames (sampled once, cached), motion, and calibration of one recording."""
    m = Manifest.model_validate_json((rec_dir / "manifest.json").read_text())
    cache = cache_root / m.rec_id
    cache.mkdir(parents=True, exist_ok=True)
    events = load_events(rec_dir)
    video_start = next((e.t_ms for e in events if e.type == "video_start"), 0.0)
    track = load_motion(rec_dir, m.step_length_m, m.duration_ms, m.device.user_agent)

    # frames
    video = _video(rec_dir, m)
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

    cal, yaw = _calibration(rec_dir, m, cache, video_start, fkey, track, p.calibrate)
    if track.source == "none" and yaw and yaw["t_ms"]:  # no sensors: the video's own yaw is the heading
        track = video_motion(np.array(yaw["t_ms"]), np.array(yaw["units"]), cal.fov_deg, m.duration_ms)
    track = _odometry(rec_dir, m, cache, video_start, fkey, track, cal, p.odometry)
    rows = [{**r, "t_ms": round(r["t_ms"] - cal.offset_ms, 1)} for r in rows]
    return Walk(m, events, track, fdir, rows, cal)


def prepare(rec_dir: Path, cache_root: Path, p: SelectParams, embed_model: str, place_model: str,
            transcribe_model: str | None) -> Prepared:
    w = load_walk(rec_dir, cache_root, p)
    m, events, track, fdir, rows = w.manifest, w.events, w.track, w.frames_dir, w.rows
    cache = cache_root / m.rec_id
    fkey = fdir.name.removeprefix("frames-")

    # embeddings
    ekey = _key(fkey, embed_model)
    epath = cache / f"emb-{ekey}.npy"
    if not epath.is_file():
        log.info("%s: embedding %d frames with %s", m.rec_id, len(rows), embed_model)
        np.save(epath, embed_files([fdir / r["file"] for r in rows], embed_model))
    emb = np.load(epath)
    pkey = _key(fkey, place_model)
    ppath = cache / f"place-{pkey}.npy"
    if not ppath.is_file():
        log.info("%s: place embeddings for %d frames with %s", m.rec_id, len(rows), place_model)
        np.save(ppath, place_files([fdir / r["file"] for r in rows], place_model))
    place = np.load(ppath)

    # voice
    texts: dict[str, str] = {}
    if transcribe_model and m.voice:
        vkey = _key(transcribe_model, [(v.clip, v.bytes) for v in m.voice])
        vpath = cache / f"voice-{vkey}.json"
        if not vpath.is_file():
            log.info("%s: transcribing %d voice notes with %s", m.rec_id, len(m.voice), transcribe_model)
            out = {}
            for v in m.voice:
                try:
                    out[str(v.clip)] = transcribe_file(rec_dir / v.file, transcribe_model)
                except InferenceUnavailable:  # not the clip's fault: don't cache a transcript without it
                    raise
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
    return Prepared(m, track, frames, marks, voice, w.calibration)


def load_motion_only(rec_id: str, recordings_root: Path | None = None, cache_root: Path | None = None) -> MotionTrack:
    from nav import config

    rec_dir = (recordings_root or config.RECORDINGS_DIR) / rec_id
    m = Manifest.model_validate_json((rec_dir / "manifest.json").read_text())
    if _video(rec_dir, m).is_file():  # the same track a build uses: calibration, video yaw, floor odometry (cached)
        return load_walk(rec_dir, cache_root or config.CACHE_DIR, SelectParams()).track
    return load_motion(rec_dir, m.step_length_m, m.duration_ms, m.device.user_agent)
