"""Geometry: a 3D model of the place from every walk's video (structure from motion). It gives the map its positions
and live localization its pose (bench/metric/REPORT.md: 90% of held-out Itnig frames within 1 m and 10°).

    frames   each walk at FPS, the sharpest frame of every 1/FPS s, LONG_SIDE px, on the sensor clock (cached per walk)
    gpu      ALIKED features (MAX_KEYPOINTS) and MegaLoc embeddings of every frame; pairs: the same walk up to SEQ_S
             ahead + the RETRIEVAL_K most similar frames anywhere; LightGlue matches for each pair. A queued job on the
             shared GPU (gpu-run) when this process doesn't hold the GPU itself (the inference host, an agent session)
    mapping  COLMAP global mapping (GLOMAP, pycolmap), one camera per walk with its calibrated field of view as prior;
             the largest component is the model [CPU]
    frame    the model levelled (the cameras' "down" is gravity: the phone is held upright), scaled to the walks' floor
             odometry, turned so camera headings agree with the walks' (compass) headings, and placed so the first
             posed walk sits where its dead reckoning puts it: the map frame, x east and y north, metres
    check    global mapping can join a weak walk wrongly (Nati House, one night walk: frames flung metres away, a
             stretch glued in the wrong place). Each walk is compared with its own motion: a frame that disagrees
             with its neighbours by more than LOCAL_M, or a stretch whose offset from dead reckoning jumps from the
             walk's main stretch by more than the drift allows, is not posed. Those frames keep dead-reckoned positions
             moved into the map frame, and add nothing to the 3D points live localization matches against.

The work happens in a job folder (the same on the inference host and locally): job.json, images/<rec>/<t>.jpg, then
gpu/ (features, matches, embeddings), sparse/ (COLMAP) and raw.npz, the model in its own frame. `place_in_map` turns
raw.npz into a Geometry in the map frame; `write`/`load` keep it in the build (geometry.json + geometry.npz).
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

GEOMETRY_VERSION = 1
FPS = 2.0  # frames a second of the model ...
FPS_SHORT = 4.0  # ... and of a place walked once or briefly: denser frames bridge fast turns (Nati House)
SHORT_S = 240  # "briefly": all its walks together
LONG_SIDE = 1024  # px: frames of the model and of the memory; live frames are matched at this size too
MIN_BRIGHTNESS = 20
MAX_KEYPOINTS = 2048
SEQ_S = 2.5
RETRIEVAL_K = 12
SCALE_WINDOW_MS = 3000  # model displacement vs dead-reckoned displacement over windows 1-2x this long
SCALE_MIN_WALKED_M = 1.0
SCALE_MIN_WINDOWS = 3  # a piece with fewer such windows has no scale (each piece of a model has its own): not posed
LOCAL_WINDOW_MS = 15_000  # a frame's offset from dead reckoning vs its neighbours' within this window ...
LOCAL_M = 2.5  # ... may differ by this much (dead reckoning drifts little in 15 s)
STEP_M = 1.0  # consecutive frames' offsets differing more than this split the walk into stretches ...
STEP_GAP_MS = 1500  # ... and so does a gap this long between trusted frames
DRIFT_M = 1.5  # a stretch's offset may differ from the main stretch's by this ...
DRIFT_M_PER_S = 0.05  # ... plus this per second between them (dead reckoning drifts a few metres a minute)
MIN_POSED_SHARE = 0.25  # fewer posed frames than this share: no geometry for the build (dead reckoning only)


# --- frames ----------------------------------------------------------------------------------------------------
@dataclass
class GeoFrame:
    name: str  # "<rec_id>/<t_ms>.jpg" under the job's images/
    rec_id: str
    t_ms: float  # sensor clock
    sharpness: float
    brightness: float


def _key(*parts) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:10]


def fps_for(rec_dirs: list[Path]) -> float:
    from nav.schemas.recording import Manifest

    seconds = sum((Manifest.model_validate_json((d / "manifest.json").read_text()).duration_ms or 0) / 1000
                  for d in rec_dirs)
    return FPS_SHORT if len(rec_dirs) == 1 or seconds < SHORT_S else FPS


def sample_walk(rec_dir: Path, cache_root: Path, fps: float = FPS) -> tuple[Path, list[GeoFrame], dict]:
    """The walk's model frames (cached in cache/<rec>/geo-<key>/), and its camera: {fov_deg, width, height}."""
    from PIL import Image

    from nav.mapping.frames import sample_frames
    from nav.mapping.prepare import _video, load_walk
    from nav.mapping.select import SelectParams

    walk = load_walk(rec_dir, cache_root, SelectParams())
    rec = walk.manifest.rec_id
    start = next((e.t_ms for e in walk.events if e.type == "video_start"), 0.0)
    video = _video(rec_dir, walk.manifest)
    key = _key(GEOMETRY_VERSION, fps, LONG_SIDE, start, walk.calibration.offset_ms,
               video.stat().st_size if video.is_file() else 0)
    fdir, meta = cache_root / rec / f"geo-{key}", cache_root / rec / f"geo-{key}.json"
    if not meta.is_file():
        log.info("%s: model frames at %.0f fps", rec, fps)
        if fdir.exists():
            shutil.rmtree(fdir)
        fdir.mkdir(parents=True)
        rows, size = [], None
        for sf in sample_frames(video, start, fps, LONG_SIDE) if video.is_file() else []:
            if sf.brightness < MIN_BRIGHTNESS:
                continue
            t = round(sf.t_ms - walk.calibration.offset_ms, 1)
            name = f"{round(t):07d}.jpg"
            if any(r["file"] == name for r in rows[-2:]):
                continue  # two frames within a millisecond at a bucket edge
            Image.fromarray(sf.image).save(fdir / name, quality=90)
            size = sf.image.shape[:2]
            rows.append({"file": name, "t_ms": t, "sharpness": sf.sharpness, "brightness": sf.brightness})
        cam = {"fov_deg": walk.calibration.fov_deg, "width": size[1] if size else 0, "height": size[0] if size else 0}
        meta.write_text(json.dumps({"rows": rows, "camera": cam}))
    got = json.loads(meta.read_text())
    frames = [GeoFrame(f"{rec}/{r['file']}", rec, r["t_ms"], r["sharpness"], r["brightness"]) for r in got["rows"]]
    return fdir, frames, got["camera"]


def prepare_job(work: Path, rec_dirs: list[Path], cache_root: Path, place_model: str) -> list[GeoFrame]:
    """A job folder for these walks: images linked (or copied) from the cache, job.json."""
    frames, walks = [], {}
    fps = fps_for(rec_dirs)
    for rec_dir in rec_dirs:
        fdir, fs, cam = sample_walk(rec_dir, cache_root, fps)
        if not fs:
            continue
        rec = fs[0].rec_id
        (work / "images" / rec).mkdir(parents=True, exist_ok=True)
        for f in fs:
            dst = work / "images" / f.name
            if not dst.exists():
                try:
                    os.link(fdir / Path(f.name).name, dst)
                except OSError:
                    shutil.copyfile(fdir / Path(f.name).name, dst)
        frames += fs
        walks[rec] = cam
    (work / "job.json").write_text(json.dumps({"version": GEOMETRY_VERSION, "place_model": place_model, "fps": fps,
                                               "frames": [asdict(f) for f in frames], "walks": walks}))
    return frames


def job_frames(work: Path) -> tuple[list[GeoFrame], dict, str]:
    j = json.loads((work / "job.json").read_text())
    return [GeoFrame(**f) for f in j["frames"]], j["walks"], j["place_model"]


# --- GPU: features, embeddings, matches ------------------------------------------------------------------------
def _load_rgb(path: Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"))


_aliked_model = None


def aliked():
    global _aliked_model
    if _aliked_model is None:
        from lightglue import ALIKED

        from nav.mapping.models import _device

        _aliked_model = ALIKED(max_num_keypoints=MAX_KEYPOINTS).to(_device()).eval()
    return _aliked_model


def extract(rgb: np.ndarray) -> dict:
    """ALIKED keypoints (pixel coordinates) and descriptors of an RGB image, on the device."""
    import torch

    from nav.mapping.models import _device

    x = torch.tensor(rgb).permute(2, 0, 1).float().div(255).to(_device())
    with torch.inference_mode():
        f = aliked().extract(x, resize=None)
    return {"keypoints": f["keypoints"], "descriptors": f["descriptors"].half(), "image_size": f["image_size"]}


def match(f0: dict, f1: dict) -> np.ndarray:
    """LightGlue matches [N, 2] (keypoint indexes) between two extract() results."""
    import torch

    from nav.mapping.models import _local

    with torch.inference_mode():
        m = _local()[1]({"image0": {**f0, "descriptors": f0["descriptors"].float()},
                         "image1": {**f1, "descriptors": f1["descriptors"].float()}})["matches"][0]
    return m.cpu().numpy().astype(np.uint32)


def make_pairs(frames: list[GeoFrame], place: np.ndarray) -> list[tuple[int, int]]:
    """Indexes (i < j): the frames of the same walk up to SEQ_S ahead, and each frame's RETRIEVAL_K most similar
    frames that aren't just its walk's next ones."""
    pairs: set[tuple[int, int]] = set()
    by_rec: dict[str, list[int]] = {}
    for i, f in enumerate(frames):
        by_rec.setdefault(f.rec_id, []).append(i)
    for idx in by_rec.values():
        idx.sort(key=lambda i: frames[i].t_ms)
        for a, i in enumerate(idx):
            for j in idx[a + 1:]:
                if frames[j].t_ms - frames[i].t_ms > SEQ_S * 1000:
                    break
                pairs.add((min(i, j), max(i, j)))
    sim = place @ place.T
    for i, f in enumerate(frames):
        n = 0
        for j in np.argsort(-sim[i]):
            o = frames[j]
            if j == i or (o.rec_id == f.rec_id and abs(o.t_ms - f.t_ms) <= SEQ_S * 1000):
                continue
            pairs.add((min(i, int(j)), max(i, int(j))))
            n += 1
            if n >= RETRIEVAL_K:
                break
    return sorted(pairs)


def gpu_stage(work: Path) -> None:
    """Features, embeddings and matches of the job's frames -> work/gpu/ (keypoints.npz, matches.npz, place.npy)."""
    from nav.mapping.models import embed_places

    frames, _, place_model = job_frames(work)
    out = work / "gpu"
    out.mkdir(exist_ok=True)
    t0 = time.time()
    feats = []
    for k, f in enumerate(frames):
        feats.append(extract(_load_rgb(work / "images" / f.name)))
        if k % 200 == 199:
            log.info("geometry: features %d/%d", k + 1, len(frames))
    place = []
    for s in range(0, len(frames), 32):
        place.append(embed_places([_load_rgb(work / "images" / f.name) for f in frames[s:s + 32]], place_model))
    place = np.concatenate(place).astype(np.float32)
    pairs = make_pairs(frames, place)
    log.info("geometry: %d frames, %d pairs (features and embeddings in %.0f s)", len(frames), len(pairs),
             time.time() - t0)
    found = []
    for k, (i, j) in enumerate(pairs):
        found.append(match(feats[i], feats[j]))
        if k % 2000 == 1999:
            log.info("geometry: matches %d/%d", k + 1, len(pairs))
    kps = [f["keypoints"][0].cpu().numpy().astype(np.float32) for f in feats]
    np.savez(out / "keypoints.npz", kps=np.concatenate(kps), offsets=np.cumsum([0] + [len(k) for k in kps]))
    np.savez(out / "matches.npz", pairs=np.array(pairs, np.int32).reshape(-1, 2),
             matches=np.concatenate(found) if found else np.zeros((0, 2), np.uint32),
             offsets=np.cumsum([0] + [len(m) for m in found]))
    np.save(out / "place.npy", place)
    (out / "done.json").write_text(json.dumps({"frames": len(frames), "pairs": len(pairs),
                                               "seconds": round(time.time() - t0)}))
    log.info("geometry: GPU stage done in %.0f s", time.time() - t0)


def _holds_gpu() -> bool:
    """This process can use a GPU itself (a job the queue started, or a machine without the queue)."""
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) == "":
        return False
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def run_gpu_stage(work: Path, queue: bool | None = None, max_time: str = "45m") -> None:
    """The GPU stage: here when this process holds the GPU, else as a job on the shared GPU's queue (gpu-run), which
    waits for its turn. `queue` forces it (the inference host, a resident service, queues its batch work)."""
    if (work / "gpu" / "done.json").is_file():
        return
    runner = shutil.which("gpu-run")
    if queue is None:
        queue = runner is not None and not _holds_gpu()
    if not queue or runner is None:
        gpu_stage(work)
        return
    nav = Path(sys.executable).with_name("nav")
    cmd = [runner, "--label", f"nav-engine/geometry-{work.parent.name[:40]}", "--max", max_time, "--",
           str(nav), "infer", "geometry-gpu", str(work)]
    log.info("geometry: GPU stage queued: %s", " ".join(cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (work / "gpu-run.log").write_text(proc.stdout or "")
    if proc.returncode != 0 or not (work / "gpu" / "done.json").is_file():
        raise RuntimeError(f"geometry GPU job exited {proc.returncode}: {(proc.stdout or '')[-800:]}")


# --- mapping (CPU) ---------------------------------------------------------------------------------------------
def _jumpy(model, frames: list[GeoFrame], index: dict[str, int], fps: float = FPS) -> float:
    """The share of a walk's consecutive frames (half a second apart) the model puts many times their usual step
    apart, for the worst walk: a global model that joined a walk wrongly flings frames (Nati House: ~10%)."""
    worst = 0.0
    by_rec: dict[str, list[tuple[float, np.ndarray]]] = {}
    for iid in model.reg_image_ids():
        im = model.images[iid]
        f = frames[index[im.name]]
        by_rec.setdefault(f.rec_id, []).append((f.t_ms, np.asarray(im.projection_center(), float).reshape(3)))
    for seq in by_rec.values():
        seq.sort(key=lambda x: x[0])
        steps = [np.linalg.norm(b[1] - a[1]) for a, b in zip(seq, seq[1:]) if b[0] - a[0] <= 1500 / fps]
        if len(steps) >= 10:
            med = float(np.median(steps)) or 1e-9
            worst = max(worst, float(np.mean(np.array(steps) > JUMP_X * med)))
    return worst


JUMP_X = 8  # a step this many times the median one is a jump
MAX_JUMPY = 0.02  # a global model with more jumps than this share is mapped again incrementally
MIN_PIECE_FRAMES = 20  # model components kept (pieces of a walk the model couldn't join)


def map_stage(work: Path) -> dict:
    """COLMAP mapping from work/gpu -> work/raw.npz: global mapping (GLOMAP), or incremental when the global model
    flings frames (see _jumpy); every component with MIN_PIECE_FRAMES frames, each in its own frame + summary."""
    import pycolmap

    frames, walks, _ = job_frames(work)
    kz, mz = np.load(work / "gpu" / "keypoints.npz"), np.load(work / "gpu" / "matches.npz")
    kps, koff = kz["kps"], kz["offsets"]
    pairs, allm, moff = mz["pairs"], mz["matches"], mz["offsets"]
    d = work / "colmap"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir()
    db_path = d / "database.db"
    opts = pycolmap.ImageReaderOptions()
    opts.camera_model = "SIMPLE_RADIAL"
    pycolmap.Database.open(db_path).close()
    pycolmap.import_images(db_path, work / "images", pycolmap.CameraMode.PER_FOLDER, [f.name for f in frames], opts)
    db = pycolmap.Database.open(db_path)
    images = db.read_all_images()
    ids = {im.name: im.image_id for im in images}
    cam_rec = {im.camera_id: im.name.split("/")[0] for im in images}
    for cam in db.read_all_cameras():
        fov = walks[cam_rec[cam.camera_id]]["fov_deg"] or 50.0
        cam.params = [0.5 * cam.width / math.tan(math.radians(fov) / 2), cam.width / 2, cam.height / 2, 0.0]
        cam.has_prior_focal_length = True
        db.update_camera(cam)
    for i, f in enumerate(frames):
        db.write_keypoints(ids[f.name], kps[koff[i]:koff[i + 1]] + 0.5)  # COLMAP: pixel centres at +0.5
    lines = []
    for k, (i, j) in enumerate(pairs):
        m = allm[moff[k]:moff[k + 1]]
        if len(m):
            db.write_matches(ids[frames[i].name], ids[frames[j].name], m)
            lines.append(f"{frames[i].name} {frames[j].name}")
    db.close()
    (d / "pairs.txt").write_text("\n".join(lines) + "\n")
    t0 = time.time()
    pycolmap.verify_matches(db_path, d / "pairs.txt", pycolmap.TwoViewGeometryOptions())
    index = {f.name: i for i, f in enumerate(frames)}
    mapper = "global"
    recs = pycolmap.global_mapping(db_path, work / "images", d / "global")
    comps = sorted(recs.values(), key=lambda r: -r.num_reg_images())
    fps = json.loads((work / "job.json").read_text()).get("fps", FPS)
    jumpy = _jumpy(comps[0], frames, index, fps) if comps else 1.0
    if jumpy > MAX_JUMPY:
        log.info("geometry: the global model flings %.0f%% of a walk's frames: mapping incrementally", 100 * jumpy)
        o = pycolmap.IncrementalPipelineOptions()
        o.num_threads = -1
        o.max_runtime_seconds = 1800
        recs = pycolmap.incremental_mapping(db_path, work / "images", d / "incremental", o)
        inc = sorted(recs.values(), key=lambda r: -r.num_reg_images())
        if inc:
            comps, mapper = inc, "incremental"
    comps = [r for r in comps if r.num_reg_images() >= MIN_PIECE_FRAMES]
    if not comps:
        raise RuntimeError("geometry: COLMAP found no model")
    n = len(frames)
    component = -np.ones(n, np.int32)
    R, t, focal = np.zeros((n, 3, 3)), np.zeros((n, 3)), np.zeros(n)
    all_points, point_comp, obs = [], [], []
    for c, model in enumerate(comps):
        iid_to_frame = {}
        for iid in model.reg_image_ids():
            im = model.images[iid]
            i = index[im.name]
            if component[i] >= 0:
                continue  # in two components (incremental mapping overlaps them): the larger keeps it
            cfw = im.cam_from_world()
            component[i] = c
            R[i] = np.asarray(cfw.rotation.matrix())
            t[i] = np.asarray(cfw.translation, float).reshape(3)
            focal[i] = float(model.cameras[im.camera_id].focal_length)
            iid_to_frame[iid] = i
        base = len(all_points)
        pids = sorted(model.points3D)
        pindex = {p: base + k for k, p in enumerate(pids)}
        all_points += [np.asarray(model.points3D[p].xyz, float).reshape(3) for p in pids]
        point_comp += [c] * len(pids)
        obs += [(iid_to_frame[el.image_id], el.point2D_idx, pindex[p]) for p in pids
                for el in model.points3D[p].track.elements if el.image_id in iid_to_frame]
    points = np.array(all_points).reshape(-1, 3)
    obs = np.array(obs, np.int64).reshape(-1, 3)
    np.savez(work / "raw.npz", registered=component >= 0, component=component, R=R, t=t, focal=focal,
             points=points, point_component=np.array(point_comp, np.int32), obs_frame=obs[:, 0].astype(np.int32),
             obs_kp=obs[:, 1].astype(np.int32), obs_point=obs[:, 2].astype(np.int32), kps=kps, kp_offsets=koff,
             place=np.load(work / "gpu" / "place.npy"))
    summary = {"frames": n, "pairs": len(lines), "registered": int((component >= 0).sum()), "points": len(points),
               "mapper": mapper, "global_jumpy": round(jumpy, 3), "components": [r.num_reg_images() for r in comps[:8]],
               "mapping_s": round(time.time() - t0)}
    (work / "summary.json").write_text(json.dumps(summary))
    log.info("geometry: %s model of %d of %d frames in %d pieces (%s), %d points in %.0f s", mapper,
             summary["registered"], n, len(comps), summary["components"][:4], len(points), time.time() - t0)
    return summary


def run_job(work: Path, queue: bool | None = None) -> dict:
    """GPU stage (queued when needed), then mapping here. -> summary; raw.npz in work."""
    run_gpu_stage(work, queue)
    return map_stage(work)


# --- into the map frame ----------------------------------------------------------------------------------------
@dataclass
class Geometry:
    """The model in the map frame. Frame i of `frames` is posed (a camera pose from the model) or not (positions
    from its walk's dead reckoning, moved into the map frame, or None)."""

    frames: list[GeoFrame]
    posed: np.ndarray  # [F] bool
    xyz: np.ndarray  # [F, 3] camera centres (nan without a position)
    heading_deg: np.ndarray  # [F] camera heading, clockwise from +y (nan unknown)
    R: np.ndarray  # [F, 3, 3] cam_from_world in the map frame (posed frames)
    focal: np.ndarray  # [F] px at the frame's size
    points: np.ndarray  # [P, 3] the 3D points posed frames see
    obs_frame: np.ndarray  # [K] observations: frame ...
    obs_xy: np.ndarray  # [K, 2] ... keypoint (pixels, +0.5 COLMAP convention removed) ...
    obs_point: np.ndarray  # [K] ... and its 3D point
    place: np.ndarray  # [F, D] MegaLoc embeddings
    info: dict = field(default_factory=dict)
    piece: np.ndarray | None = None  # [F] the model piece a posed frame is in (-1: not posed)

    def position(self, rec_id: str, t_ms: float, max_dt_ms: float = 700) -> tuple[float, float] | None:
        """Where the walk was at t_ms: its nearest frame within max_dt_ms with a position."""
        best = None
        for i, f in enumerate(self.frames):
            if f.rec_id == rec_id and not np.isnan(self.xyz[i, 0]):
                dt = abs(f.t_ms - t_ms)
                if dt <= max_dt_ms and (best is None or dt < best[0]):
                    best = (dt, i)
        return None if best is None else (float(self.xyz[best[1], 0]), float(self.xyz[best[1], 1]))


def _heading(v: np.ndarray) -> np.ndarray:
    """Clockwise from +y (north) of horizontal directions [N, 3]."""
    return np.degrees(np.arctan2(v[:, 0], v[:, 1])) % 360


def _circ_median(deg: np.ndarray) -> float:
    """Robust circular centre: the circular mean of the angles within 45° of the plain circular mean, twice."""
    c = float(np.degrees(np.angle(np.exp(1j * np.radians(deg)).mean())))
    for _ in range(2):
        d = (deg - c + 180) % 360 - 180
        near = deg[np.abs(d) <= 45]
        if not len(near):
            break
        c = float(np.degrees(np.angle(np.exp(1j * np.radians(near)).mean())))
    return c % 360


def _rot_to_z_up(down: np.ndarray) -> np.ndarray:
    """Rotation taking the world `down` direction to -z."""
    a = down / np.linalg.norm(down)
    b = np.array([0.0, 0.0, -1.0])
    v, c = np.cross(a, b), float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1 / (1 + c))


def _check_walk(t: np.ndarray, off: np.ndarray) -> np.ndarray:
    """Which of a walk's posed frames (times t, offsets model -> dead reckoning) to trust: not a lone frame that
    disagrees with its neighbours, not a stretch whose offset jumps from the walk's main stretch."""
    n = len(t)
    keep = np.ones(n, bool)
    for i in range(n):
        near = np.abs(t - t[i]) <= LOCAL_WINDOW_MS
        if near.sum() >= 3 and np.linalg.norm(off[i] - np.median(off[near], axis=0)) > LOCAL_M:
            keep[i] = False
    # stretches: split where consecutive trusted frames' offsets jump (or a long gap leaves it open)
    idx = np.flatnonzero(keep)
    if len(idx) < 2:
        return keep
    stretches, cur = [], [idx[0]]
    for a, b in zip(idx, idx[1:]):
        if np.linalg.norm(off[b] - off[a]) > STEP_M or t[b] - t[a] > STEP_GAP_MS:
            stretches.append(cur)
            cur = []
        cur.append(b)
    stretches.append(cur)
    main = max(stretches, key=lambda s: t[s[-1]] - t[s[0]] + 1e-3 * len(s))
    m_off, m_t = np.median(off[main], axis=0), float(np.median(t[main]))
    for s in stretches:
        if s is main:
            continue
        allowed = DRIFT_M + DRIFT_M_PER_S * abs(float(np.median(t[s])) - m_t) / 1000
        if np.linalg.norm(np.median(off[s], axis=0) - m_off) > allowed:
            keep[s] = False
    return keep


def _orient(C: np.ndarray, R0: np.ndarray, idx: list[int], frames: list[GeoFrame], tracks: dict
            ) -> tuple[np.ndarray, float | None, float, np.ndarray]:
    """For one piece of the model (frames `idx`): (L, scale, yaw radians, Rz) taking it to the map frame up to a
    shift: levelled by the cameras' down, scaled by walked metres, turned onto the walks' headings. Scale None: too
    little walking in the piece to measure it (a pause looking around)."""
    down = np.einsum("nji,j->ni", R0[idx], np.array([0.0, 1.0, 0.0]))
    L = _rot_to_z_up(np.median(down, axis=0))
    Cl = C @ L.T
    view = np.einsum("nji,j->ni", R0, np.array([0.0, 0.0, 1.0])) @ L.T
    ratios = []
    for rec in {frames[i].rec_id for i in idx}:
        tr = tracks.get(rec)
        seq = sorted((i for i in idx if frames[i].rec_id == rec), key=lambda i: frames[i].t_ms)
        if tr is None or len(seq) < 2:
            continue
        a = 0
        for b in range(1, len(seq)):
            i, j = seq[a], seq[b]
            dt = frames[j].t_ms - frames[i].t_ms
            if dt < SCALE_WINDOW_MS:
                continue
            if dt <= 2 * SCALE_WINDOW_MS:  # not across a gap: a piece can hold a revisit minutes later
                pi, pj = tr.at(frames[i].t_ms), tr.at(frames[j].t_ms)
                moved = math.hypot(pj["x_m"] - pi["x_m"], pj["y_m"] - pi["y_m"])  # dead reckoning, not walked
                model_m = float(np.linalg.norm(Cl[j, :2] - Cl[i, :2]))  # metres: sway and curves aren't moving
                if moved >= SCALE_MIN_WALKED_M and model_m > 1e-6:
                    ratios.append(moved / model_m)
            a = b
    scale = float(np.median(ratios)) if len(ratios) >= SCALE_MIN_WINDOWS else None
    diffs = []
    for i in idx:
        tr = tracks.get(frames[i].rec_id)
        if tr is not None and (h := tr.at(frames[i].t_ms)["heading_deg"]) is not None:
            diffs.append((h - _heading(view[i:i + 1])[0] + 180) % 360 - 180)
    yaw = math.radians(_circ_median(np.array(diffs))) if diffs else 0.0
    Rz = np.array([[math.cos(yaw), math.sin(yaw), 0], [-math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])  # clockwise
    return L, scale, yaw, Rz


def place_in_map(work: Path, tracks: dict, first_rec: str | None = None) -> Geometry | None:
    """raw.npz -> the model in the map frame (see the module doc). `tracks`: {rec_id: MotionTrack} of the walks
    (dead reckoning in each walk's own frame, compass-anchored headings). The largest piece of the model is placed
    by the first walk's dead reckoning; other pieces (parts of walks the model couldn't join) each by their own
    walk's, through where that walk is in the largest piece. None when too little is posed."""
    frames, _, _ = job_frames(work)
    raw = np.load(work / "raw.npz")
    comp = raw["component"].copy() if "component" in raw else np.where(raw["registered"], 0, -1)
    n = len(frames)
    if (comp >= 0).sum() < max(10, MIN_POSED_SHARE * n):
        log.warning("geometry: %d of %d frames in the model: not used", (comp >= 0).sum(), n)
        return None
    R0, t0 = raw["R"], raw["t"]
    reg = comp >= 0
    C = np.full((n, 3), np.nan)
    C[reg] = -np.einsum("nji,nj->ni", R0[reg], t0[reg])  # -R^T t
    recs = list(dict.fromkeys(f.rec_id for f in frames))
    order = ([first_rec] if first_rec in recs else []) + [r for r in recs if r != first_rec]

    def dr(i: int) -> np.ndarray:
        a = tracks[frames[i].rec_id].at(frames[i].t_ms)
        return np.array([a["x_m"], a["y_m"]])

    Cm = np.full((n, 3), np.nan)
    Rm = np.zeros((n, 3, 3))
    heading = np.full(n, np.nan)
    transforms: dict[int, tuple[float, np.ndarray, np.ndarray]] = {}  # piece -> (scale, M, shift)
    walk_off: dict[str, np.ndarray] = {}  # dead reckoning - map, per walk, from the largest piece
    pieces = sorted(set(comp[reg].tolist()))
    info_pieces = []
    for c in pieces:
        idx = [i for i in range(n) if comp[i] == c]
        L, scale, yaw, Rz = _orient(C, R0, idx, frames, tracks)
        if scale is None:
            if c != pieces[0] or len(pieces) > 1:
                comp[idx] = -1  # its metres are unknown: dead reckoning there
                info_pieces.append({"frames": len(idx), "scale": None})
                continue
            scale = 1.0
        M = Rz @ L
        Cc = scale * (C[idx] @ M.T)
        shift = np.zeros(3)
        if c == pieces[0]:
            for rec in order:
                ii = [k for k, i in enumerate(idx) if frames[i].rec_id == rec]
                if ii and rec in tracks:
                    shift[:2] = np.median(np.array([dr(idx[k]) for k in ii]) - Cc[ii, :2], axis=0)
                    break
            shift[2] = -float(np.median(Cc[:, 2]))
        else:
            target = [(k, dr(i) - walk_off[frames[i].rec_id]) for k, i in enumerate(idx)
                      if frames[i].rec_id in walk_off and frames[i].rec_id in tracks]
            if len(target) < MIN_PIECE_FRAMES // 2:
                comp[idx] = -1  # no walk ties it to the largest piece
                continue
            shift[:2] = np.median(np.array([tg for _, tg in target]) - Cc[[k for k, _ in target], :2], axis=0)
            shift[2] = -float(np.median(Cc[:, 2]))
        Cm[idx] = Cc + shift
        Rm[idx] = R0[idx] @ M.T  # cam_from_map = cam_from_model @ model_from_map
        view = np.einsum("nji,j->ni", R0[idx], np.array([0.0, 0.0, 1.0])) @ M.T
        heading[idx] = _heading(view)
        transforms[c] = (scale, M, shift)
        info_pieces.append({"frames": len(idx), "scale": round(scale, 4), "yaw_deg": round(math.degrees(yaw), 1)})
        if c == pieces[0]:
            for rec in recs:
                ii = [i for i in idx if frames[i].rec_id == rec]
                if ii and rec in tracks:
                    walk_off[rec] = np.median(np.array([dr(i) for i in ii]) - Cm[ii, :2], axis=0)
    reg = comp >= 0
    # check each walk against its own motion
    per_walk = {}
    xyz = np.full((n, 3), np.nan)
    for rec in recs:
        idx = np.array(sorted((i for i, f in enumerate(frames) if f.rec_id == rec), key=lambda i: frames[i].t_ms))
        tr = tracks.get(rec)
        posed_idx = idx[reg[idx]] if len(idx) else idx
        if tr is not None and len(posed_idx):
            d = np.array([dr(i) for i in idx])
            off_all = np.full((len(idx), 2), np.nan)
            pos_mask = reg[idx]
            off_all[pos_mask] = d[pos_mask] - Cm[idx[pos_mask], :2]
            keep = _check_walk(np.array([frames[i].t_ms for i in posed_idx]), off_all[pos_mask])
            comp[posed_idx[~keep]] = -1
            reg = comp >= 0
            ok = reg[idx]
            xyz[idx[ok]] = Cm[idx[ok]]
            # frames without a trusted pose: dead reckoning, by the offset of the nearest trusted frames in time
            tt = np.array([frames[i].t_ms for i in idx])
            for k in np.flatnonzero(~ok):
                if not ok.any():
                    break
                near = np.flatnonzero(ok)[np.argsort(np.abs(tt[ok] - tt[k]))[:6]]
                xyz[idx[k], :2] = d[k] - np.median(off_all[near], axis=0)
                xyz[idx[k], 2] = 0.0
            per_walk[rec] = {"frames": len(idx), "posed": int(ok.sum()), "distrusted": int((~keep).sum())}
        else:
            comp[idx] = -1
            reg = comp >= 0
            per_walk[rec] = {"frames": len(idx), "posed": 0, "distrusted": 0}
    if reg.sum() < max(10, MIN_POSED_SHARE * n):
        log.warning("geometry: %d of %d frames trusted: not used", reg.sum(), n)
        return None
    for i, f in enumerate(frames):  # unposed: the walk's own heading
        if not reg[i]:
            heading[i] = np.nan
            if (tr := tracks.get(f.rec_id)) is not None and (h := tr.at(f.t_ms)["heading_deg"]) is not None:
                heading[i] = h
    Rm[~reg] = 0.0
    # observations of trusted frames only; the points they see (each in its piece's transform)
    pc = raw["point_component"] if "point_component" in raw else np.zeros(len(raw["points"]), np.int32)
    of, ok_, op = raw["obs_frame"], raw["obs_kp"], raw["obs_point"]
    keep_obs = reg[of]
    of, ok_, op = of[keep_obs], ok_[keep_obs], op[keep_obs]
    good = (np.bincount(op, minlength=len(raw["points"])) >= 2) & np.isin(pc, list(transforms))
    keep_obs = good[op]
    of, ok_, op = of[keep_obs], ok_[keep_obs], op[keep_obs]
    new_id = -np.ones(len(raw["points"]), np.int64)
    new_id[good] = np.arange(good.sum())
    points = np.zeros((int(good.sum()), 3))
    for c, (scale, M, shift) in transforms.items():
        sel = good & (pc == c)
        points[new_id[sel]] = scale * (raw["points"][sel] @ M.T) + shift
    kps, koff = raw["kps"], raw["kp_offsets"]
    obs_xy = kps[koff[of] + ok_]
    scale0 = transforms[pieces[0]][0] if pieces[0] in transforms else 1.0
    info = {"version": GEOMETRY_VERSION, "fps": json.loads((work / "job.json").read_text()).get("fps", FPS),
            "frames": n, "posed": int(reg.sum()), "points": int(good.sum()),
            "scale": round(scale0, 4), "pieces": info_pieces, "walks": per_walk,
            "summary": json.loads((work / "summary.json").read_text()) if (work / "summary.json").is_file() else {}}
    log.info("geometry: %d of %d frames posed in the map frame (%d pieces), %d points", reg.sum(), n,
             len(transforms), good.sum())
    return Geometry(frames, reg, xyz, heading, Rm, raw["focal"].copy(), points, of.astype(np.int32),
                    obs_xy.astype(np.float32), new_id[op].astype(np.int32), raw["place"].astype(np.float32), info,
                    np.where(reg, comp, -1).astype(np.int32))


# --- in the build ----------------------------------------------------------------------------------------------
def write(geo: Geometry, build_dir: Path) -> None:
    """geometry.json (frames, poses, info) + geometry.npz (points, observations) in the build."""
    frames = []
    for i, f in enumerate(geo.frames):
        x, y, z = (None if np.isnan(v) else round(float(v), 3) for v in geo.xyz[i])
        frames.append({**asdict(f), "posed": bool(geo.posed[i]), "x_m": x, "y_m": y, "z_m": z,
                       "piece": None if geo.piece is None or geo.piece[i] < 0 else int(geo.piece[i]),
                       "heading_deg": None if np.isnan(geo.heading_deg[i]) else round(float(geo.heading_deg[i]), 1)})
    (build_dir / "geometry.json").write_text(json.dumps({"info": geo.info, "frames": frames}))
    np.savez_compressed(build_dir / "geometry.npz", R=geo.R.astype(np.float32), focal=geo.focal.astype(np.float32),
                        points=geo.points.astype(np.float32), obs_frame=geo.obs_frame, obs_xy=geo.obs_xy,
                        obs_point=geo.obs_point)
    np.save(build_dir / "geometry-place.npy", geo.place)


def load(build_dir: Path) -> Geometry | None:
    meta = build_dir / "geometry.json"
    if not meta.is_file():
        return None
    j = json.loads(meta.read_text())
    z = np.load(build_dir / "geometry.npz")
    frames = [GeoFrame(**{k: f[k] for k in ("name", "rec_id", "t_ms", "sharpness", "brightness")}) for f in j["frames"]]
    xyz = np.array([[np.nan if f[k] is None else f[k] for k in ("x_m", "y_m", "z_m")] for f in j["frames"]], float)
    heading = np.array([np.nan if f["heading_deg"] is None else f["heading_deg"] for f in j["frames"]], float)
    posed = np.array([f["posed"] for f in j["frames"]], bool)
    place = np.load(build_dir / "geometry-place.npy")
    piece = np.array([-1 if f.get("piece") is None else f["piece"] for f in j["frames"]], np.int32)
    return Geometry(frames, posed, xyz, heading, z["R"].astype(np.float64), z["focal"].astype(np.float64),
                    z["points"].astype(np.float64), z["obs_frame"], z["obs_xy"], z["obs_point"], place, j["info"],
                    piece)


def build_geometry(rec_dirs: list[Path], cache_root: Path, build_dir: Path, place_model: str, tracks: dict,
                   first_rec: str | None, warnings: list[str]) -> Geometry | None:
    """The build's geometry: the model made on the inference host (NAV_INFER_URL) or here, placed in the map frame
    and written into the build. None (with a warning) when it can't be made: the map falls back to dead reckoning."""
    from nav import config

    work = build_dir / "geometry-job"
    try:
        done = work / "gpu" / "done.json"
        before = json.loads((work / "job.json").read_text())["frames"] if done.is_file() else None
        if work.exists() and before is None:
            shutil.rmtree(work)
        work.mkdir(parents=True, exist_ok=True)
        frames = prepare_job(work, rec_dirs, cache_root, place_model)
        if before is not None and before != [asdict(f) for f in frames]:  # other frames: start again
            shutil.rmtree(work / "gpu")
        if len(frames) < 10:
            warnings.append("geometry: too few frames for a 3D model; positions from dead reckoning")
            return None
        if config.INFER_URL:
            from nav.mapping.inference import remote

            remote().geometry(work, f"{build_dir.parent.parent.name}-{build_dir.name}")
        else:
            run_job(work)
        geo = place_in_map(work, tracks, first_rec)
    except Exception as e:  # the map is still worth building from dead reckoning
        from nav.infer.errors import InferenceUnavailable

        if isinstance(e, InferenceUnavailable):  # the host or its GPU queue didn't get to it: build again later
            raise
        log.exception("geometry failed")
        warnings.append(f"geometry: no 3D model ({type(e).__name__}: {e}); positions from dead reckoning")
        return None
    finally:
        if work.exists():
            for sub in ("images", "gpu", "colmap"):
                shutil.rmtree(work / sub, ignore_errors=True)
    if geo is None:
        warnings.append("geometry: too few frames posed by the 3D model; positions from dead reckoning")
        return None
    write(geo, build_dir)
    shutil.rmtree(work, ignore_errors=True)
    for rec, w in geo.info["walks"].items():
        if w["frames"] and w["posed"] < 0.5 * w["frames"]:
            warnings.append(f"geometry: walk {rec} has {w['posed']} of {w['frames']} frames posed by the 3D model; "
                            f"the rest is placed by dead reckoning")
    return geo


# --- live poses (the inference host) -----------------------------------------------------------------------------
POSE_MATCH_PX = 1.0  # a matched model-frame keypoint is the one of its 3D observations this close
RANSAC_MAX_ERROR_PX = 12.0  # at LONG_SIDE


@dataclass
class PoseRef:
    """A memory frame ready to match live frames against: its ALIKED features (CPU, half precision) and the 3D points
    its keypoints see (pixel xy at LONG_SIDE, map-frame xyz)."""

    feats: dict
    obs_xy: np.ndarray  # [K, 2]
    obs_xyz: np.ndarray  # [K, 3]


def pose_ref(image: np.ndarray, obs_xy: np.ndarray, obs_xyz: np.ndarray) -> PoseRef:
    f = extract(fit_long_side(image))
    return PoseRef({k: v.cpu() for k, v in f.items()}, np.asarray(obs_xy, np.float32), np.asarray(obs_xyz, np.float64))


def fit_long_side(image: np.ndarray, long_side: int = LONG_SIDE) -> np.ndarray:
    """The image at long_side px (live frames come smaller or larger; ALIKED isn't scale-invariant)."""
    import cv2

    h, w = image.shape[:2]
    k = long_side / max(h, w)
    if abs(k - 1) < 1e-3:
        return image
    return cv2.resize(image, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC)


def solve_pose(query: np.ndarray, refs: list[PoseRef]) -> dict | None:
    """The camera pose of a live frame against memory frames: LightGlue with each, the matched keypoints' 3D points,
    PnP + RANSAC with the focal length estimated (a live phone's is unknown). -> {inliers, matches, R, t, centre,
    focal, per_ref} in the map frame, or None without enough correspondences."""
    import pycolmap
    from scipy.spatial import cKDTree

    from nav.mapping.models import _device

    img = fit_long_side(query)
    h, w = img.shape[:2]
    fq = extract(img)
    p2d, p3d, per_ref = [], [], []
    for r in refs:
        if not len(r.obs_xy):
            per_ref.append(0)
            continue
        fr = {k: v.to(_device()) for k, v in r.feats.items()}
        m = match(fq, fr)
        if not len(m):
            per_ref.append(0)
            continue
        kq = fq["keypoints"][0].cpu().numpy()[m[:, 0]]
        kr = fr["keypoints"][0].cpu().numpy()[m[:, 1]]
        d, j = cKDTree(r.obs_xy).query(kr, distance_upper_bound=POSE_MATCH_PX)
        ok = np.isfinite(d)
        p2d.append(kq[ok])
        p3d.append(r.obs_xyz[j[ok]])
        per_ref.append(int(ok.sum()))
    if not p2d or sum(len(a) for a in p2d) < 6:
        return None
    p2d = np.concatenate(p2d).astype(np.float64) + 0.5
    p3d = np.concatenate(p3d)
    cam = pycolmap.Camera(model="SIMPLE_PINHOLE", width=w, height=h, params=[1.2 * max(w, h), w / 2, h / 2])
    est = pycolmap.AbsolutePoseEstimationOptions()
    est.estimate_focal_length = True
    est.ransac.max_error = RANSAC_MAX_ERROR_PX
    ref = pycolmap.AbsolutePoseRefinementOptions()
    ref.refine_focal_length = True
    got = pycolmap.estimate_and_refine_absolute_pose(p2d, p3d, cam, est, ref)
    if got is None:
        return None
    cfw = got["cam_from_world"]
    R = np.asarray(cfw.rotation.matrix())
    t = np.asarray(cfw.translation, float).reshape(3)
    centre = -R.T @ t
    axis = R.T @ np.array([0.0, 0.0, 1.0])
    return {"inliers": int(got["num_inliers"]), "matches": int(len(p2d)), "centre": centre.tolist(),
            "heading_deg": float(np.degrees(np.arctan2(axis[0], axis[1])) % 360), "R": R.ravel().tolist(),
            "t": t.tolist(), "focal": float(cam.focal_length), "size": [w, h], "per_ref": per_ref}
