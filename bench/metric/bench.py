"""Metric localization check: against a 3D model of the place, does PnP pose held-out and live frames correctly where
today's localizer (embeddings + LightGlue votes) doesn't?

    uv run --group bench-metric python bench/metric/bench.py itnig              # held-out walks, with truth
    uv run --group bench-metric python bench/metric/bench.py nati-house --live  # captured live calls, no truth

Per place, cached under data/bench-metric/<map_id>/ (gitignored: the frames show people and homes), each stage
once:
1. frames: every walk's video at FPS, the sharpest frame of each 1/FPS s, long side LONG_SIDE, on the sensor clock
   (load_walk's calibration offset), like the map's 1 fps cache but denser and larger;
2. ALIKED features (MAX_KEYPOINTS) and MegaLoc embeddings of every frame;
3. pairs: the next SEQ_WINDOW frames of the same walk + the RETRIEVAL_K most similar frames anywhere (MegaLoc), and
   LightGlue matches for each pair;
4. a COLMAP model (pycolmap) per set of walks: one camera per walk (SIMPLE_RADIAL; focal length from the walk's
   calibrated field of view, as a prior), matches geometrically verified, then incremental (or global) mapping;
5. the model in the map's frame and metres: a similarity from its camera centres to the walks' placed tracks (the
   review's scale) at the same times.

Localizing a frame: the QUERY_K most similar frames of the model (MegaLoc), LightGlue with each, the 3D points of the
matched keypoints, then PnP + RANSAC (pycolmap; the focal length is estimated, as a live phone's is unknown).
Posed: at least POSED inliers (the tables also give 20 and 80).

Itnig (--holdout, the default with 2+ walks): each walk is localized against a model built without it, one frame a
second, degraded the way live frames are (640 px, blur, darker, JPEG 50). Truth: the same frames registered in the
model of every walk (the usual pseudo ground truth), the held-out model aligned to it through their shared frames.
The current localizer (localize.py with the memory, and with verify; the map without the walk) gets the same
degraded frames; its `position` is a memory frame, scored by that frame's pose in the same model.

Nati House (--live): the walk's model against every frame of every captured live call. No truth: how many are posed,
whether consecutive poses move as far as the phone's steps say, and overlays to check by eye.

GPU stages wait until the GPU has MIN_FREE_GB free (it is shared) and release it afterwards; frames and mapping run
on the CPU.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import pickle
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1]))
from nav import config  # noqa: E402
from nav.schemas.map import PlaceMap  # noqa: E402

log = logging.getLogger("metric")

FPS = 2.0  # model frames per second of video
LONG_SIDE = 1024  # px, model frames; queries are resized to it too (ALIKED isn't scale-invariant)
MIN_BRIGHTNESS = 20  # mean grey level; darker frames are left out
MAX_KEYPOINTS = 2048
SEQ_WINDOW = 5  # frames ahead in the same walk (2.5 s at 2 fps)
RETRIEVAL_K = 12
QUERY_K = 10
RANSAC_MAX_ERROR = 12.0  # px at LONG_SIDE
POSED = 40
POSED_LEVELS = (20, 40, 80)
LIVE_LONG_SIDE = 640  # what Orient's live frames are (480 x 640 at Nati House)
TRACK_FIT_MAX_ERROR_M = 1.5  # similarity model -> placed tracks: dead reckoning drifts
MODEL_FIT_MAX_ERROR_M = 0.3  # held-out model -> all-walks model: shared frames
TRUTH_MAX_DT_MS = 600  # a memory frame's pose: the model frame of its walk at most this far in time
JUMP_SLACK_M = 2.0  # live: consecutive poses further apart than the steps walked + this are implausible
MIN_FREE_GB = 6.0
OUT = config.DATA_DIR / "bench-metric"


# --- GPU sharing -----------------------------------------------------------------------------------------------
def wait_gpu(min_free_gb: float = MIN_FREE_GB, every_s: int = 120) -> None:
    import torch

    if not torch.cuda.is_available():
        return
    while True:
        free = torch.cuda.mem_get_info()[0] / 2**30
        if free >= min_free_gb:
            log.info("GPU: %.1f GB free", free)
            return
        log.info("GPU: %.1f GB free, waiting for %.1f", free, min_free_gb)
        time.sleep(every_s)


def release_gpu() -> None:
    import torch

    from nav.mapping import models

    _aliked.cache_clear()
    models._local.cache_clear()
    models._place_model.cache_clear()
    _gpu_feats.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --- frames ----------------------------------------------------------------------------------------------------
@dataclass
class Img:
    name: str  # "<rec>/<t_ms>.jpg" under images/
    rec: str
    t_ms: float  # sensor clock, as the map's tracks and memory
    walked_m: float  # since the walk started (the walk's own dead reckoning / floor odometry)


@dataclass
class WalkInfo:
    rec: str
    fov_deg: float  # horizontal, calibrated (or the default)
    width: int
    height: int


def sample_walks(place: Path, recs: list[str]) -> tuple[list[Img], dict[str, WalkInfo]]:
    from PIL import Image

    from nav.mapping.frames import sample_frames
    from nav.mapping.prepare import _video, load_walk
    from nav.mapping.select import SelectParams

    index = place / "frames.json"
    known = json.loads(index.read_text()) if index.is_file() else {"images": [], "walks": {}}
    for rec in recs:
        if rec in known["walks"]:
            continue
        rec_dir = config.RECORDINGS_DIR / rec
        walk = load_walk(rec_dir, config.CACHE_DIR, SelectParams())
        start = next((e.t_ms for e in walk.events if e.type == "video_start"), 0.0)
        out = place / "images" / rec
        out.mkdir(parents=True, exist_ok=True)
        log.info("%s: frames at %.0f fps", rec, FPS)
        size = None
        for sf in sample_frames(_video(rec_dir, walk.manifest), start, FPS, LONG_SIDE):
            if sf.brightness < MIN_BRIGHTNESS:
                continue
            t = round(sf.t_ms - walk.calibration.offset_ms, 1)
            name = f"{rec}/{round(t):07d}.jpg"
            Image.fromarray(sf.image).save(place / "images" / name, quality=92)
            size = sf.image.shape[:2]
            known["images"].append(asdict(Img(name, rec, t, float(walk.track.at(t)["walked_m"]))))
        known["walks"][rec] = asdict(WalkInfo(rec, walk.calibration.fov_deg, size[1], size[0]))
        index.write_text(json.dumps(known))
    imgs = [Img(**r) for r in known["images"] if r["rec"] in recs]
    return imgs, {r: WalkInfo(**known["walks"][r]) for r in recs}


def load_rgb(path: Path, long_side: int | None = None) -> np.ndarray:
    import cv2
    from PIL import Image

    img = np.asarray(Image.open(path).convert("RGB"))
    if long_side:
        h, w = img.shape[:2]
        k = long_side / max(h, w)
        if abs(k - 1) > 1e-3:
            img = cv2.resize(img, (round(w * k), round(h * k)),
                             interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC)
    return img


def degrade(img: np.ndarray) -> bytes:
    """A model frame the way a live frame comes: 640 px, a little blur, darker, JPEG 50."""
    import cv2

    h, w = img.shape[:2]
    k = LIVE_LONG_SIDE / max(h, w)
    small = cv2.resize(img, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 1.2)
    small = np.clip(small.astype(np.float32) * 0.75, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(small, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 50])
    return buf.tobytes()


# --- features, embeddings, matches -----------------------------------------------------------------------------
@cache
def _aliked():
    from lightglue import ALIKED

    from nav.mapping.models import _device

    return ALIKED(max_num_keypoints=MAX_KEYPOINTS).to(_device()).eval()


def extract(rgb: np.ndarray) -> dict[str, np.ndarray]:
    import torch

    from nav.mapping.models import _device

    x = torch.tensor(rgb).permute(2, 0, 1).float().div(255).to(_device())
    with torch.inference_mode():
        f = _aliked().extract(x, resize=None)
    return {"keypoints": f["keypoints"][0].cpu().numpy().astype(np.float32),
            "descriptors": f["descriptors"][0].cpu().numpy().astype(np.float16),
            "image_size": f["image_size"][0].cpu().numpy().astype(np.float32)}


_gpu_feats: dict[str, dict] = {}


def _on_device(f: dict) -> dict:
    import torch

    from nav.mapping.models import _device

    d = _device()
    return {"keypoints": torch.from_numpy(f["keypoints"])[None].to(d),
            "descriptors": torch.from_numpy(f["descriptors"].astype(np.float32))[None].to(d),
            "image_size": torch.from_numpy(f["image_size"])[None].to(d)}


def match(f0: dict, f1: dict) -> np.ndarray:
    """LightGlue (the live verifier's: models._local) between two features already on the device -> [N, 2]."""
    import torch

    from nav.mapping.models import _local

    with torch.inference_mode():
        m = _local()[1]({"image0": f0, "image1": f1})["matches"][0]
    return m.cpu().numpy().astype(np.uint32)


def features(place: Path, imgs: list[Img]) -> dict[str, dict]:
    path = place / "features.pkl"
    feats = pickle.loads(path.read_bytes()) if path.is_file() else {}
    todo = [i for i in imgs if i.name not in feats]
    if todo:
        wait_gpu()
        log.info("ALIKED on %d frames", len(todo))
        for k, im in enumerate(todo):
            feats[im.name] = extract(load_rgb(place / "images" / im.name))
            if k % 200 == 199:
                path.write_bytes(pickle.dumps(feats))
        path.write_bytes(pickle.dumps(feats))
    return feats


def place_embeddings(place: Path, imgs: list[Img], model: str) -> dict[str, np.ndarray]:
    from nav.mapping.models import embed_places

    path = place / "place.pkl"
    emb = pickle.loads(path.read_bytes()) if path.is_file() else {}
    todo = [i for i in imgs if i.name not in emb]
    if todo:
        wait_gpu()
        log.info("MegaLoc on %d frames", len(todo))
        for s in range(0, len(todo), 64):
            chunk = todo[s:s + 64]
            for im, v in zip(chunk, embed_places([load_rgb(place / "images" / i.name) for i in chunk], model)):
                emb[im.name] = v
        path.write_bytes(pickle.dumps(emb))
    return emb


def make_pairs(imgs: list[Img], emb: dict[str, np.ndarray]) -> list[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    by_rec: dict[str, list[Img]] = {}
    for im in imgs:
        by_rec.setdefault(im.rec, []).append(im)
    for walk in by_rec.values():
        walk.sort(key=lambda i: i.t_ms)
        for a in range(len(walk)):
            for b in range(a + 1, min(a + 1 + SEQ_WINDOW, len(walk))):
                pairs.add((walk[a].name, walk[b].name))
    v = np.stack([emb[i.name] for i in imgs])
    sim = v @ v.T
    near_s = SEQ_WINDOW / FPS * 1000
    for a, im in enumerate(imgs):
        n = 0
        for b in np.argsort(-sim[a]):
            o = imgs[b]
            if b == a or (o.rec == im.rec and abs(o.t_ms - im.t_ms) <= near_s):
                continue
            pairs.add(tuple(sorted((im.name, o.name))))
            n += 1
            if n >= RETRIEVAL_K:
                break
    return sorted(pairs)


def matches(place: Path, pairs: list[tuple[str, str]], feats: dict[str, dict]) -> dict[tuple[str, str], np.ndarray]:
    path = place / "matches.pkl"
    got = pickle.loads(path.read_bytes()) if path.is_file() else {}
    todo = [p for p in pairs if p not in got]
    if todo:
        wait_gpu()
        log.info("LightGlue on %d pairs", len(todo))
        t = time.time()
        for k, (a, b) in enumerate(todo):
            for n in (a, b):
                if n not in _gpu_feats:
                    _gpu_feats[n] = _on_device(feats[n])
            got[(a, b)] = match(_gpu_feats[a], _gpu_feats[b])
            if k % 2000 == 1999:
                path.write_bytes(pickle.dumps(got))
                log.info("  %d/%d pairs, %.0f ms each", k + 1, len(todo), (time.time() - t) / (k + 1) * 1000)
        path.write_bytes(pickle.dumps(got))
    return got


# --- models ----------------------------------------------------------------------------------------------------
def build_model(place: Path, name: str, imgs: list[Img], walks: dict[str, WalkInfo], feats: dict[str, dict],
                pair_matches: dict[tuple[str, str], np.ndarray], mapper: str):
    """The largest COLMAP model of these frames (cached in models/<name>/model), and a summary of every component."""
    import pycolmap

    d = place / "models" / name
    out = d / "model"
    if (out / "images.bin").is_file():
        return pycolmap.Reconstruction(out), json.loads((d / "summary.json").read_text())
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    names = {i.name for i in imgs}
    db_path = d / "database.db"
    opts = pycolmap.ImageReaderOptions()
    opts.camera_model = "SIMPLE_RADIAL"
    pycolmap.Database.open(db_path).close()  # import_images wants the file
    pycolmap.import_images(db_path, place / "images", pycolmap.CameraMode.PER_FOLDER, sorted(names), opts)
    db = pycolmap.Database.open(db_path)
    images = db.read_all_images()
    ids = {im.name: im.image_id for im in images}
    cam_rec = {im.camera_id: im.name.split("/")[0] for im in images}
    for cam in db.read_all_cameras():
        w = walks[cam_rec[cam.camera_id]]
        f = 0.5 * cam.width / math.tan(math.radians(w.fov_deg) / 2)
        cam.params = [f, cam.width / 2, cam.height / 2, 0.0]
        cam.has_prior_focal_length = True
        db.update_camera(cam)
    for n, i in ids.items():
        db.write_keypoints(i, feats[n]["keypoints"].astype(np.float32) + 0.5)  # COLMAP: pixel centres at +0.5
    lines = []
    for (a, b), m in pair_matches.items():
        if a in names and b in names and len(m):
            db.write_matches(ids[a], ids[b], m)
            lines.append(f"{a} {b}")
    db.close()
    (d / "pairs.txt").write_text("\n".join(lines) + "\n")
    t = time.time()
    pycolmap.verify_matches(db_path, d / "pairs.txt", pycolmap.TwoViewGeometryOptions())
    log.info("%s: %d images, %d pairs verified in %.0f s, mapping (%s)", name, len(ids), len(lines),
             time.time() - t, mapper)
    t = time.time()
    if mapper == "global":
        recs = pycolmap.global_mapping(db_path, place / "images", d / "sparse")
    else:
        o = pycolmap.IncrementalPipelineOptions()
        o.num_threads = -1
        recs = pycolmap.incremental_mapping(db_path, place / "images", d / "sparse", o)
    comps = sorted(recs.values(), key=lambda r: -r.num_reg_images())
    summary = {"images": len(ids), "pairs": len(lines), "mapper": mapper, "mapping_s": round(time.time() - t),
               "components": []}
    for r in comps:
        per = {}
        for iid in r.reg_image_ids():
            rec = r.images[iid].name.split("/")[0]
            per[rec] = per.get(rec, 0) + 1
        summary["components"].append({"registered": r.num_reg_images(), "points": r.num_points3D(), "walks": per})
    if not comps:
        raise RuntimeError(f"{name}: no model")
    out.mkdir()
    comps[0].write(out)
    (d / "summary.json").write_text(json.dumps(summary, indent=1))
    log.info("%s: %s", name, json.dumps(summary["components"][:3]))
    return comps[0], summary


def centres(model) -> dict[str, np.ndarray]:
    return {model.images[i].name: np.asarray(model.images[i].projection_center(), float).reshape(3)
            for i in model.reg_image_ids()}


def to_map_frame(model, pmap: PlaceMap, by_name: dict[str, Img]) -> dict:
    """Move the model into the map's frame and metres: a similarity from its camera centres to the walks' placed
    tracks (published scale) at the same times. Returns the fit's numbers."""
    import pycolmap

    k = pmap.frame.distance_scale if pmap.frame else 1.0
    tracks = {t.rec_id: np.array([p[:3] for p in t.points], float) for t in pmap.tracks}
    names, src, tgt = [], [], []
    for n, c in centres(model).items():
        im = by_name[n]
        tr = tracks.get(im.rec)
        if tr is None:
            continue
        names.append(n)
        src.append(c)
        tgt.append([np.interp(im.t_ms, tr[:, 0], tr[:, 1]) * k, np.interp(im.t_ms, tr[:, 0], tr[:, 2]) * k, 0.0])
    ro = pycolmap.RANSACOptions()
    ro.max_error = TRACK_FIT_MAX_ERROR_M
    sim = pycolmap.estimate_sim3d_robust(np.array(src), np.array(tgt), ro)
    if sim is None:
        log.warning("model -> map frame: no robust similarity, fitting all %d frames", len(src))
        sim = pycolmap.estimate_sim3d(np.array(src), np.array(tgt))
    if sim is None:
        raise RuntimeError("model -> map frame: no similarity")
    model.transform(sim)
    after = centres(model)
    res = np.array([np.hypot(*(after[n][:2] - np.array(t[:2]))) for n, t in zip(names, tgt)])
    return {"frames": len(names), "scale": round(float(sim.scale), 4),
            "track_vs_model_median_m": round(float(np.median(res)), 2),
            "track_vs_model_p90_m": round(float(np.percentile(res, 90)), 2)}


def align_to(model, ref) -> dict:
    """Move `model` onto `ref` (already in the map frame) through the frames registered in both."""
    import pycolmap

    a, b = centres(model), centres(ref)
    common = sorted(set(a) & set(b))
    ro = pycolmap.RANSACOptions()
    ro.max_error = MODEL_FIT_MAX_ERROR_M
    sim = pycolmap.estimate_sim3d_robust(np.array([a[n] for n in common]), np.array([b[n] for n in common]), ro)
    if sim is None:
        raise RuntimeError("no similarity between the models")
    model.transform(sim)
    after = centres(model)
    res = np.array([np.linalg.norm(after[n] - b[n]) for n in common])
    return {"common": len(common), "median_m": round(float(np.median(res)), 3),
            "p90_m": round(float(np.percentile(res, 90)), 3)}


# --- localization ----------------------------------------------------------------------------------------------
@dataclass
class Pose:
    inliers: int
    matches: int  # 2D-3D correspondences tried
    xyz: np.ndarray | None  # camera centre, map frame
    R: np.ndarray | None  # cam_from_world rotation
    t: np.ndarray | None
    focal: float | None
    top: list[str]
    ms: dict[str, float]
    p2d: np.ndarray | None = None  # inlier correspondences, for overlays
    p3d: np.ndarray | None = None


class Localizer:
    def __init__(self, model, feats: dict[str, dict], emb: dict[str, np.ndarray], place_model: str):
        self.model = model
        self.names = [model.images[i].name for i in model.reg_image_ids()]
        self.emb = np.stack([emb[n] for n in self.names])
        self.feats = feats
        self.place_model = place_model
        self.p3d_ids = {}
        for i in model.reg_image_ids():
            im = model.images[i]
            self.p3d_ids[im.name] = np.array([p.point3D_id if p.has_point3D() else -1 for p in im.points2D],
                                             np.int64)
        self.xyz = {pid: np.asarray(p.xyz, float).reshape(3) for pid, p in model.points3D.items()}

    def __call__(self, rgb: np.ndarray) -> Pose:
        import pycolmap

        from nav.mapping.models import embed_places

        ms = {}
        t = time.perf_counter()
        fq = extract(rgb)
        ms["extract"] = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        e = embed_places([rgb], self.place_model)[0]
        top = [self.names[j] for j in np.argsort(-(self.emb @ e))[:QUERY_K]]
        ms["retrieve"] = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        q = _on_device(fq)
        pairs: set[tuple[int, int]] = set()
        for n in top:
            if n not in _gpu_feats:
                _gpu_feats[n] = _on_device(self.feats[n])
            m = match(q, _gpu_feats[n])
            ids = self.p3d_ids[n][m[:, 1]] if len(m) else np.zeros(0, np.int64)
            pairs.update((int(a), int(p)) for a, p in zip(m[:, 0], ids) if p >= 0)
        ms["match"] = (time.perf_counter() - t) * 1000
        h, w = rgb.shape[:2]
        if len(pairs) < 6:
            return Pose(0, len(pairs), None, None, None, None, top, ms)
        t = time.perf_counter()
        pl = sorted(pairs)
        p2d = fq["keypoints"][[a for a, _ in pl]].astype(np.float64) + 0.5
        p3d = np.array([self.xyz[p] for _, p in pl])
        cam = pycolmap.Camera(model="SIMPLE_PINHOLE", width=w, height=h, params=[1.2 * max(w, h), w / 2, h / 2])
        est = pycolmap.AbsolutePoseEstimationOptions()
        est.estimate_focal_length = True
        est.ransac.max_error = RANSAC_MAX_ERROR
        ref = pycolmap.AbsolutePoseRefinementOptions()
        ref.refine_focal_length = True
        r = pycolmap.estimate_and_refine_absolute_pose(p2d, p3d, cam, est, ref)
        ms["pnp"] = (time.perf_counter() - t) * 1000
        if r is None:
            return Pose(0, len(pl), None, None, None, None, top, ms)
        cfw = r["cam_from_world"]
        R, tr = np.asarray(cfw.rotation.matrix()), np.asarray(cfw.translation, float).reshape(3)
        mask = np.asarray(r["inlier_mask"], bool)
        return Pose(int(r["num_inliers"]), len(pl), -R.T @ tr, R, tr, float(cam.focal_length), top, ms,
                    p2d[mask], p3d[mask])


def rot_err_deg(R0: np.ndarray, R1: np.ndarray) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R0 @ R1.T) - 1) / 2))))


def stats(xs: list[float]) -> dict:
    a = np.array(xs, float)
    if not len(a):
        return {"n": 0}
    return {"n": len(a), "median": round(float(np.median(a)), 2), "p90": round(float(np.percentile(a, 90)), 2)}


def pct(a: int, b: int) -> str:
    return f"{100 * a / b:.0f}%" if b else "-"


# --- Itnig: held-out walks -------------------------------------------------------------------------------------
def bench_holdout(map_id: str, pmap: PlaceMap, map_dir: Path, place: Path, extra: list[str], mapper: str,
                  every_s: float, baseline: bool) -> dict:
    from nav.mapping.inference import place_files
    from nav.navigation.locbench import Query, region, run_sequence, score
    from nav.navigation.localize import VerifyParams
    from nav.navigation.replay import ref_indexes

    walks_map = list(pmap.recordings)
    recs = walks_map + [r for r in extra if r not in walks_map]
    imgs, walks = sample_walks(place, recs)
    by_name = {i.name: i for i in imgs}
    feats = features(place, imgs)
    model_name = pmap.memory.model if pmap.memory else pmap.embedding.model
    emb = place_embeddings(place, imgs, model_name)
    pair_list = make_pairs(imgs, emb)
    pm = matches(place, pair_list, feats)
    release_gpu()

    out: dict = {"map": map_id, "walks": recs, "frames": len(imgs), "pairs": len(pair_list), "mapper": mapper}
    full, out["model_all"] = build_model(place, "all", imgs, walks, feats, pm, mapper)
    out["map_fit"] = to_map_frame(full, pmap, by_name)
    truth_c = centres(full)
    truth_R = {full.images[i].name: full.images[i].cam_from_world().rotation.matrix() for i in full.reg_image_ids()}
    by_rec_t: dict[str, list[tuple[float, str]]] = {}
    for n in truth_c:
        by_rec_t.setdefault(by_name[n].rec, []).append((by_name[n].t_ms, n))

    def model_pos(rec: str, t_ms: float) -> np.ndarray | None:
        """Where the all-walks model puts a frame of `rec` near t_ms (a memory frame's pose)."""
        best = min(by_rec_t.get(rec, []), key=lambda x: abs(x[0] - t_ms), default=None)
        return truth_c[best[1]] if best and abs(best[0] - t_ms) <= TRUTH_MAX_DT_MS else None

    held = {}
    indexes = ref_indexes(pmap, map_dir, holdout=True, memory=True) if baseline else None
    index_all = ref_indexes(pmap, map_dir, holdout=False, memory=True) if baseline else None
    thresholds = config.loc_thresholds()
    for w in recs:
        others = [r for r in walks_map if r != w]
        model, summ = build_model(place, f"without-{w}", [i for i in imgs if i.rec in others], walks, feats, pm,
                                  mapper)
        fit = align_to(model, full)
        res = {"model": summ, "fit_to_all": fit, "in_map": w in walks_map}
        qdir = place / "queries" / w
        qdir.mkdir(parents=True, exist_ok=True)
        qs, last = [], -1e18
        for im in sorted((i for i in imgs if i.rec == w), key=lambda i: i.t_ms):
            if im.t_ms - last < every_s * 1000 - 1:
                continue
            last = im.t_ms
            p = qdir / Path(im.name).name
            if not p.is_file():
                p.write_bytes(degrade(load_rgb(place / "images" / im.name)))
            qs.append((im, p))
        t0 = qs[0][0].t_ms if qs else 0
        wait_gpu()
        loc = Localizer(model, feats, emb, model_name)
        rows = []
        for im, p in qs:
            pose = loc(load_rgb(p, LONG_SIDE))
            row = {"t_s": round((im.t_ms - t0) / 1000, 1), "inliers": pose.inliers, "matches": pose.matches,
                   "focal": pose.focal, "ms": pose.ms, "truth": im.name in truth_c}
            if pose.xyz is not None:
                row["xy"] = [round(float(pose.xyz[0]), 2), round(float(pose.xyz[1]), 2)]
                row["node"] = region(pmap, *row["xy"])
            if im.name in truth_c:
                tc = truth_c[im.name]
                row["truth_xy"] = [round(float(tc[0]), 2), round(float(tc[1]), 2)]
                row["truth_node"] = region(pmap, *row["truth_xy"])
                if pose.xyz is not None:
                    row["err_m"] = round(float(np.hypot(*(pose.xyz[:2] - tc[:2]))), 2)
                    row["rot_deg"] = round(rot_err_deg(pose.R, truth_R[im.name]), 1)
            rows.append(row)
        res["pnp"] = rows
        if baseline:
            vecs = place_files([p for _, p in qs], model_name)
            index = indexes[w] if w in walks_map else index_all
            queries = []
            for (im, p), v in zip(qs, vecs):
                tc = truth_c.get(im.name)
                x, y = (float(tc[0]), float(tc[1])) if tc is not None else (None, None)
                queries.append(Query(w, im.t_ms, v[None], p, im.walked_m, x, y, region(pmap, x, y)))
            res["baseline"] = {}
            for method, vp in (("memory", None), ("verify", VerifyParams())):
                seq = run_sequence(lambda q: index, queries, thresholds, map_dir, vp)
                sc = score(method, pmap, [seq])
                same_model = []
                for r, (im, _) in zip(seq, qs):
                    pos = r.loc.position
                    tc = truth_c.get(im.name)
                    mp = model_pos(pos.rec_id, pos.t_ms) if pos else None
                    if tc is not None and mp is not None:
                        same_model.append(float(np.hypot(*(mp[:2] - tc[:2]))))
                res["baseline"][method] = {
                    "line": sc.line(), "located_s": sc.located_s[0], "commits": sc.commits,
                    "commits_right": sc.commits_right, "commits_known": sc.commits_known,
                    "commit_err": stats(sc.commit_err), "node_right": sc.node_right, "with_truth": sc.with_truth,
                    "pos_err_map": stats(sc.pos_err), "pos_err_model": stats(same_model),
                    "best": [r.loc.best for r in seq], "status": [r.loc.status for r in seq]}
        held[w] = res
        release_gpu()
        log.info("%s: %s", w, summarize_walk(w, res))
    out["held_out"] = held
    return out


def summarize_walk(w: str, res: dict) -> str:
    rows = res["pnp"]
    n = len(rows)
    posed = [r for r in rows if r["inliers"] >= POSED]
    errs = [r["err_m"] for r in posed if "err_m" in r]
    rots = [r["rot_deg"] for r in posed if "rot_deg" in r]
    good = sum(1 for r in posed if r.get("err_m", 99) <= 1 and r.get("rot_deg", 99) <= 10)
    known = sum(1 for r in rows if r["truth"])
    return (f"{n} queries ({known} with truth): posed {pct(len(posed), n)}; within 1 m/10° {pct(good, known)} of "
            f"those with truth; error {stats(errs)}, rotation {stats(rots)}")


# --- Nati House: live calls ------------------------------------------------------------------------------------
def bench_live(map_id: str, pmap: PlaceMap, map_dir: Path, place: Path, mapper: str, since: str | None,
               baseline: bool, overlays: int) -> dict:
    from nav.navigation.captured import sessions
    from nav.navigation.locbench import bench_sessions, region
    from nav.navigation.localize import VerifyParams

    recs = list(pmap.recordings)
    imgs, walks = sample_walks(place, recs)
    by_name = {i.name: i for i in imgs}
    feats = features(place, imgs)
    model_name = pmap.memory.model if pmap.memory else pmap.embedding.model
    emb = place_embeddings(place, imgs, model_name)
    pair_list = make_pairs(imgs, emb)
    pm = matches(place, pair_list, feats)
    release_gpu()
    model, summ = build_model(place, "all", imgs, walks, feats, pm, mapper)
    out: dict = {"map": map_id, "walks": recs, "frames": len(imgs), "pairs": len(pair_list), "model": summ,
                 "map_fit": to_map_frame(model, pmap, by_name)}

    found = sessions(config.LIVE_DIR, map_id, since)
    step = (pmap.frame.live_step_length_m or pmap.frame.step_length_m) if pmap.frame else 0.7
    wait_gpu()
    loc = Localizer(model, feats, emb, model_name)
    sess_out, shown = [], 0
    odir = place / "overlays"
    odir.mkdir(exist_ok=True)
    for s in found:
        calls = []
        for c in s.localizations():
            frames = []
            for f in c.frames:
                rgb = load_rgb(f)
                pose = loc(load_rgb(f, LONG_SIDE))
                fr = {"frame": f.name, "size": [rgb.shape[1], rgb.shape[0]], "inliers": pose.inliers,
                      "matches": pose.matches, "focal": pose.focal, "ms": pose.ms, "top": pose.top[:3]}
                if pose.xyz is not None:
                    fr["xy"] = [round(float(pose.xyz[0]), 2), round(float(pose.xyz[1]), 2)]
                    fr["node"] = region(pmap, *fr["xy"])
                frames.append(fr)
                if shown < overlays and pose.inliers >= POSED and len(calls) % 6 == 0:
                    draw_overlay(odir / f"{s.session[:8]}-{f.stem}.jpg", f, place / "images" / pose.top[0], pose)
                    shown += 1
            resp = c.response or {}
            calls.append({"t_utc": c.t_utc.isoformat(), "steps": c.step_count, "frames": frames,
                          "live": {k: resp.get(k) for k in ("status", "best", "score", "margin")}})
        sess_out.append({"session": s.session, "date": s.date, "calls": calls})
    out["sessions"] = sess_out
    out["step_length_m"] = step
    release_gpu()
    if baseline:
        scores = bench_sessions(pmap, map_dir, found, config.loc_thresholds(), ["memory", "verify"], VerifyParams())
        out["baseline"] = {sc.method: {"line": sc.line(), "located_s": sc.located_s, "commits": sc.commits}
                           for sc in scores}
        release_gpu()
    return out


def draw_overlay(path: Path, query: Path, db_image: Path, pose: Pose) -> None:
    """The live frame with the inliers (green) and their 3D points reprojected (red), next to the best model frame."""
    import cv2

    q = load_rgb(query, LONG_SIDE).copy()
    h, w = q.shape[:2]
    pc = pose.p3d @ pose.R.T + pose.t
    uv = pose.focal * pc[:, :2] / pc[:, 2:] + np.array([w / 2, h / 2])
    for (x, y), (u, v) in zip(pose.p2d - 0.5, uv - 0.5):
        cv2.circle(q, (int(x), int(y)), 4, (0, 255, 0), 1)
        cv2.circle(q, (int(u), int(v)), 2, (255, 0, 0), -1)
    cv2.putText(q, f"{pose.inliers} inliers  f={pose.focal:.0f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (255, 255, 0), 2)
    d = load_rgb(db_image)
    d = cv2.resize(d, (round(d.shape[1] * h / d.shape[0]), h))
    cv2.imwrite(str(path), cv2.cvtColor(np.hstack([q, d]), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])


# --- report ----------------------------------------------------------------------------------------------------
def report_holdout(r: dict) -> str:
    lines = [f"## {r['map']}: held-out walks ({r['frames']} model frames, {r['pairs']} pairs, {r['mapper']} mapping)",
             "", f"All-walks model: {json.dumps(r['model_all']['components'][:3])}",
             f"Model -> map frame: {json.dumps(r['map_fit'])}", ""]
    lines.append("| walk | queries | truth | " + " | ".join(f"posed ≥{k}" for k in POSED_LEVELS)
                 + " | ≤1 m & ≤10° | error median / p90 | rotation median / p90 | first fix |")
    lines.append("|---" * (6 + len(POSED_LEVELS)) + "|")
    tot = {"n": 0, "known": 0, "good": 0, **{k: 0 for k in POSED_LEVELS}}
    all_err, all_rot = [], []
    for w, res in r["held_out"].items():
        rows = res["pnp"]
        known = sum(1 for x in rows if x["truth"])
        posed = [x for x in rows if x["inliers"] >= POSED]
        good = sum(1 for x in posed if x.get("err_m", 99) <= 1 and x.get("rot_deg", 99) <= 10)
        errs = [x["err_m"] for x in posed if "err_m" in x]
        rots = [x["rot_deg"] for x in posed if "rot_deg" in x]
        all_err += errs
        all_rot += rots
        first = next((x["t_s"] for x in rows if x["inliers"] >= POSED), None)
        e, ro = stats(errs), stats(rots)
        tot["n"] += len(rows)
        tot["known"] += known
        tot["good"] += good
        cells = []
        for k in POSED_LEVELS:
            c = sum(1 for x in rows if x["inliers"] >= k)
            tot[k] += c
            cells.append(pct(c, len(rows)))
        mark = "" if res["in_map"] else " (not in the map)"
        lines.append(f"| {w}{mark} | {len(rows)} | {known} | " + " | ".join(cells) + f" | {pct(good, known)} | "
                     f"{e.get('median', '-')} / {e.get('p90', '-')} m | {ro.get('median', '-')} / "
                     f"{ro.get('p90', '-')}° | {first if first is not None else '-'} s |")
    e, ro = stats(all_err), stats(all_rot)
    lines.append(f"| **all** | {tot['n']} | {tot['known']} | "
                 + " | ".join(pct(tot[k], tot['n']) for k in POSED_LEVELS)
                 + f" | **{pct(tot['good'], tot['known'])}** | {e.get('median', '-')} / {e.get('p90', '-')} m | "
                 f"{ro.get('median', '-')} / {ro.get('p90', '-')}° | |")
    if any("baseline" in res for res in r["held_out"].values()):
        lines += ["", "Current localizer on the same degraded frames, and PnP's node (the region its pose is in):", "",
                  "| walk | method | node right | position error, map frame (median / p90) | its memory frame's "
                  "pose vs truth (median / p90) | belief located | commits right |",
                  "|---|---|---|---|---|---|---|"]
        for w, res in r["held_out"].items():
            rows = res["pnp"]
            nk = [x for x in rows if x.get("truth_node")]
            pn = sum(1 for x in nk if x["inliers"] >= POSED and x.get("node") == x["truth_node"])
            lines.append(f"| {w} | PnP (≥{POSED}) | {pct(pn, len(nk))} | - | - | - | - |")
            for m, b in res.get("baseline", {}).items():
                lines.append(f"| {w} | {m} | {pct(b['node_right'], b['with_truth'])} | "
                             f"{b['pos_err_map'].get('median', '-')} / {b['pos_err_map'].get('p90', '-')} m | "
                             f"{b['pos_err_model'].get('median', '-')} / {b['pos_err_model'].get('p90', '-')} m | "
                             f"{b['located_s'] if b['located_s'] is not None else 'never'} | "
                             f"{b['commits_right']}/{b['commits_known']} |")
    ms = [x["ms"] for res in r["held_out"].values() for x in res["pnp"]]
    lines += ["", latency(ms)]
    return "\n".join(lines)


def latency(ms: list[dict]) -> str:
    keys = ["extract", "retrieve", "match", "pnp"]
    med = {k: float(np.median([m[k] for m in ms if k in m])) for k in keys if any(k in m for m in ms)}
    tot = float(np.median([sum(m.values()) for m in ms])) if ms else 0
    return "Latency per frame (median): " + ", ".join(f"{k} {v:.0f} ms" for k, v in med.items()) + \
        f"; total {tot:.0f} ms"


def report_live(r: dict) -> str:
    step = r["step_length_m"]
    lines = [f"## {r['map']}: captured live calls against the walk's model ({r['frames']} model frames, "
             f"{r['pairs']} pairs)", "", f"Model: {json.dumps(r['model']['components'][:3])}",
             f"Model -> map frame: {json.dumps(r['map_fit'])}", "",
             "| session | calls | frames | " + " | ".join(f"posed ≥{k}" for k in POSED_LEVELS)
             + f" | calls posed (≥{POSED}) | implausible jumps | live answers then |", "|---" * (7 + len(POSED_LEVELS))
             + "|"]
    ms, focals = [], []
    for s in r["sessions"]:
        frames = [f for c in s["calls"] for f in c["frames"]]
        ms += [f["ms"] for f in frames]
        focals += [f["focal"] for f in frames if f["inliers"] >= POSED and f["focal"]]  # px at LONG_SIDE
        cells = [pct(sum(1 for f in frames if f["inliers"] >= k), len(frames)) for k in POSED_LEVELS]
        best = []  # per call: the frame with most inliers, if posed
        for c in s["calls"]:
            b = max(c["frames"], key=lambda f: f["inliers"], default=None)
            if b and b["inliers"] >= POSED:
                best.append((c["steps"], b["xy"]))
        jumps = bad = 0
        for (s0, p0), (s1, p1) in zip(best, best[1:]):
            if s0 is None or s1 is None:
                continue
            jumps += 1
            walked = max(s1 - s0, 0) * step
            bad += math.hypot(p1[0] - p0[0], p1[1] - p0[1]) > walked + JUMP_SLACK_M
        live = {}
        for c in s["calls"]:
            live[c["live"]["status"]] = live.get(c["live"]["status"], 0) + 1
        lines.append(f"| {s['session'][:8]} | {len(s['calls'])} | {len(frames)} | " + " | ".join(cells)
                     + f" | {pct(len(best), len(s['calls']))} | {bad}/{jumps} | "
                     + ", ".join(f"{k} {v}" for k, v in sorted(live.items(), key=lambda kv: str(kv[0]))) + " |")
    if focals:
        lines += ["", f"Estimated focal length of posed live frames: median {np.median(focals):.0f} px at {LONG_SIDE} "
                  f"px (p10 {np.percentile(focals, 10):.0f}, p90 {np.percentile(focals, 90):.0f})."]
    for m, b in r.get("baseline", {}).items():
        lines.append(f"\nCurrent localizer, {m}: `{b['line']}`")
    lines += ["", latency(ms)]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("map_id")
    ap.add_argument("--live", action="store_true", help="Captured live calls instead of held-out walks.")
    ap.add_argument("--since", help="Live sessions from this day (YYYY-MM-DD).")
    ap.add_argument("--extra", default="", help="Comma-separated walks of the place not in the map, also held out.")
    ap.add_argument("--mapper", choices=["incremental", "global"], default="incremental")
    ap.add_argument("--every-s", type=float, default=1.0, help="Held-out query spacing.")
    ap.add_argument("--no-baseline", action="store_true", help="Skip the current localizer.")
    ap.add_argument("--overlays", type=int, default=12)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")
    map_dir = config.MAPS_DIR / a.map_id
    pmap = PlaceMap.model_validate_json((map_dir / "map.json").read_text())
    place = OUT / a.map_id
    place.mkdir(parents=True, exist_ok=True)
    if a.live:
        r = bench_live(a.map_id, pmap, map_dir, place, a.mapper, a.since, not a.no_baseline, a.overlays)
        text = report_live(r)
    else:
        extra = [x for x in a.extra.split(",") if x]
        r = bench_holdout(a.map_id, pmap, map_dir, place, extra, a.mapper, a.every_s, not a.no_baseline)
        text = report_holdout(r)
    kind = "live" if a.live else "holdout"
    (place / f"results-{kind}.json").write_text(json.dumps(r, indent=1, default=str))
    (HERE / f"results-{a.map_id}-{kind}.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
