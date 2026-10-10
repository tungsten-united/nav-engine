"""Metric walked distance from the video: floor-plane visual odometry.

Steps x a step length is a poor ruler for mapping walks: handheld steps are hard to count (Itnig: 153 counted with
hand sway while talking, ~97 by the causal detector) and the recorder's step length is a default (0.7 m) that
doesn't fit a slow walk while filming (~0.35 m there). The camera sees the floor, though:

- floor features in the lower 40% of the frame are tracked (LK) over half a second,
- each pixel's ray is intersected with the floor using the camera tilt (gravity from the IMU), the calibrated field
  of view and the phone's height above the floor (assumed CAMERA_HEIGHT_M; a measured distance in the review
  fixes the scale, see review.py),
- a rotation + translation is fitted between the two metric point sets: the translation is the distance walked.
  Turning on the spot is a rotation about the camera's foot point, so it reads ~0 m.
- The fitted similarity's scale should be ~1 if the geometry holds; segments with too few inliers or a scale far
  from 1 are invalid, and the steps fill them in (`fuse`).

On 20261009T202101Z_7f42 this gave 6.95 m for Main door -> Drinks area, against 7 m measured on site (steps x 0.7 m
said 15.4 m); standing reads 0.01-0.06 m per half second, slow walking 0.2-0.3 m.

Ported from tungsten-united/demo-repository@navigation (mapping/src/mapper/motion.py, pdr.py) by LPA.
"""

import logging
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from nav.mapping.motion import MotionTrack

log = logging.getLogger(__name__)

ODOMETRY_VERSION = 1
CAMERA_HEIGHT_M = 1.35  # phone held in front of the chest while filming (Itnig, calibrated: 1.36 m)
ODO_FPS = 10.0  # frames per second tracked
ODO_WIDTH = 540  # px, width of the greyscale frames tracked
INTERVAL_S = 0.5  # distance is measured over this span
MIN_INLIERS = 15
SCALE_RANGE = (0.8, 1.25)
JITTER_M = 0.08  # per INTERVAL_S: below this it is hand jitter, not walking (standing: 0.01-0.06)

# Rear camera axes (OpenCV: x right, y down, z forward) from device axes (x right, y up the screen, z out of the
# screen), per holding. accelerationIncludingGravity points up, so this maps it to "up" in the camera frame.
CAMERA_FROM_DEVICE = {"handheld_portrait": np.diag([1.0, -1.0, -1.0])}


@dataclass
class FloorSegment:
    t0_ms: float  # sensor clock
    t1_ms: float
    distance_m: float
    scale: float
    inliers: int

    @property
    def valid(self) -> bool:
        return (self.inliers >= MIN_INLIERS and np.isfinite(self.distance_m)
                and SCALE_RANGE[0] <= self.scale <= SCALE_RANGE[1])

    def as_dict(self) -> dict:
        return asdict(self)


def intrinsics(width: int, height: int, fov_deg: float) -> np.ndarray:
    fx = (width / 2) / math.tan(math.radians(fov_deg / 2))
    return np.array([[fx, 0, width / 2], [0, fx, height / 2], [0, 0, 1.0]])


def to_floor(px: np.ndarray, up: np.ndarray, k_inv: np.ndarray, height_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Pixels -> 2D floor coordinates in metres, in a basis attached to the camera (e1 = camera x projected on the
    floor, e2 = up x e1). `up` is the unit up vector in camera coordinates. Returns (coords, mask of pixels whose
    ray goes below the horizon)."""
    rays = (k_inv @ np.c_[px, np.ones(len(px))].T).T
    down = rays @ up
    ok = down < -0.05
    points = rays[ok] * (-height_m / down[ok])[:, None]
    e1 = np.array([1.0, 0.0, 0.0]) - up * up[0]
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    return np.c_[points @ e1, points @ e2], ok


def floor_motion(frames: list[np.ndarray], up0: np.ndarray, up1: np.ndarray, k: np.ndarray, height_m: float
                 ) -> tuple[float, float, int]:
    """(distance_m, scale, inliers) of the camera over the floor from frames[0] to frames[-1] (greyscale), tracking
    through the frames in between."""
    h = frames[0].shape[0]
    mask = np.zeros_like(frames[0])
    mask[int(h * 0.6):] = 255
    p0 = cv2.goodFeaturesToTrack(frames[0], 400, 0.005, 7, mask=mask)
    if p0 is None or len(p0) < MIN_INLIERS:
        return float("nan"), float("nan"), 0
    p, ok = p0, np.ones(len(p0), bool)
    for a, b in zip(frames, frames[1:]):
        p, status, _err = cv2.calcOpticalFlowPyrLK(a, b, p, None, winSize=(21, 21), maxLevel=3)
        ok &= status.ravel() == 1
    a_px, b_px = p0[ok].reshape(-1, 2), p[ok].reshape(-1, 2)
    k_inv = np.linalg.inv(k)
    _, hit0 = to_floor(a_px, up0, k_inv, height_m)
    _, hit1 = to_floor(b_px, up1, k_inv, height_m)
    both = hit0 & hit1
    q0, _ = to_floor(a_px[both], up0, k_inv, height_m)
    q1, _ = to_floor(b_px[both], up1, k_inv, height_m)
    if len(q0) < 10:
        return float("nan"), float("nan"), int(len(q0))
    m, inliers = cv2.estimateAffinePartial2D(q0.astype(np.float32), q1.astype(np.float32), method=cv2.RANSAC,
                                             ransacReprojThreshold=0.05)
    if m is None:
        return float("nan"), float("nan"), 0
    return float(np.hypot(m[0, 2], m[1, 2])), float(np.hypot(m[0, 0], m[1, 0])), int(inliers.sum())


def floor_odometry(video: Path, video_start_ms: float, offset_ms: float, fov_deg: float, track: MotionTrack,
                   holding: str | None, height_m: float = CAMERA_HEIGHT_M) -> list[FloorSegment] | None:
    """Distance walked every INTERVAL_S from the floor seen by the camera; None when it can't be measured (no
    gravity from the IMU, or a holding without a known camera orientation). Times are on the sensor clock:
    a frame at video time v shows the sensors' v - offset_ms."""
    import av

    axes = CAMERA_FROM_DEVICE.get(holding or "")
    if axes is None or track.up is None or track.source != "imu":
        return None

    def up_at(t_ms: float) -> np.ndarray:
        u = axes @ track.up[track.index(t_ms)]
        return u / np.linalg.norm(u)

    # Each segment is measured as soon as its frames are decoded, keeping only the frames of one segment: the whole
    # walk at ODO_FPS would be ~5 MB per second of video, on the worker that also serves the API.
    stride = max(1, round(INTERVAL_S * ODO_FPS))
    chunk: list[tuple[float, np.ndarray]] = []  # the segment being filled; its first frame ends the previous one
    k, n_frames, next_t = None, 0, 0.0
    out = []
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            if frame.time is None or frame.time < next_t:
                continue
            next_t = frame.time + 1 / ODO_FPS
            h = max(2, round(frame.height * ODO_WIDTH / frame.width))
            grey = frame.to_ndarray(format="gray", width=ODO_WIDTH, height=h)
            if k is None:
                k = intrinsics(grey.shape[1], grey.shape[0], fov_deg)
            n_frames += 1
            chunk.append((video_start_ms + float(frame.time) * 1000 - offset_ms, grey))
            if len(chunk) == stride + 1:
                t0, t1 = chunk[0][0], chunk[-1][0]
                d, s, n = floor_motion([g for _t, g in chunk], up_at(t0), up_at(t1), k, height_m)
                out.append(FloorSegment(round(t0, 1), round(t1, 1), round(d, 4), round(s, 4), n))
                chunk = chunk[-1:]
    if n_frames < 2:
        return None
    return out


@dataclass
class Fused:
    track: MotionTrack
    video_share: float  # fraction of the walked metres measured by the floor odometry
    step_length_m: float | None  # video metres / steps where both measured the walking; None if too few steps


def fuse(track: MotionTrack, segments: list[FloorSegment] | None, step_length_m: float,
         min_steps: int = 20) -> Fused:
    """The track with walked_m and the dead-reckoned x/y from the floor odometry where it is valid, steps x step
    length elsewhere. The step length is re-estimated from the video (used where the floor can't be seen, and by
    the live tracker as its starting value)."""
    if not segments:
        return Fused(replace(track, distance_source="steps"), 0.0, None)
    steps = track.step_t_ms
    valid = [s for s in segments if s.valid]
    walking = [s for s in valid if track.moving[track.index((s.t0_ms + s.t1_ms) / 2)] and s.distance_m >= JITTER_M]
    n_steps = int(sum(((steps >= s.t0_ms) & (steps < s.t1_ms)).sum() for s in walking))
    est = (float(np.clip(sum(s.distance_m for s in walking) / n_steps, 0.25, 1.0)) if n_steps >= min_steps
           else None)
    step_len = est or step_length_m

    # Per-grid-sample increments: video where valid, else the steps' own increments rescaled to step_len.
    inc = np.diff(track.walked_m, prepend=0.0) * (step_len / step_length_m if step_length_m else 1.0)
    video_m = 0.0
    for s in segments:
        i0, i1 = track.index(s.t0_ms), track.index(s.t1_ms)
        if i1 <= i0 or not s.valid:
            continue
        moving = track.moving[track.index((s.t0_ms + s.t1_ms) / 2)]
        d = s.distance_m if s.distance_m >= JITTER_M or (moving and s.distance_m >= JITTER_M / 2) else 0.0
        inc[i0:i1] = d / (i1 - i0)
        video_m += d
    walked = np.cumsum(inc)
    heading = np.radians(np.nan_to_num(track.heading_deg, nan=0.0))
    has_heading = ~np.isnan(track.heading_deg)
    x = np.cumsum(inc * np.sin(heading) * has_heading)
    y = np.cumsum(inc * np.cos(heading) * has_heading)
    share = video_m / walked[-1] if walked[-1] > 0 else 0.0
    share = round(float(share), 3)
    return Fused(replace(track, walked_m=walked, x_m=x, y_m=y, distance_source="video" if share >= 0.5 else "steps",
                         step_length_m=round(step_len, 3), video_share=share), share, est)
