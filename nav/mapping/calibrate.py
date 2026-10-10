"""Camera yaw from the video itself, and the calibration it allows: the sensor-to-video time offset and the camera's
horizontal field of view.

Turning moves the whole image sideways, walking forward mostly doesn't (radial flow), so the median horizontal shift
between frames a tenth of a second apart, in image widths, is the yaw in units of the field of view. Cross-correlating
its rate with the sensor heading's rate gives:
- the offset between the two clocks: video frames are stamped `video_start + frame time`, which can be late by
  the encoder's start latency (+0.45 s on 20261009T202101Z_7f42), and
- the field of view: degrees of sensor yaw per image width of video yaw over the strongest turns (36 degrees there).
Both are applied only when the correlation is clear. Without sensors (an imported video), the video yaw times a
default field of view is the heading.

Ported from tungsten-united/demo-repository@c5dc26f (navigation, mapping/src/mapper/motion.py) by LPA.
"""

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

CALIB_VERSION = 2  # 2: gravity turned up for iOS (motion.py): the heading it calibrates against turns the right way
YAW_FPS = 10.0  # frames per second compared for the video yaw
YAW_WIDTH = 360  # px, width of the greyscale frames compared
DEFAULT_FOV_DEG = {"handheld_portrait": 40.0}  # 16:9 sensor held in portrait
FALLBACK_FOV_DEG = 60.0
MAX_LAG_MS = 1500
MIN_CORRELATION = 0.5
FOV_RANGE = (20.0, 120.0)


def default_fov(holding: str | None) -> float:
    return DEFAULT_FOV_DEG.get(holding or "", FALLBACK_FOV_DEG)


@dataclass
class Calibration:
    offset_ms: float  # subtract from video times to get sensor times (0 when not applied)
    fov_deg: float  # horizontal field of view used (calibrated when applied, else the default)
    measured_offset_ms: float | None = None  # what the cross-correlation found, applied or not
    measured_fov_deg: float | None = None
    correlation: float | None = None
    applied: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


_orb = None


def horizontal_shift(prev: np.ndarray, cur: np.ndarray) -> float:
    """Horizontal translation in pixels of `cur` relative to `prev` (greyscale): ORB matches + a RANSAC similarity,
    else phase correlation."""
    global _orb
    if _orb is None:
        _orb = cv2.ORB_create(nfeatures=800)
    kp1, des1 = _orb.detectAndCompute(prev, None)
    kp2, des2 = _orb.detectAndCompute(cur, None)
    if des1 is not None and des2 is not None and len(kp1) >= 20 and len(kp2) >= 20:
        matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(des1, des2)
        if len(matches) >= 15:
            src = np.float32([kp1[m.queryIdx].pt for m in matches])
            dst = np.float32([kp2[m.trainIdx].pt for m in matches])
            matrix, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
            if matrix is not None and inliers is not None and inliers.sum() >= 12:
                return float(matrix[0, 2])
    (dx, _dy), _response = cv2.phaseCorrelate(np.float32(prev), np.float32(cur))
    return float(dx)


def video_yaw(video: Path, video_start_ms: float, fps: float = YAW_FPS, width: int = YAW_WIDTH
              ) -> tuple[np.ndarray, np.ndarray]:
    """(t_ms on the recording clock, cumulative clockwise yaw in image widths) at about `fps`."""
    import av

    times, yaw = [], []
    prev, total, next_t = None, 0.0, 0.0
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            if frame.time is None or frame.time < next_t:
                continue
            next_t = frame.time + 1 / fps
            h = max(2, round(frame.height * width / frame.width))
            grey = frame.to_ndarray(format="gray", width=width, height=h)
            if prev is not None:
                total += -horizontal_shift(prev, grey) / width  # turning right moves the scene left
            times.append(video_start_ms + float(frame.time) * 1000)
            yaw.append(total)
            prev = grey
    return np.array(times), np.array(yaw)


def estimate(t_sensor_ms: np.ndarray, heading_deg: np.ndarray, t_video_ms: np.ndarray, yaw_units: np.ndarray,
             max_lag_ms: float = MAX_LAG_MS) -> tuple[float, float, float] | None:
    """(offset_ms, fov_deg, correlation): the lag that best aligns the sensor's yaw rate with the video's (a sensor
    sample at time s shows in the video at s + offset), and degrees of sensor yaw per image width over the
    strongest rotations. `heading_deg` may be wrapped; NaNs are dropped."""
    ok = ~np.isnan(heading_deg)
    if ok.sum() < 20 or len(t_video_ms) < 20:
        return None
    ts, hs = t_sensor_ms[ok], np.degrees(np.unwrap(np.radians(heading_deg[ok])))
    sensor_rate = np.gradient(hs, ts / 1000)
    k = max(1, round(150 / float(np.median(np.diff(ts)))))  # 0.15 s smoothing
    sensor_rate = np.convolve(sensor_rate, np.ones(k) / k, mode="same")
    video_rate = np.gradient(yaw_units, t_video_ms / 1000)
    if np.std(video_rate) == 0:
        return None
    best = (0.0, -1.0, None)
    for lag in np.arange(-max_lag_ms, max_lag_ms + 1e-9, 50.0):
        shifted = np.interp(t_video_ms, ts + lag, sensor_rate)
        if np.std(shifted) == 0:
            continue
        c = float(np.corrcoef(video_rate, shifted)[0, 1])
        if c > best[1]:
            best = (float(lag), c, shifted)
    lag, corr, shifted = best
    if shifted is None:
        return None
    strong = np.abs(video_rate) > np.percentile(np.abs(video_rate), 85)
    strong &= np.sign(video_rate) == np.sign(shifted)
    fov = float(np.median(shifted[strong] / video_rate[strong])) if strong.any() else float("nan")
    return lag, fov, corr


def calibrate(t_sensor_ms: np.ndarray, heading_deg: np.ndarray, t_video_ms: np.ndarray, yaw_units: np.ndarray,
              holding: str | None) -> Calibration:
    fov0 = default_fov(holding)
    found = estimate(t_sensor_ms, heading_deg, t_video_ms, yaw_units)
    if found is None:
        return Calibration(offset_ms=0, fov_deg=fov0)
    lag, fov, corr = found
    applied = (corr >= MIN_CORRELATION and abs(lag) < MAX_LAG_MS and np.isfinite(fov)
               and FOV_RANGE[0] <= fov <= FOV_RANGE[1])
    log.info("sensor/video calibration: offset %+.0f ms, FOV %.0f°, corr %.2f%s", lag, fov, corr,
             "" if applied else " (not applied)")
    return Calibration(offset_ms=round(lag) if applied else 0, fov_deg=round(fov, 1) if applied else fov0,
                       measured_offset_ms=round(lag), measured_fov_deg=round(fov, 1) if np.isfinite(fov) else None,
                       correlation=round(corr, 3), applied=bool(applied))
