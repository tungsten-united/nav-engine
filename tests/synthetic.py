"""Synthetic test data: a video that pans over a textured panorama the way a turning camera would, and the
accelerometer of a walk. Adapted from tungsten-united/demo-repository@c5dc26f (mapping/tests/make_synthetic_recording.py)."""

from pathlib import Path

import cv2
import numpy as np

W, H = 240, 400  # portrait frames


def panorama(width: int = 2400, height: int = H, seed: int = 0) -> np.ndarray:
    """Greyscale texture with plenty of corners for ORB: random rectangles and circles on noise."""
    rng = np.random.default_rng(seed)
    img = (rng.random((height, width)) * 40 + 100).astype(np.uint8)
    for _ in range(width // 6):
        x, y = int(rng.integers(0, width)), int(rng.integers(0, height))
        c = int(rng.integers(0, 255))
        if rng.random() < 0.5:
            cv2.rectangle(img, (x, y), (x + int(rng.integers(6, 40)), y + int(rng.integers(6, 40))), c, -1)
        else:
            cv2.circle(img, (x, y), int(rng.integers(4, 20)), c, -1)
    return cv2.GaussianBlur(img, (3, 3), 0)


def turn_right(t_s: float, at: float = 5.0, over: float = 2.0, deg: float = 90.0) -> float:
    """Heading (clockwise degrees from the start) of a walk that turns right `deg` between `at` and `at + over`."""
    return deg * min(max((t_s - at) / over, 0.0), 1.0)


def make_video(path: Path, yaw_deg=turn_right, seconds: float = 12.0, fps: float = 15.0, fov_deg: float = 40.0
               ) -> None:
    """An mp4 whose camera turns by yaw_deg(t): turning right moves the scene left by W / fov_deg px per degree."""
    import av

    pano = panorama()
    px_per_deg = W / fov_deg
    with av.open(str(path), "w") as out:
        stream = out.add_stream("libx264", rate=int(fps))
        stream.width, stream.height, stream.pix_fmt = W, H, "yuv420p"
        for i in range(int(seconds * fps)):
            x = int(round(200 + yaw_deg(i / fps) * px_per_deg))
            grey = np.ascontiguousarray(pano[:, x:x + W])
            rgb = np.repeat(grey[:, :, None], 3, axis=2)
            for packet in stream.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)


def walking_accel(seconds: float = 20.0, cadence: float = 1.8, rate: float = 60.0, amp: float = 1.2,
                  seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """(t_s, [N, 3] accelerationIncludingGravity) of a phone held upright, walking from 1 s to seconds - 1."""
    t = np.arange(0, seconds, 1 / rate)
    walking = (t > 1) & (t < seconds - 1)
    noise = np.random.default_rng(seed).normal(0, 0.03, (len(t), 3))
    acc = noise.copy()
    acc[:, 1] += 9.81 + amp * np.sin(2 * np.pi * cadence * t) * walking
    return t, acc
