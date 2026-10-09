"""Frame sampling: the sharpest frame of every 1/fps seconds of video, with its sharpness and brightness.

Every frame is decoded (PyAV), scored on a small greyscale copy (variance of the Laplacian = focus / motion blur),
and only the winner of each time bucket is converted to a full-size RGB image.
"""

from dataclasses import dataclass
from pathlib import Path

import av
import cv2
import numpy as np

SCORE_WIDTH = 270  # px, greyscale copy used for scoring


@dataclass
class SampledFrame:
    t_ms: float  # recording clock
    sharpness: float  # variance of the Laplacian on the SCORE_WIDTH copy
    brightness: float  # mean grey level, 0-255
    image: np.ndarray  # RGB, long side <= long_side


def _score(frame: av.VideoFrame) -> tuple[float, float]:
    h = max(1, round(frame.height * SCORE_WIDTH / frame.width))
    grey = frame.to_ndarray(format="gray", width=SCORE_WIDTH, height=h)
    return float(cv2.Laplacian(grey, cv2.CV_64F).var()), float(grey.mean())


def sample_frames(video: Path, video_start_ms: float, fps: float = 1.0, long_side: int = 768) -> list[SampledFrame]:
    out: list[SampledFrame] = []
    best: tuple[int, float, float, av.VideoFrame] | None = None  # bucket, sharpness, brightness, frame

    def flush():
        b, sharp, bright, frame = best
        scale = min(1.0, long_side / max(frame.width, frame.height))
        w, h = round(frame.width * scale) // 2 * 2, round(frame.height * scale) // 2 * 2
        img = frame.to_ndarray(format="rgb24", width=w, height=h)
        out.append(SampledFrame(round(video_start_ms + float(frame.time) * 1000, 1), round(sharp, 1),
                                round(bright, 1), img))

    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            if frame.time is None:
                continue
            bucket = int(frame.time * fps)
            sharp, bright = _score(frame)
            if best is not None and bucket != best[0]:
                flush()
                best = None
            if best is None or sharp > best[1]:
                best = (bucket, sharp, bright, frame)
        if best is not None:
            flush()
    return out
