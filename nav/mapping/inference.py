"""The model calls of the map pipeline, on files: local models, or the inference host when NAV_INFER_URL is set.

Images and audio go as the files on disk (JPEG frames, voice clips) and are decoded the same way on both sides,
so a remote call gives what the local one would. Callers keep their caches keyed by model name either way.
"""

import io
from collections.abc import Sequence
from functools import cache
from pathlib import Path

import numpy as np
from PIL import Image

from nav import config

ImageSource = str | Path | bytes


def load_rgb(src: ImageSource) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(src) if isinstance(src, bytes) else src).convert("RGB"))


@cache
def _client(url: str):
    from nav.infer.client import InferenceClient

    return InferenceClient.from_config()


def remote():
    return _client(config.INFER_URL) if config.INFER_URL else None


def embed_files(paths: Sequence[Path | str], model: str) -> np.ndarray:
    """DINOv2 frame embeddings, [N, D] unit vectors."""
    if client := remote():
        return client.embed([Path(p) for p in paths], model)
    from nav.mapping.models import embed_images

    return embed_images([load_rgb(p) for p in paths], model)


def place_files(paths: Sequence[Path | str], model: str) -> np.ndarray:
    """MegaLoc place embeddings, [N, D] unit vectors."""
    if client := remote():
        return client.place([Path(p) for p in paths], model)
    from nav.mapping.models import embed_places

    return embed_places([load_rgb(p) for p in paths], model)


def place_live(images: Sequence[bytes], model: str, deadline_s: float) -> np.ndarray:
    """MegaLoc place embeddings of live camera frames (JPEG bytes), for nav-api."""
    if client := remote():
        return client.place_live(list(images), model, deadline_s)
    from nav.mapping.models import embed_places

    return embed_places([load_rgb(b) for b in images], model)


def local_pair_inliers(images: Sequence[ImageSource], pairs: Sequence[tuple[int, int]]) -> list[int]:
    """ALIKED + LightGlue inliers for each (i, j) pair of images; each image's features are extracted once."""
    from nav.mapping.models import count_inliers, local_features

    feats: dict[int, dict] = {}
    out = []
    for i, j in pairs:
        for k in (i, j):
            if k not in feats:
                feats[k] = local_features(load_rgb(images[k]))
        out.append(count_inliers(feats[i], feats[j]))
    return out


def pair_inliers(paths: Sequence[Path | str], pairs: Sequence[tuple[int, int]]) -> list[int]:
    if client := remote():
        return client.match([Path(p) for p in paths], list(pairs))
    return local_pair_inliers(paths, pairs)


def transcribe_file(path: Path, model: str) -> str:
    if client := remote():
        return client.transcribe(path, model)
    from nav.mapping.models import transcribe

    return transcribe(path, model)
