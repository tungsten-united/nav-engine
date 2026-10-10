"""The local models of the map pipeline, loaded once per process on the GPU when there is one.

- Frame embeddings: DINOv2 (NAV_EMBED_MODEL). The whole frame is resized, not centre-cropped, to a patch-size
  multiple keeping its aspect ratio; the embedding is the L2-normalised [CLS] token concatenated with the
  normalised mean patch token, normalised again. Cosine similarity = dot product. Used for novelty within a walk.
- Place embeddings: MegaLoc (NAV_PLACE_MODEL, torch.hub "owner/repo:commit"), a model trained to recognise the
  same place across viewpoint and lighting. Preprocessed as in its README (squashed to 322x322). Unit vectors.
- Local features: ALIKED keypoints + LightGlue matching; `count_inliers` is the number of matches consistent
  with one fundamental matrix, i.e. whether two frames really see the same structure.
- Voice transcription: Whisper (NAV_TRANSCRIBE_MODEL) through the transformers ASR pipeline; clips are decoded
  to 16 kHz mono with ffmpeg. `detect_language` gives a clip's language probabilities, so the caller can decide one
  language for a whole walk (a short clip on its own is often misdetected) and pass it to `transcribe`.
"""

import subprocess
from functools import cache
from pathlib import Path

import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
EMBED_SHORT_SIDE = 224


def _device():
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


@cache
def _embedder(name: str):
    import torch
    from transformers import AutoModel

    dtype = torch.float16 if _device() == "cuda" else torch.float32
    model = AutoModel.from_pretrained(name, dtype=dtype).to(_device()).eval()
    return model, dtype


def embed_images(images: list[np.ndarray], name: str, batch: int = 16) -> np.ndarray:
    """RGB uint8 images (same size) -> float32 [N, D] unit vectors."""
    import cv2
    import torch

    if not images:
        return np.zeros((0, 0), np.float32)
    model, dtype = _embedder(name)
    patch = getattr(model.config, "patch_size", 14)
    h, w = images[0].shape[:2]
    scale = EMBED_SHORT_SIDE / min(h, w)
    size = (max(patch, round(w * scale / patch) * patch), max(patch, round(h * scale / patch) * patch))
    out = []
    for i in range(0, len(images), batch):
        x = np.stack([cv2.resize(im, size, interpolation=cv2.INTER_AREA) for im in images[i:i + batch]])
        x = (x.astype(np.float32) / 255 - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(x).permute(0, 3, 1, 2).to(_device(), dtype)
        with torch.inference_mode():
            tokens = model(pixel_values=x).last_hidden_state.float()
        cls = torch.nn.functional.normalize(tokens[:, 0], dim=-1)
        mean = torch.nn.functional.normalize(tokens[:, 1:].mean(dim=1), dim=-1)
        out.append(torch.nn.functional.normalize(torch.cat([cls, mean], dim=-1), dim=-1).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


PLACE_SIZE = 322


@cache
def _place_model(name: str):
    import torch

    repo, _, ref = name.partition(":")
    # A commit ref isn't a branch or tag, which hub's fork check would reject.
    model = torch.hub.load(f"{repo}:{ref}" if ref else repo, "get_trained_model", trust_repo=True,
                           skip_validation=bool(ref))
    return model.to(_device()).eval()


def embed_places(images: list[np.ndarray], name: str, batch: int = 16) -> np.ndarray:
    """RGB uint8 images -> float32 [N, D] unit vectors (MegaLoc; fp32, its optimal-transport head dislikes fp16)."""
    import cv2
    import torch

    if not images:
        return np.zeros((0, 0), np.float32)
    model = _place_model(name)
    out = []
    for i in range(0, len(images), batch):
        x = np.stack([cv2.resize(im, (PLACE_SIZE, PLACE_SIZE), interpolation=cv2.INTER_AREA) for im in images[i:i + batch]])
        x = (x.astype(np.float32) / 255 - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(x).permute(0, 3, 1, 2).to(_device())
        with torch.inference_mode():
            out.append(torch.nn.functional.normalize(model(x).float(), dim=-1).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


@cache
def _local():
    from lightglue import ALIKED, LightGlue

    return (ALIKED(max_num_keypoints=1024).to(_device()).eval(),
            LightGlue(features="aliked").to(_device()).eval())


def local_features(image: np.ndarray) -> dict:
    """ALIKED keypoints of an RGB uint8 image, at its own size (frames are stored at 768 px already)."""
    import torch

    x = torch.tensor(image).permute(2, 0, 1).float().div(255).to(_device())  # copy: PIL arrays are read-only
    return _local()[0].extract(x, resize=None)


def count_inliers(f0: dict, f1: dict) -> int:
    """LightGlue matches between two local_features() results that fit one fundamental matrix (MAGSAC)."""
    import cv2
    import torch

    with torch.inference_mode():
        m = _local()[1]({"image0": f0, "image1": f1})["matches"][0].cpu().numpy()
    if len(m) < 8:
        return 0
    p0 = f0["keypoints"][0].cpu().numpy()[m[:, 0]]
    p1 = f1["keypoints"][0].cpu().numpy()[m[:, 1]]
    _, mask = cv2.findFundamentalMat(p0, p1, cv2.USAC_MAGSAC, 1.5, 0.999, 10000)
    return int(mask.sum()) if mask is not None else 0


@cache
def _asr(name: str):
    import torch
    from transformers import pipeline

    dtype = torch.float16 if _device() == "cuda" else torch.float32
    return pipeline("automatic-speech-recognition", model=name, dtype=dtype, device=_device())


def decode_audio(path: Path, rate: int = 16000) -> np.ndarray:
    pcm = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(rate),
                          "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(pcm, dtype=np.float32)


def detect_language(path: Path, name: str) -> dict[str, float]:
    """Whisper's language probabilities for a clip ({"en": 0.93, ...}) from its first 30 s; {} for a silent stub."""
    import torch

    audio = decode_audio(path)[:30 * 16000]
    if len(audio) < 1600:  # < 0.1 s
        return {}
    pipe = _asr(name)
    model, gen = pipe.model, pipe.model.generation_config
    feats = pipe.feature_extractor(audio, sampling_rate=16000, return_tensors="pt").input_features
    start = torch.tensor([[gen.decoder_start_token_id]], device=model.device)
    with torch.inference_mode():
        logits = model(input_features=feats.to(model.device, model.dtype), decoder_input_ids=start).logits[0, -1]
    codes = list(gen.lang_to_id)
    p = torch.softmax(logits[list(gen.lang_to_id.values())].float(), -1).cpu().numpy()
    return {c.strip("<|>"): float(x) for c, x in zip(codes, p)}


def transcribe(path: Path, name: str, language: str | None = None) -> str:
    """The clip's text; in `language` (a Whisper code such as "en") when given, else Whisper detects it."""
    audio = decode_audio(path)
    if len(audio) < 1600:  # < 0.1 s
        return ""
    kw = {"task": "transcribe"} | ({"language": language} if language else {})
    result = _asr(name)({"raw": audio, "sampling_rate": 16000}, return_timestamps=len(audio) > 30 * 16000,
                        generate_kwargs=kw)
    return " ".join(result["text"].split())


def check(embed_model: str, place_model: str, transcribe_model: str | None) -> dict:
    """Download (into the Hugging Face / torch hub caches) and load the models, and run them once on dummy input."""
    import torch

    info = {"device": _device(), "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    dummy = np.random.default_rng(0).integers(0, 256, (768, 432, 3), dtype=np.uint8)
    emb = embed_images([dummy] * 2, embed_model)
    info["embed_model"], info["embed_dim"] = embed_model, int(emb.shape[1])
    emb = embed_places([dummy] * 2, place_model)
    info["place_model"], info["place_dim"] = place_model, int(emb.shape[1])
    f = local_features(dummy)
    info["local_features"] = f"aliked+lightglue, {count_inliers(f, f)} inliers on a self-match"
    if transcribe_model:
        _asr(transcribe_model)({"raw": np.zeros(16000, np.float32), "sampling_rate": 16000})
        info["transcribe_model"] = transcribe_model
    return info
