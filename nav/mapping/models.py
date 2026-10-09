"""The two local models of the map pipeline, loaded once per process on the GPU when there is one.

- Frame embeddings: DINOv2 (NAV_EMBED_MODEL). The whole frame is resized, not centre-cropped, to a patch-size
  multiple keeping its aspect ratio; the embedding is the L2-normalised [CLS] token concatenated with the
  normalised mean patch token, normalised again. Cosine similarity = dot product.
- Voice transcription: Whisper (NAV_TRANSCRIBE_MODEL) through the transformers ASR pipeline; clips are decoded
  to 16 kHz mono with ffmpeg. The language is detected per clip.
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


def transcribe(path: Path, name: str) -> str:
    audio = decode_audio(path)
    if len(audio) < 1600:  # < 0.1 s
        return ""
    result = _asr(name)({"raw": audio, "sampling_rate": 16000}, return_timestamps=len(audio) > 30 * 16000,
                        generate_kwargs={"task": "transcribe"})
    return " ".join(result["text"].split())


def check(embed_model: str, transcribe_model: str | None) -> dict:
    """Download (into the Hugging Face cache) and load both models, and run them once on dummy input."""
    import torch

    info = {"device": _device(), "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    emb = embed_images([np.zeros((768, 432, 3), np.uint8)] * 2, embed_model)
    info["embed_model"], info["embed_dim"] = embed_model, int(emb.shape[1])
    if transcribe_model:
        _asr(transcribe_model)({"raw": np.zeros(16000, np.float32), "sampling_rate": 16000})
        info["transcribe_model"] = transcribe_model
    return info
