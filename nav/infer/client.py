"""Client of the inference host (nav/infer/server.py), used by a worker without GPU (NAV_INFER_URL).

Requests go through Cloudflare Access with the service token (CF_ACCESS_CLIENT_ID / CF_ACCESS_CLIENT_SECRET), never a
login. They are cut to the server's caps, and retried with backoff while the host is busy (503 + Retry-After) or
unreachable (Cloudflare 502/504/52x, connection errors), up to a deadline per call; then InferenceUnavailable.
A Cloudflare login redirect means the token was rejected: that fails at once.
"""

import base64
import json
import logging
import os
import random
import time
from pathlib import Path

import httpx
import numpy as np

from nav import config
from nav.infer.errors import InferenceError, InferenceUnavailable
from nav.infer.server import MAX_IMAGES, MAX_MATCH_IMAGES, MAX_PAIRS

log = logging.getLogger(__name__)

RETRY_STATUS = {429, 502, 503, 504, 520, 521, 522, 523, 524}
AUDIO_MIME = {".webm": "audio/webm", ".m4a": "audio/mp4", ".ogg": "audio/ogg", ".mp3": "audio/mpeg", ".wav": "audio/wav"}


def match_chunks(pairs: list[tuple[int, int]], max_pairs: int = MAX_PAIRS, max_images: int = MAX_MATCH_IMAGES
                 ) -> list[list[int]]:
    """Positions into `pairs`, grouped so each group has at most max_pairs pairs over at most max_images images."""
    chunks: list[list[int]] = []
    cur: list[int] = []
    imgs: set[int] = set()
    for k, (i, j) in enumerate(pairs):
        if cur and (len(cur) >= max_pairs or len(imgs | {i, j}) > max_images):
            chunks.append(cur)
            cur, imgs = [], set()
        cur.append(k)
        imgs |= {i, j}
    return chunks + [cur] if cur else chunks


class InferenceClient:
    def __init__(self, base_url: str, headers: dict[str, str] | None = None, deadline_s: float = 600,
                 transport: httpx.BaseTransport | None = None, sleep=time.sleep):
        self.base_url = base_url.rstrip("/")
        self.deadline_s = deadline_s
        self._sleep = sleep
        self.http = httpx.Client(base_url=self.base_url, headers=headers or {}, follow_redirects=False,
                                 timeout=httpx.Timeout(95, connect=10), transport=transport)

    @classmethod
    def from_config(cls) -> "InferenceClient":
        headers = {}
        if os.environ.get("CF_ACCESS_CLIENT_ID"):
            headers = {"CF-Access-Client-Id": os.environ["CF_ACCESS_CLIENT_ID"],
                       "CF-Access-Client-Secret": os.environ.get("CF_ACCESS_CLIENT_SECRET", "")}
        return cls(config.INFER_URL, headers)

    # --- transport ----------------------------------------------------------------------------------------------
    def _send(self, method: str, path: str, deadline_s: float | None = None, **kw) -> httpx.Response:
        end = time.monotonic() + (deadline_s or self.deadline_s)
        attempt, last = 0, ""
        while True:
            wait = None
            left = max(1.0, end - time.monotonic())  # no single request outlives the deadline
            try:
                r = self.http.request(method, path, timeout=httpx.Timeout(min(95.0, left), connect=min(10.0, left)),
                                      **kw)
            except httpx.TransportError as e:  # connect/read errors and timeouts: the host may be down or restarting
                last = f"{type(e).__name__}: {e}"
            else:
                if r.is_redirect or r.status_code in (401, 403):
                    raise InferenceUnavailable(f"{self.base_url} answered {r.status_code}: the Cloudflare Access "
                                               "service token (CF_ACCESS_CLIENT_ID/SECRET) is missing or rejected")
                if r.status_code < 400:
                    return r
                if r.status_code not in RETRY_STATUS:
                    raise InferenceError(f"{method} {path}: {r.status_code} {r.text[:500]}")
                last = f"{r.status_code} {r.text[:200]}"
                if ra := r.headers.get("retry-after", "").strip():
                    wait = float(ra) if ra.isdigit() else None
            attempt += 1
            wait = min(60.0, wait if wait is not None else min(30.0, 2.0 ** attempt) * (0.5 + random.random() / 2))
            if time.monotonic() + wait > end:
                raise InferenceUnavailable(f"inference host unreachable or busy ({method} {path}, "
                                           f"{attempt} attempts): {last}")
            log.info("inference host: %s; retrying %s %s in %.0f s", last, method, path, wait)
            self._sleep(wait)

    def health(self, deadline_s: float = 30) -> dict:
        return self._send("GET", "/infer/v1/health", deadline_s=deadline_s).json()

    # --- models ---------------------------------------------------------------------------------------------------
    def _vectors(self, endpoint: str, images: list[tuple[str, bytes]], model: str, live: bool = False,
                 deadline_s: float | None = None) -> np.ndarray:
        out = []
        data = {"model": model, **({"priority": "live"} if live else {})}
        for i in range(0, len(images), MAX_IMAGES):
            files = [("images", (name, content, "image/jpeg")) for name, content in images[i:i + MAX_IMAGES]]
            j = self._send("POST", f"/infer/v1/{endpoint}", deadline_s=deadline_s, data=data, files=files).json()
            if j["model"] != model or j["n"] != len(files):
                raise InferenceError(f"{endpoint}: asked {len(files)} x {model}, got {j['n']} x {j['model']}")
            out.append(np.frombuffer(base64.b64decode(j["data"]), dtype="<f4").reshape(j["n"], j["dim"]))
        return np.concatenate(out).astype(np.float32) if out else np.zeros((0, 0), np.float32)

    def embed(self, paths: list[Path], model: str) -> np.ndarray:
        return self._vectors("embed", [(p.name, p.read_bytes()) for p in paths], model)

    def place(self, paths: list[Path], model: str) -> np.ndarray:
        return self._vectors("place", [(p.name, p.read_bytes()) for p in paths], model)

    def place_live(self, images: list[bytes], model: str, deadline_s: float) -> np.ndarray:
        """Place embeddings of live camera frames (JPEG bytes) for nav-api: ahead of queued map-build calls on the
        host, and given up after `deadline_s` (InferenceUnavailable) rather than keeping a walking user waiting."""
        return self._vectors("place", [(f"live{i}.jpg", b) for i, b in enumerate(images)], model, live=True,
                             deadline_s=deadline_s)

    def match(self, paths: list[Path], pairs: list[tuple[int, int]]) -> list[int]:
        out = [0] * len(pairs)
        for chunk in match_chunks(pairs):
            local = sorted({k for pos in chunk for k in pairs[pos]})
            index = {k: n for n, k in enumerate(local)}
            files = [("images", (Path(paths[k]).name, Path(paths[k]).read_bytes(), "image/jpeg")) for k in local]
            sub = [[index[pairs[pos][0]], index[pairs[pos][1]]] for pos in chunk]
            got = self._send("POST", "/infer/v1/match", data={"pairs": json.dumps(sub)}, files=files).json()["inliers"]
            for pos, k in zip(chunk, got):
                out[pos] = int(k)
        return out

    @staticmethod
    def _audio(path: Path) -> dict:
        return {"audio": (path.name, path.read_bytes(), AUDIO_MIME.get(path.suffix.lower(), "application/octet-stream"))}

    def language(self, path: Path, model: str) -> dict[str, float]:
        return self._send("POST", "/infer/v1/language", data={"model": model}, files=self._audio(path)).json()["probs"]

    def transcribe(self, path: Path, model: str, language: str | None = None) -> str:
        data = {"model": model} | ({"language": language} if language else {})
        return self._send("POST", "/infer/v1/transcribe", data=data, files=self._audio(path)).json()["text"]

    # --- the Claude Code map job ----------------------------------------------------------------------------------
    def map_draft(self, build_dir: Path, job_id: str, place: str, language: str, model: str, effort: str,
                  budget_usd: float, poll_s: float = 10, deadline_s: float = 2400):
        """Run the map LLM job on the host for a build folder (candidates.json, frames/, previous.json?) and copy its
        files into build_dir/job/, as a local run would leave them. -> (MapDraft, LlmRun)."""
        from nav.mapping.llm import LlmError, parse_claude_result

        files = [("candidates", ("candidates.json", (build_dir / "candidates.json").read_bytes(), "application/json"))]
        if (build_dir / "previous.json").is_file():
            files.append(("previous", ("previous.json", (build_dir / "previous.json").read_bytes(), "application/json")))
        files += [("frames", (p.name, p.read_bytes(), "image/jpeg")) for p in sorted((build_dir / "frames").glob("*.jpg"))]
        data = {"job_id": job_id, "place": place, "language": language, "model": model, "effort": effort,
                "budget_usd": str(budget_usd)}
        status = self._send("POST", "/infer/v1/jobs/map-draft", data=data, files=files).json()
        log.info("map job %s on %s: %s", job_id, self.base_url, status["status"])
        end = time.monotonic() + deadline_s
        while status["status"] in ("queued", "running"):
            if time.monotonic() > end:
                raise InferenceUnavailable(f"map job {job_id} still {status['status']} after {deadline_s:.0f} s")
            self._sleep(poll_s)
            status = self._send("GET", f"/infer/v1/jobs/{job_id}").json()
        job = build_dir / "job"
        job.mkdir(parents=True, exist_ok=True)
        for name in status.get("files", []):
            (job / name).write_bytes(self._send("GET", f"/infer/v1/jobs/{job_id}/files/{name}").content)
        if not (job / "claude.json").is_file():
            raise LlmError(f"map job {job_id} {status['status']} without a result: {status.get('error')}")
        draft, run = parse_claude_result(job / "claude.json", model)
        run.duration_s, run.effort = status.get("duration_s"), effort
        return draft, run
