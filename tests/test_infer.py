"""Inference host (nav/infer/server.py) and its client, with stub models in place of the GPU ones."""

import asyncio
import json
import threading
import time

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from nav.infer.client import InferenceClient, match_chunks
from nav.infer.errors import InferenceError, InferenceUnavailable
from nav.infer.server import Busy, GpuSlot, create_infer_app
from nav.mapping.build import run_llm_on_build
from test_mapping import candidate_set, draft

MODELS = {"embed": "dino", "place": "megaloc", "transcribe": "whisper"}
# The client bounds each request by its deadline; Starlette's TestClient (standing in for the network) ignores that.
pytestmark = pytest.mark.filterwarnings("ignore:You should not use the 'timeout' argument")


class Stub:
    models = MODELS

    def __init__(self):
        self.calls = []
        self.release = threading.Event()
        self.release.set()

    def embed(self, images, model):
        self.calls.append(("embed", len(images)))
        return np.array([[len(b), 1.0] for b in images], np.float32)

    def place(self, images, model):
        return np.array([[1.0, 0.0, float(len(b))] for b in images], np.float32)

    def match(self, images, pairs):
        self.calls.append(("match", len(images), len(pairs)))
        return [len(images[i]) + len(images[j]) for i, j in pairs]

    def transcribe(self, path, model):
        return f"heard {path.suffix}"

    def map_draft(self, work_dir, place, language, model, effort, budget_usd):
        self.release.wait(5)
        job = work_dir / "job"
        job.mkdir()
        assert (work_dir / "candidates.json").is_file() and list((work_dir / "frames").glob("*.jpg"))
        result = {"type": "result", "subtype": "success", "is_error": False, "total_cost_usd": 0.4,
                  "session_id": "s", "modelUsage": {"claude-x": {"costUSD": 0.4}},
                  "structured_output": draft(edges=[]).model_dump()}
        (job / "claude.json").write_text(json.dumps(result))
        (job / "prompt.md").write_text(f"{place} in {language}")


@pytest.fixture
def host(tmp_path):
    stub = Stub()
    app = create_infer_app(stub, jobs_dir=tmp_path / "jobs", queue=1, job_queue=0)
    with TestClient(app) as http:
        client = InferenceClient("http://testserver", sleep=lambda s: time.sleep(min(s, 0.01)))
        client.http = http
        yield stub, client, http


def frames(tmp_path, n: int) -> list:
    d = tmp_path / "frames"
    d.mkdir(exist_ok=True)
    out = []
    for i in range(n):
        p = d / f"{i:03d}.jpg"
        p.write_bytes(b"x" * (i + 1))
        out.append(p)
    return out


def test_models_round_trip_in_chunks(host, tmp_path):
    stub, client, _ = host
    paths = frames(tmp_path, 70)
    v = client.embed(paths, "dino")
    assert v.shape == (70, 2) and v[69, 0] == 70 and v.dtype == np.float32
    assert [c for c in stub.calls if c[0] == "embed"] == [("embed", 32), ("embed", 32), ("embed", 6)]
    assert client.place(paths[:3], "megaloc")[:, 2].tolist() == [1, 2, 3]

    pairs = [(i, j) for i in range(20) for j in range(i + 1, 20)][:150]
    got = client.match(paths, pairs)
    assert got == [(i + 1) + (j + 1) for i, j in pairs]
    for c in stub.calls:
        if c[0] == "match":
            assert c[1] <= 48 and c[2] <= 64

    clip = tmp_path / "000.webm"
    clip.write_bytes(b"opus")
    assert client.transcribe(clip, "whisper") == "heard .webm"


def test_wrong_model_and_bad_requests_are_refused(host, tmp_path):
    _, client, http = host
    paths = frames(tmp_path, 2)
    with pytest.raises(InferenceError, match="409"):
        client.embed(paths, "some-other-dino")
    files = [("images", (p.name, p.read_bytes(), "image/jpeg")) for p in frames(tmp_path, 33)]
    assert http.post("/infer/v1/embed", data={"model": "dino"}, files=files).status_code == 422
    assert http.post("/infer/v1/match", data={"pairs": "[[0, 5]]"}, files=files[:2]).status_code == 422
    big = http.post("/infer/v1/embed", content=b"x", headers={"content-length": str(17 * 1024 * 1024)})
    assert big.status_code == 413


def test_match_chunks_ship_each_image_once():
    pairs = [(0, i) for i in range(1, 100)]
    chunks = match_chunks(pairs, max_pairs=64, max_images=48)
    assert sorted(k for c in chunks for k in c) == list(range(99))
    for c in chunks:
        assert len(c) <= 64 and len({k for pos in c for k in pairs[pos]}) <= 48


def test_gpu_slot_runs_one_at_a_time_and_turns_away_the_overflow():
    async def main():
        slot = GpuSlot(queue=1)
        gate, running, peak = threading.Event(), [0], [0]

        def work():
            running[0] += 1
            peak[0] = max(peak[0], running[0])
            gate.wait(2)
            running[0] -= 1
            return "ok"

        a = asyncio.create_task(slot.run(work))
        b = asyncio.create_task(slot.run(work))
        await asyncio.sleep(0.05)
        with pytest.raises(Busy):
            await slot.run(work)
        gate.set()
        assert await a == "ok" and await b == "ok" and peak[0] == 1

    asyncio.run(main())


def test_client_retries_busy_then_gives_up_at_the_deadline():
    seen, waits = [], []

    def handler(request):
        seen.append(request.url.path)
        return httpx.Response(503, headers={"Retry-After": "2"}) if len(seen) < 3 else httpx.Response(200, json={"ok": 1})

    c = InferenceClient("http://h", transport=httpx.MockTransport(handler), sleep=waits.append)
    assert c.health() == {"ok": 1} and waits == [2.0, 2.0]

    always = InferenceClient("http://h", deadline_s=5, transport=httpx.MockTransport(lambda r: httpx.Response(503)),
                             sleep=lambda s: None)
    with pytest.raises(InferenceUnavailable, match="busy"):
        always._send("GET", "/infer/v1/health")


def test_cloudflare_login_redirect_means_a_bad_token_and_is_not_retried():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(302, headers={"location": "https://x.cloudflareaccess.com/cdn-cgi/access/login"})

    c = InferenceClient("http://h", transport=httpx.MockTransport(handler), sleep=lambda s: None)
    with pytest.raises(InferenceUnavailable, match="service token"):
        c.health()
    assert len(calls) == 1


def test_map_job_through_the_host_lands_like_a_local_run(host, tmp_path, monkeypatch):
    stub, client, http = host
    build = tmp_path / "maps" / "test" / "builds" / "20261010T000000Z"
    (build / "frames").mkdir(parents=True)
    (build / "candidates.json").write_text(candidate_set().model_dump_json())
    for cid in ("c1", "c2"):
        (build / "frames" / f"{cid}.jpg").write_bytes(b"jpg")

    from nav import config
    from nav.mapping import inference

    monkeypatch.setattr(config, "INFER_URL", "http://testserver")
    monkeypatch.setattr(inference, "remote", lambda: client)
    res = run_llm_on_build(build, llm_model="opus", language="es", budget_usd=1)
    assert res.map.build.llm.model == "claude-x" and res.map.build.llm.effort == "medium"
    assert (build / "job" / "claude.json").is_file() and (build / "job" / "prompt.md").read_text() == "Test in es"
    assert json.loads((tmp_path / "maps" / "test" / "map.json").read_text())["build"]["build_id"] == "20261010T000000Z"

    # resubmitting the same job returns it; a second job while one runs and none may wait is turned away
    files = {"candidates": ("candidates.json", b"{}"), "frames": ("c1.jpg", b"jpg")}
    data = {"job_id": "test-20261010T000000Z", "place": "Test", "language": "es", "model": "opus",
            "effort": "medium", "budget_usd": "1"}
    assert http.post("/infer/v1/jobs/map-draft", data=data, files=files).json()["status"] == "done"
    stub.release.clear()
    assert http.post("/infer/v1/jobs/map-draft", data={**data, "job_id": "a"}, files=files).status_code == 202
    busy = http.post("/infer/v1/jobs/map-draft", data={**data, "job_id": "b"}, files=files)
    assert busy.status_code == 503 and busy.headers["retry-after"]
    stub.release.set()
    assert http.get("/infer/v1/jobs/nope").status_code == 404
    assert http.get("/infer/v1/jobs/a/files/../../etc").status_code == 404


def test_gpu_slot_serves_live_calls_before_waiting_batches():
    async def main():
        slot = GpuSlot(queue=4, live_queue=1)
        gate, order = threading.Event(), []

        def work(name):
            if name == "build-1":
                gate.wait(2)
            order.append(name)
            return name

        first = asyncio.create_task(slot.run(work, "build-1"))
        await asyncio.sleep(0.05)
        batch = [asyncio.create_task(slot.run(work, f"build-{i}")) for i in (2, 3)]
        await asyncio.sleep(0.05)
        live = asyncio.create_task(slot.run(work, "live", live=True))
        await asyncio.sleep(0.05)
        assert slot.live_pending == 1
        extra = asyncio.create_task(slot.run(work, "live-2", live=True))  # one more may wait
        await asyncio.sleep(0.05)
        with pytest.raises(Busy):  # the live lane is full; the batch queue is not what limits it
            await slot.run(work, "live-3", live=True)
        gate.set()
        await asyncio.gather(first, live, extra, *batch)
        assert order == ["build-1", "live", "live-2", "build-2", "build-3"]
        assert slot.pending == slot.live_pending == 0 and not slot._busy

    asyncio.run(main())


def test_gpu_slot_survives_a_cancelled_waiter():
    async def main():
        slot = GpuSlot(queue=2)
        gate = threading.Event()
        first = asyncio.create_task(slot.run(lambda: gate.wait(2)))
        await asyncio.sleep(0.05)
        waiter = asyncio.create_task(slot.run(lambda: "never"))
        await asyncio.sleep(0.05)
        waiter.cancel()
        gate.set()
        await first
        assert await slot.run(lambda: "next") == "next" and slot.pending == 0

    asyncio.run(main())


def test_live_frames_go_in_the_live_lane(host):
    stub, client, http = host
    gpu, seen = http.app.state.gpu, []
    place = stub.place

    def spy(images, model):
        seen.append((gpu.live_pending, gpu.pending))
        return place(images, model)

    stub.place = spy
    v = client.place_live([b"abc", b"de"], "megaloc", deadline_s=2)
    assert v.shape == (2, 3) and v[:, 2].tolist() == [3, 2]
    client.place([p for p in frames(http.app.state.jobs.root.parent, 1)], "megaloc")
    assert seen == [(1, 0), (0, 1)]
