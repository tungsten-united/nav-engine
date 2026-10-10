import json
import subprocess

import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient

from nav.map_api.app import create_app
from nav.recorder.store import RecordingStore

API = "/api/v1"
CREATE = {
    "place": "  Test   floor ",
    "t0_epoch_ms": 1_791_500_000_000,
    "device": {"user_agent": "pytest"},
    "camera": {"width": 320, "height": 240, "fps": 10, "mime": "video/webm;codecs=vp9"},
}


@pytest.fixture(scope="session")
def webm(tmp_path_factory) -> bytes:
    """A 3 s VP9 webm, standing in for what MediaRecorder produces."""
    path = tmp_path_factory.mktemp("media") / "clip.webm"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10:duration=3",
                    "-c:v", "libvpx-vp9", "-deadline", "realtime", str(path)], check=True)
    return path.read_bytes()


@pytest.fixture
def client(tmp_path):
    store = RecordingStore(tmp_path / "recordings")
    with TestClient(create_app(store)) as c:
        yield c


def create(client) -> str:
    r = client.post(f"{API}/recordings", json=CREATE, headers={"Cf-Access-Authenticated-User-Email": "a@b.c"})
    assert r.status_code == 201, r.text
    m = r.json()
    assert m["status"] == "recording" and m["created_by"] == "a@b.c" and m["schema_version"] == 1
    return m["rec_id"]


def batch(seq: int, t0: float) -> dict:
    return {
        "seq": seq,
        "imu": [{"t_ms": t0 + i * 16.7, "ax": 0.1, "ay": 9.8, "az": 0.2, "gx": 1.0, "gy": 0.0, "gz": -2.0}
                for i in range(60)],
        "orientation": [{"t_ms": t0 + i * 16.7, "alpha": 10.0, "beta": 90.0, "gamma": 0.0, "absolute": True,
                         "heading_deg": 350.0} for i in range(60)],
        "events": [{"t_ms": t0 + 500, "type": "mark", "data": {"tag": "door"}}],
    }


def upload(client, rec_id: str, webm: bytes, n_chunks: int = 3, skip_chunk: int | None = None) -> None:
    size = len(webm) // n_chunks + 1
    for seq in range(n_chunks):
        if seq != skip_chunk:
            r = client.put(f"{API}/recordings/{rec_id}/video/{seq}", content=webm[seq * size:(seq + 1) * size])
            assert r.status_code == 204, r.text
    for seq in range(3):
        assert client.post(f"{API}/recordings/{rec_id}/samples", json=batch(seq, seq * 1000.0)).status_code == 204
    r = client.put(f"{API}/recordings/{rec_id}/voice/0", params={"t_start_ms": 1200, "t_end_ms": 2400},
                   content=b"fake-opus", headers={"Content-Type": "audio/webm;codecs=opus"})
    assert r.status_code == 200 and r.json()["file"] == "voice/000.webm"


def finish(client, rec_id: str, video_chunks: int = 3) -> dict:
    body = {"duration_ms": 3000, "video_chunks": video_chunks, "sample_batches": 3, "voice_clips": 1}
    r = client.post(f"{API}/recordings/{rec_id}/finish", json=body)  # TestClient runs the background finalize
    assert r.status_code == 202, r.text
    return client.get(f"{API}/recordings/{rec_id}").json()


def test_full_recording(client, webm, tmp_path):
    rec_id = create(client)
    upload(client, rec_id, webm)
    # retried uploads are idempotent
    assert client.post(f"{API}/recordings/{rec_id}/samples", json=batch(1, 1000.0)).status_code == 204

    m = finish(client, rec_id)
    assert m["status"] == "complete", m
    assert m["gaps"] == [] and m["error"] is None
    assert m["video"]["codec"] == "vp9" and m["video"]["width"] == 320 and not m["video"]["reencoded"]
    assert abs(m["video"]["duration_s"] - 3.0) < 0.3
    c = m["counts"]
    assert (c["video_chunks"], c["sample_batches"], c["imu"], c["orientation"], c["events"], c["marks"]) == (3, 3, 180, 180, 3, 3)

    d = tmp_path / "recordings" / rec_id
    imu = pq.read_table(d / "imu.parquet")
    assert imu.num_rows == 180 and imu.column_names[:4] == ["t_ms", "ax", "ay", "az"]
    assert imu.column("t_ms").to_pylist() == sorted(imu.column("t_ms").to_pylist())
    assert imu.column("lax").null_count == 180
    assert pq.read_table(d / "orientation.parquet").column("absolute").to_pylist()[0] is True
    events = [json.loads(line) for line in (d / "events.jsonl").read_text().splitlines()]
    assert [e["type"] for e in events] == ["mark"] * 3
    assert not list((d / "raw" / "video").glob("*.bin")), "video chunks are removed once video.mp4 exists"

    files = {f["path"] for f in client.get(f"{API}/recordings/{rec_id}/files").json()}
    assert {"manifest.json", "video.mp4", "imu.parquet", "orientation.parquet", "events.jsonl", "voice/000.webm"} <= files
    assert client.get(f"{API}/recordings/{rec_id}/files/video.mp4").content[4:8] == b"ftyp"

    # finalize is rerunnable after the chunks are gone
    m2 = client.post(f"{API}/recordings/{rec_id}/finalize", params={"wait": True}).json()
    assert m2["status"] == "complete" and m2["video"] == m["video"]
    # complete recordings are read-only
    assert client.put(f"{API}/recordings/{rec_id}/video/9", content=b"x").status_code == 409

    series = client.get(f"{API}/recordings/{rec_id}/series/imu").json()
    assert len(series["t_ms"]) == 180 and series["ay"][0] == 9.8 and series["lax"][0] is None
    thin = client.get(f"{API}/recordings/{rec_id}/series/orientation", params={"max_points": 100}).json()
    assert 90 <= len(thin["t_ms"]) <= 100 and thin["absolute"][0] is True
    assert client.get(f"{API}/recordings/{rec_id}/series/manifest").status_code == 404

    listed = client.get(f"{API}/recordings").json()
    assert [s["rec_id"] for s in listed] == [rec_id] and listed[0]["counts"]["marks"] == 3


def test_places_group_free_walks(client):
    ids = [create(client) for _ in range(2)]
    other = client.post(f"{API}/recordings", json={**CREATE, "place": "Library"}).json()["rec_id"]
    assert client.get(f"{API}/recordings/{ids[0]}").json()["place"] == "Test floor"  # whitespace normalised
    places = {p["place"]: p["recordings"] for p in client.get(f"{API}/places").json()}
    assert places == {"Test floor": 2, "Library": 1}
    same = client.get(f"{API}/recordings", params={"place": "test FLOOR"}).json()
    assert sorted(s["rec_id"] for s in same) == sorted(ids) and other not in [s["rec_id"] for s in same]
    assert client.post(f"{API}/recordings", json={**CREATE, "place": "  "}).status_code == 422


def test_missing_uploads_are_reported(client, webm):
    rec_id = create(client)
    upload(client, rec_id, webm, n_chunks=4, skip_chunk=3)  # lose the tail chunk
    m = finish(client, rec_id, video_chunks=4)
    assert m["status"] == "incomplete"
    assert m["gaps"] == ["video chunk 3"]
    assert m["video"]["duration_s"] > 1  # what arrived is still usable


def test_unknown_and_invalid(client):
    assert client.get(f"{API}/recordings/20260101T000000Z_0000").status_code == 404
    assert client.get(f"{API}/recordings/..%2F..%2Fetc").status_code == 404
    rec_id = create(client)
    assert client.get(f"{API}/recordings/{rec_id}/files/../../../etc/passwd").status_code == 404
    assert client.get(f"{API}/recordings/{rec_id}/files/%2E%2E/%2E%2E/pyproject.toml").status_code == 404
    bad = {"seq": 0, "events": [{"t_ms": 1, "type": "teleport"}]}
    assert client.post(f"{API}/recordings/{rec_id}/samples", json=bad).status_code == 422
    assert client.put(f"{API}/recordings/{rec_id}/video/0", content=b"").status_code == 400
    assert client.patch(f"{API}/recordings/{rec_id}", json={"step_length_m": 0.75}).json()["step_length_m"] == 0.75
    assert client.delete(f"{API}/recordings/{rec_id}").status_code == 204
    assert client.get(f"{API}/recordings/{rec_id}").status_code == 404


def test_web_app_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "Nav tools" in r.text
    assert "orient.harshdeepsingh.dev/map" in r.text  # recording moved to Orient's app
    assert client.get("/index.js").status_code == 200
    assert client.get(f"{API}/health").json() == {"ok": True, "service": "map-api", "schema_version": 1}


def test_dashboard_served(client):
    assert client.get("/dash", follow_redirects=False).headers["location"] == "/dash/"
    page = client.get("/dash/")
    assert page.status_code == 200 and "Nav Recordings" in page.text
    assert 'src="dash.js?v=' in page.text and 'href="dash.css?v=' in page.text
    assert client.get("/dash/dash.js").status_code == 200


def test_assets_are_not_cached(client):
    # Cloudflare edge-caches .js/.css unless told otherwise
    assert client.get("/index.js").headers["cache-control"] == "no-cache"
    page = client.get("/")
    assert page.headers["cache-control"] == "no-cache"
    assert 'src="index.js?v=' in page.text and 'href="style.css?v=' in page.text
    assert client.get(f"{API}/health").headers["cache-control"] == "no-store"
