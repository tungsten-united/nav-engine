"""The options of the public (Cloud Run) deployment: API token, CORS, no deletes, map builds through the API."""

import time
import threading
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from nav.mapping.store import MapStore
from nav.map_api.app import create_app
from nav.recorder.store import RecordingStore
from test_recorder import API, CREATE, create, finish, upload, webm  # noqa: F401 (webm is a fixture)


def app(tmp_path, **kw):
    return create_app(RecordingStore(tmp_path / "recordings"), MapStore(tmp_path / "maps"),
                      cache_dir=tmp_path / "cache", **kw)


def test_token_gates_the_api_but_not_health_or_pages(tmp_path):
    with TestClient(app(tmp_path, api_token="s3cret")) as c:
        assert c.get(f"{API}/health").status_code == 200
        r = c.get(f"{API}/recordings")
        assert r.status_code == 401 and r.headers["www-authenticate"] == "Bearer"
        assert c.get(f"{API}/recordings", headers={"Authorization": "Bearer nope"}).status_code == 401
        assert c.get(f"{API}/recordings", headers={"Authorization": "Bearer s3cret"}).json() == []
        assert c.get(f"{API}/places", params={"token": "s3cret"}).status_code == 200  # for <video>/<img> URLs
        r = c.post(f"{API}/recordings", json=CREATE, headers={"Authorization": "Bearer s3cret", "X-Nav-User": "ana"})
        assert r.status_code == 201 and r.json()["created_by"] == "ana"
        page = c.get("/")
        assert page.status_code == 200 and 'src="config.js?v=' in page.text
        assert 'src="../config.js?v=' in c.get("/dash/").text


def test_without_settings_nothing_changes(tmp_path):
    with TestClient(app(tmp_path, api_token=None, cors_origins=[], cors_origin_regex=None)) as c:
        assert c.get(f"{API}/recordings").status_code == 200
        r = c.options(f"{API}/recordings", headers={"Origin": "http://localhost:5500",
                                                     "Access-Control-Request-Method": "GET"})
        assert "access-control-allow-origin" not in r.headers


def test_cors_preflight_and_401_carry_cors_headers(tmp_path):
    with TestClient(app(tmp_path, api_token="s3cret", cors_origins=["https://front.example"],
                        cors_origin_regex=r"^http://localhost(:\d+)?$")) as c:
        pre = c.options(f"{API}/recordings", headers={"Origin": "http://localhost:5500",
                                                       "Access-Control-Request-Method": "GET",
                                                       "Access-Control-Request-Headers": "authorization"})
        assert pre.status_code == 200 and pre.headers["access-control-allow-origin"] == "http://localhost:5500"
        r = c.get(f"{API}/recordings", headers={"Origin": "https://front.example"})
        assert r.status_code == 401 and r.headers["access-control-allow-origin"] == "https://front.example"
        r = c.get(f"{API}/recordings", headers={"Origin": "https://evil.example"})
        assert "access-control-allow-origin" not in r.headers


def test_deleting_can_be_disabled(tmp_path):
    with TestClient(app(tmp_path, allow_delete=False)) as c:
        rec_id = create(c)
        assert c.delete(f"{API}/recordings/{rec_id}").status_code == 403
        assert c.get(f"{API}/recordings/{rec_id}").status_code == 200


def test_no_frontend_when_web_dir_is_off(tmp_path):
    with TestClient(app(tmp_path, web_dir=None)) as c:
        assert c.get("/").status_code == 404
        assert c.get(f"{API}/health").status_code == 200


def test_map_build_runs_in_the_background_one_at_a_time(tmp_path, webm, monkeypatch):  # noqa: F811
    release, calls = threading.Event(), []

    def fake_build_map(place, **kw):
        calls.append((place, kw))
        release.wait(5)
        return SimpleNamespace(map=None, candidates=SimpleNamespace(candidates=[1, 2, 3]))

    monkeypatch.setattr("nav.mapping.build.build_map", fake_build_map)
    with TestClient(app(tmp_path)) as c:
        assert c.post(f"{API}/maps/build", json={"place": "Test floor"}).status_code == 404  # nothing recorded yet
        rec_id = create(c)
        upload(c, rec_id, webm)
        assert finish(c, rec_id)["status"] == "complete"

        r = c.post(f"{API}/maps/build", json={"place": "test FLOOR", "llm": False})
        assert r.status_code == 202, r.text
        b = r.json()
        assert b["map_id"] == "test-floor" and b["status"] == "running"
        assert c.post(f"{API}/maps/build", json={"place": "Test floor"}).status_code == 409
        assert c.get(b["status_url"]).json()["status"] == "running"
        release.set()
        for _ in range(100):
            s = c.get(b["status_url"]).json()
            if s["status"] != "running":
                break
            time.sleep(0.02)
        assert s["status"] == "done" and s["published"] is False and s["candidates"] == 3 and s["finished_at_utc"]
        assert calls[0][1]["run_llm"] is False and calls[0][1]["build_id"] == b["build_id"]
        assert c.get(f"{API}/maps/test-floor/builds/nope/status").status_code == 404


def test_failed_build_reports_why(tmp_path, webm, monkeypatch):  # noqa: F811
    from nav.infer.errors import InferenceUnavailable

    def broken(place, **kw):
        raise InferenceUnavailable("inference host unreachable")

    monkeypatch.setattr("nav.mapping.build.build_map", broken)
    with TestClient(app(tmp_path)) as c:
        rec_id = create(c)
        upload(c, rec_id, webm)
        finish(c, rec_id)
        b = c.post(f"{API}/maps/build", json={"place": "Test floor"}).json()
        for _ in range(100):
            s = c.get(b["status_url"]).json()
            if s["status"] != "running":
                break
            time.sleep(0.02)
        assert s["status"] == "failed" and "unreachable" in s["error"]
        assert c.post(f"{API}/maps/build", json={"place": "Test floor"}).status_code == 202  # free again


@pytest.fixture(autouse=True)
def _no_env_settings(monkeypatch):
    for k in ("NAV_API_TOKEN", "NAV_CORS_ORIGINS", "NAV_INFER_URL"):
        monkeypatch.delenv(k, raising=False)


def test_large_files_are_served_in_ranges_under_the_cloud_run_cap(tmp_path, monkeypatch):
    monkeypatch.setattr("nav.web.MAX_RESPONSE", 1000)
    data = bytes(range(256)) * 20  # 5120 bytes
    with TestClient(app(tmp_path)) as c:
        rec_id = create(c)
        (tmp_path / "recordings" / rec_id / "video.mp4").write_bytes(data)
        url = f"{API}/recordings/{rec_id}/files/video.mp4"
        r = c.get(url, headers={"Range": "bytes=0-"})  # what a <video> asks first
        assert r.status_code == 206 and r.headers["content-range"] == "bytes 0-999/5120" and r.content == data[:1000]
        assert r.headers["content-type"] == "video/mp4"
        r = c.get(url, headers={"Range": "bytes=4500-"})
        assert r.headers["content-range"] == "bytes 4500-5119/5120" and r.content == data[4500:]
        assert c.get(url, headers={"Range": "bytes=-100"}).content == data[-100:]
        assert c.get(url, headers={"Range": "bytes=9000-"}).status_code == 416
        whole = c.get(url)
        assert whole.status_code == 200 and whole.content == data and "content-length" not in whole.headers
        small = c.get(f"{API}/recordings/{rec_id}/files/manifest.json")
        assert small.status_code == 200 and small.json()["rec_id"] == rec_id


def test_localization_and_navigation_pages_are_served_with_hashed_assets(tmp_path):
    with TestClient(app(tmp_path, api_token="s3cret")) as c:
        loc = c.get("/loc/")
        assert loc.status_code == 200 and "Nav Localize" in loc.text
        assert 'src="../config.js?v=' in loc.text and 'src="loc-core.js?v=' in loc.text and 'href="loc.css?v=' in loc.text
        nav = c.get("/nav/")
        assert nav.status_code == 200 and "Nav Navigate" in nav.text
        assert 'src="../loc/loc-core.js?v=' in nav.text and 'src="nav.js?v=' in nav.text
        assert 'href="../loc/loc.css?v=' in nav.text
        assert c.get("/nav", follow_redirects=False).headers["location"] == "/nav/"
        assert c.get("/loc/loc-core.js").status_code == 200 and c.get("/nav/nav.js").status_code == 200
