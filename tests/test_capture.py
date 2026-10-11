"""nav-api's live capture (nav/nav_api/capture.py) and reading it back (nav/navigation/captured.py)."""

import json
from datetime import datetime, timezone

import numpy as np
from fastapi.testclient import TestClient

from nav.infer.errors import InferenceUnavailable
from nav.nav_api.app import create_nav_app
from nav.nav_api.capture import Capture, safe
from nav.nav_api.maps import MapCache
from nav.navigation.captured import sessions
from test_nav_api import API, JPEG, THRESHOLDS, Places, publish, unit
from test_navigation import BUILD

MOTION = {"stepCount": 12, "headingDeg": 95.0, "orientation": {"alpha": 0, "beta": 80, "gamma": 0}}


def client(tmp_path, capture):
    places = Places()
    map_id = publish(tmp_path / "maps")
    app = create_nav_app(MapCache(tmp_path / "maps", reload_s=0), place_fn=places, api_token=None, cors_origins=[],
                         cors_origin_regex=None, frontend_url=None, thresholds=THRESHOLDS,
                         infer_health_fn=lambda: {"ok": True}, capture=capture)
    return TestClient(app), places, map_id


def test_every_call_of_a_session_is_recorded_with_its_frames(tmp_path):
    clock = iter(datetime(2026, 10, 11, 18, 0, s, 250_000, tzinfo=timezone.utc) for s in range(10))
    cap = Capture(tmp_path / "live", clock=lambda: next(clock))
    c, places, map_id = client(tmp_path, cap)
    who = {"X-Session-Id": "walk/42", "X-Client-Id": "phone-1", "X-Request-Id": "req-1"}
    with c:
        r = c.post(f"{API}/maps/{map_id}/route", json={"start": "n1", "goal": "n3", "trust": "any"}, headers=who)
        assert r.status_code == 200
        places.answer = np.stack([unit(0.1, 0, 1), unit(0, 0, 1)])
        files = [("images", (f"f{i}.jpg", JPEG + bytes([i]), "image/jpeg")) for i in range(2)]
        r = c.post(f"{API}/maps/{map_id}/localize", files=files, headers={**who, "X-Request-Id": "req-2"},
                   data={"previous": "n1", "expected": "n2", "previous_step_count": "3",
                         "motion": [json.dumps(MOTION), "null"]})
        assert r.status_code == 200
        places.answer = InferenceUnavailable("down")  # a failed call is kept too: its frames are still a test case
        r = c.post(f"{API}/maps/{map_id}/localize", files=files[:1], headers={"X-Session-Id": "walk/42"})
        assert r.status_code == 503
        assert cap.flush()

    folder = tmp_path / "live" / map_id / "2026-10-11" / "walk-42"
    names = sorted(p.name for p in folder.iterdir())
    assert names[0] == "180000250-req-1-route.json"
    assert "180001250-req-2-localize-f0.jpg" in names and "180001250-req-2-localize-f1.jpg" in names
    assert (folder / "180001250-req-2-localize-f1.jpg").read_bytes() == JPEG + b"\x01"

    [s] = sessions(tmp_path / "live")
    assert (s.map_id, s.date, s.session) == (map_id, "2026-10-11", "walk-42")
    route, loc, failed = s.calls
    assert route.kind == "route" and route.response["found"] and route.request["goal"] == "n3"
    assert route.build_id == BUILD.build_id and route.who["clientId"] == "phone-1"
    assert loc.response["best"] == "n2" and loc.request["previous"] == "n1" and loc.request["previous_step_count"] == 3
    assert loc.images() == [JPEG + b"\x00", JPEG + b"\x01"]
    m0, m1 = loc.motions()
    assert m0.step_count == 12 and m0.heading_deg == 95 and m1 is None and loc.step_count == 12
    assert failed.response is None and failed.error["status"] == 503 and len(failed.frames) == 1
    assert s.summary() | {"builds": None} == {
        "map_id": map_id, "date": "2026-10-11", "session": "walk-42", "calls": 3, "localize": 2, "routes": 1,
        "frames": 3, "minutes": 0.0, "statuses": {"uncertain": 1, "error 503": 1}, "builds": None}


def test_nothing_is_recorded_unless_asked(tmp_path, monkeypatch):
    monkeypatch.setattr("nav.config.CAPTURE_DIR", None)
    places = Places()
    map_id = publish(tmp_path / "maps")
    app = create_nav_app(MapCache(tmp_path / "maps", reload_s=0), place_fn=places, api_token=None, cors_origins=[],
                         cors_origin_regex=None, frontend_url=None, thresholds=THRESHOLDS)
    assert app.state.capture is None
    monkeypatch.setattr("nav.config.CAPTURE_DIR", tmp_path / "live")
    app = create_nav_app(MapCache(tmp_path / "maps", reload_s=0), place_fn=places, api_token=None, cors_origins=[],
                         cors_origin_regex=None, frontend_url=None, thresholds=THRESHOLDS)
    assert app.state.capture.root == tmp_path / "live"


def test_a_bad_request_without_frames_leaves_nothing(tmp_path):
    cap = Capture(tmp_path / "live")
    c, _, map_id = client(tmp_path, cap)
    with c:
        assert c.post(f"{API}/maps/{map_id}/localize", data={"top_k": "3"}).status_code == 422
        assert cap.flush()
    assert not (tmp_path / "live").exists()


def test_session_names_are_safe_folder_names():
    assert safe("../../etc/passwd", "x") == "etc-passwd"
    assert safe("", "anonymous") == "anonymous" and safe(None, "anonymous") == "anonymous"
    assert safe("a" * 100, "x") == "a" * 64
