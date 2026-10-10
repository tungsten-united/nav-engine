"""nav-api's /localize with the phone's motion (nav/navigation/localize.py): frames pointed at the floor are left out,
the walked distance since the last confirmed node gates confirmations, and the compass can penalise references
facing away. Without motion nothing changes. The map is test_nav_api's lshape: n1 -> n2 10 m east, n2 -> n3 10 m
south; step length 0.7 m."""

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from test_nav_api import API, THRESHOLDS, Places, axis, localize, publish, unit

from nav.nav_api.app import create_nav_app
from nav.nav_api.maps import MapCache
from nav.navigation.localize import camera_pitch
from nav.schemas.navigation import LiveOrientation

LEVEL = {"alpha": 0.0, "beta": 90.0, "gamma": 0.0, "absolute": True}  # phone upright, camera level
FLOOR = {"alpha": 0.0, "beta": 15.0, "gamma": 0.0, "absolute": True}  # tilted back: camera at the floor


def motion(steps: int | None = None, heading: float | None = None, orientation=LEVEL, cadence: float = 1.0) -> str:
    return json.dumps({"speedMps": 0.5, "cadenceHz": cadence, "stepCount": steps, "stepLengthM": 0.7,
                       "headingDeg": heading, "headingSource": None if heading is None else "orientation_absolute",
                       "headingAccuracyDeg": None, "orientation": orientation, "measuredAt": 1791561600000,
                       "somethingNew": 1})


def client(tmp_path, **thresholds):
    places = Places()
    map_id = publish(tmp_path / "maps")
    th = THRESHOLDS.model_copy(update=thresholds)
    app = create_nav_app(MapCache(tmp_path / "maps", reload_s=0), place_fn=places, api_token=None, cors_origins=[],
                         cors_origin_regex=None, frontend_url=None, thresholds=th)
    return TestClient(app), places, map_id


def test_camera_pitch():
    assert camera_pitch(LiveOrientation(**LEVEL)) == pytest.approx(0, abs=0.1)
    assert camera_pitch(LiveOrientation(**FLOOR)) == pytest.approx(-75, abs=0.5)
    assert camera_pitch(LiveOrientation(alpha=0, beta=0, gamma=0)) == pytest.approx(-90)  # flat on a table
    assert camera_pitch(None) is None


def test_walked_distance_gates_a_confirmation(tmp_path):
    c, places, map_id = client(tmp_path)
    with c:
        places.answer = axis(2)[None]  # n2's "at" photo, 10 m from n1
        assert localize(c, map_id, previous="n1").json()["status"] == "confirmed"  # no motion: as before
        j = localize(c, map_id, previous="n1", motion=motion(steps=5), previous_step_count="0").json()
        assert j["status"] == "uncertain" and "only 4 m walked" in j["reason"] and j["walked_m"] == pytest.approx(3.5)
        n2 = next(x for x in j["candidates"] if x["node"] == "n2")
        assert n2["distance_m"] == pytest.approx(10) and n2["plausible"] is False
        j = localize(c, map_id, previous="n1", motion=motion(steps=10), previous_step_count="0").json()
        assert j["status"] == "confirmed" and j["walked_m"] == pytest.approx(7.0)  # >= 10 - 3.5
        # staying at the last confirmed node is always plausible
        places.answer = axis(0)[None]
        j = localize(c, map_id, previous="n1", motion=motion(steps=0), previous_step_count="0").json()
        assert j["status"] == "confirmed" and j["best"] == "n1"
        # no previous_step_count: no gate
        places.answer = axis(2)[None]
        j = localize(c, map_id, previous="n1", motion=motion(steps=5)).json()
        assert j["status"] == "confirmed" and j["walked_m"] is None and j["moving"] is True


def test_frames_pointed_at_the_floor_are_left_out(tmp_path):
    c, places, map_id = client(tmp_path)
    with c:
        places.answer = np.stack([axis(4), axis(2)])  # frame 1 (at the floor) looks like n3, frame 2 like n2
        files = [("images", (f"f{i}.jpg", b"\xff\xd8x", "image/jpeg")) for i in range(2)]
        j = c.post(f"{API}/maps/{map_id}/localize", files=files,
                   data={"motion": [motion(orientation=FLOOR), motion(orientation=LEVEL)]}).json()
        assert j["frames_used"] == 1 and j["best"] == "n2" and j["pitch_deg"][0] < -55
        j = c.post(f"{API}/maps/{map_id}/localize", files=files,
                   data={"motion": [motion(orientation=FLOOR), motion(orientation=FLOOR)]}).json()
        assert j["status"] == "lost" and "floor" in j["reason"] and j["frames_used"] == 0


def test_heading_penalty_breaks_a_look_alike_tie(tmp_path):
    query = unit(0, 0, 1, 0, 1)[None]  # exactly as close to n2's "at" (facing east) as to n3's (facing south)
    c, places, map_id = client(tmp_path)
    with c:
        places.answer = query
        j = localize(c, map_id, motion=motion(heading=180)).json()
        assert j["status"] == "uncertain" and not j["heading_used"]  # heading_weight 0: off by default
    c, places, map_id = client(tmp_path / "b", heading_weight=0.4)
    with c:
        places.answer = query
        j = localize(c, map_id, motion=motion(heading=180)).json()  # facing south, like n3's photo
        assert j["heading_used"] and j["best"] == "n3" and j["status"] == "confirmed" and j["heading_map_deg"] == 180
        j = localize(c, map_id, motion=motion(heading=None)).json()  # no compass: no penalty
        assert not j["heading_used"] and j["status"] == "uncertain"


def test_motion_parts_are_validated(tmp_path):
    c, places, map_id = client(tmp_path)
    with c:
        assert localize(c, map_id, motion="null").status_code == 200
        assert localize(c, map_id, motion='{"stepCount": "many"}').status_code == 422
        files = [("images", ("f.jpg", b"\xff\xd8x", "image/jpeg"))]
        r = c.post(f"{API}/maps/{map_id}/localize", files=files, data={"motion": [motion(), motion()]})
        assert r.status_code == 422 and "more motion parts" in r.json()["detail"]


def test_walked_distance_replaces_one_edge_from_previous(tmp_path):
    """A missed node no longer stalls the route: n3 is two edges (20 m) from n1, fine once walked that far."""
    c, places, map_id = client(tmp_path)
    with c:
        places.answer = axis(4)[None]  # n3's "at" photo
        j = localize(c, map_id, previous="n1").json()  # no motion: the old rule
        assert j["status"] == "uncertain" and "not next to" in j["reason"]
        j = localize(c, map_id, previous="n1", motion=motion(steps=10), previous_step_count="0").json()  # 7 m
        assert j["status"] == "uncertain" and "only 7 m walked" in j["reason"]
        j = localize(c, map_id, previous="n1", motion=motion(steps=20), previous_step_count="0").json()  # 14 m
        assert j["status"] == "confirmed" and j["best"] == "n3"  # >= 20 - 7
