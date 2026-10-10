"""nav-api's /localize with the phone's motion (nav/navigation/localize.py): frames pointed at the floor are left out,
and the walked distance since the last confirmed node gates confirmations. Without motion nothing changes. The map
is test_nav_api's lshape: n1 -> n2 10 m east, n2 -> n3 10 m south; step length 0.7 m."""

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


def frames(n: int) -> list:
    return [("images", (f"f{i}.jpg", b"\xff\xd8x", "image/jpeg")) for i in range(n)]


def client(tmp_path, edges=("e1", "e2")):
    places = Places()
    map_id = publish(tmp_path / "maps", edges=edges)
    app = create_nav_app(MapCache(tmp_path / "maps", reload_s=0), place_fn=places, api_token=None, cors_origins=[],
                         cors_origin_regex=None, frontend_url=None, thresholds=THRESHOLDS)
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
        assert "7 m walked from the last confirmed node" in j["reason"]
        # staying at the last confirmed node is always plausible
        places.answer = axis(0)[None]
        j = localize(c, map_id, previous="n1", motion=motion(steps=0), previous_step_count="0").json()
        assert j["status"] == "confirmed" and j["best"] == "n1"
        # no previous_step_count: no gate
        places.answer = axis(2)[None]
        j = localize(c, map_id, previous="n1", motion=motion(steps=5)).json()
        assert j["status"] == "confirmed" and j["walked_m"] is None


def test_a_step_count_that_went_back_leaves_the_gate_out(tmp_path):
    """The phone restarted counting while the caller kept its old count: the one-edge rule decides, not 0 m walked."""
    c, places, map_id = client(tmp_path)
    with c:
        places.answer = axis(2)[None]  # n2, one edge from n1
        j = localize(c, map_id, previous="n1", motion=motion(steps=3), previous_step_count="50").json()
        assert j["status"] == "confirmed" and j["walked_m"] is None
        assert all(x["plausible"] is None for x in j["candidates"])
        places.answer = axis(4)[None]  # n3, two edges from n1
        j = localize(c, map_id, previous="n1", motion=motion(steps=3), previous_step_count="50").json()
        assert j["status"] == "uncertain" and "not next to" in j["reason"]


def test_a_node_not_connected_to_previous_stays_uncertain(tmp_path):
    c, places, map_id = client(tmp_path, edges=("e1",))  # n3 has no edge
    with c:
        places.answer = axis(4)[None]
        j = localize(c, map_id, previous="n1", motion=motion(steps=100), previous_step_count="0").json()
        assert j["status"] == "uncertain" and "is not connected to" in j["reason"]


def test_frames_pointed_at_the_floor_are_left_out(tmp_path):
    c, places, map_id = client(tmp_path)
    with c:
        places.answer = np.stack([axis(4), axis(2)])  # frame 1 (at the floor) looks like n3, frame 2 like n2
        j = c.post(f"{API}/maps/{map_id}/localize", files=frames(2),
                   data={"motion": [motion(orientation=FLOOR), motion(orientation=LEVEL)]}).json()
        assert j["frames_used"] == 1 and j["best"] == "n2" and j["pitch_deg"][0] < -55
        j = c.post(f"{API}/maps/{map_id}/localize", files=frames(2),
                   data={"motion": [motion(orientation=FLOOR), motion(orientation=FLOOR)]}).json()
        assert j["status"] == "lost" and "floor" in j["reason"] and j["frames_used"] == 0


def test_the_compass_of_the_newest_motion_is_reported_not_scored(tmp_path):
    c, places, map_id = client(tmp_path)
    with c:
        places.answer = unit(0, 0, 1, 0, 1)[None]  # exactly as close to n2's "at" (facing east) as to n3's (south)
        j = localize(c, map_id, heading_deg="90", motion=motion(heading=180)).json()
        assert j["heading_map_deg"] == 180 and j["status"] == "uncertain"  # the motion's heading wins; no scoring
        j = localize(c, map_id, heading_deg="90", motion=motion(heading=None)).json()
        assert j["heading_map_deg"] == 90


def test_motion_parts_are_validated(tmp_path):
    c, places, map_id = client(tmp_path)
    with c:
        assert localize(c, map_id, motion="null").status_code == 200
        assert localize(c, map_id, motion='{"stepCount": "many"}').status_code == 422
        r = c.post(f"{API}/maps/{map_id}/localize", files=frames(1), data={"motion": [motion(), motion()]})
        assert r.status_code == 422 and "one motion part per image" in r.json()["detail"]
        # one motion for two frames would land on the older one: refused, not guessed
        r = c.post(f"{API}/maps/{map_id}/localize", files=frames(2), data={"motion": [motion()]})
        assert r.status_code == 422 and "one motion part per image" in r.json()["detail"]
        places.answer = np.stack([axis(2), axis(2)])
        r = c.post(f"{API}/maps/{map_id}/localize", files=frames(2), data={"motion": ["null", motion()]})
        assert r.status_code == 200 and r.json()["frames_used"] == 2


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


def test_the_motion_gate_reaches_the_log_line(tmp_path, capsys):
    c, places, map_id = client(tmp_path)
    with c:
        places.answer = axis(2)[None]
        localize(c, map_id, previous="n1", motion=motion(steps=5), previous_step_count="0")
    [e] = [json.loads(x) for x in capsys.readouterr().out.splitlines() if x.startswith("{")]
    assert e["kind"] == "nav_localize" and e["status"] == "uncertain" and "only 4 m walked" in e["message"]
    assert e["walkedM"] == pytest.approx(3.5) and e["distanceM"] == pytest.approx(10) and e["plausible"] is False
    assert e["framesUsed"] == 1
