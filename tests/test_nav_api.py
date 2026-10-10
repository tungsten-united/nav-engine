"""nav-api (nav/nav_api/app.py): published maps read-only, routes, localization against reference embeddings."""

import json
import os

import numpy as np
import pytest
from fastapi.testclient import TestClient

from nav.infer.errors import InferenceError, InferenceUnavailable
from nav.map_api.app import create_app
from nav.mapping.build import finish_map
from nav.mapping.store import MapStore
from nav.nav_api.app import create_nav_app
from nav.nav_api.maps import MapCache
from nav.recorder.store import RecordingStore
from nav.schemas.map import EmbeddingInfo, ReferenceImage
from nav.schemas.navigation import Thresholds
from test_navigation import BUILD, lshape

API = "/api/v1"
MODEL = "megaloc@test"
DIM = 8
JPEG = b"\xff\xd8 not really a jpeg"
# reference id -> (node, role, camera heading, embedding axis)
REFS = {"r1": ("n1", "at", 90.0, 0), "r2": ("n1", "outgoing", 90.0, 1), "r3": ("n2", "at", 90.0, 2),
        "r4": ("n2", "approach", 90.0, 3), "r5": ("n3", "at", 180.0, 4)}
THRESHOLDS = Thresholds(confirm_score=0.6, confirm_margin=0.05, lost_score=0.45)


def unit(*axes: float) -> np.ndarray:
    v = np.zeros(DIM, np.float32)
    for i, w in enumerate(axes):
        v[i] = w
    return v / np.linalg.norm(v)


def axis(i: int) -> np.ndarray:
    v = np.zeros(DIM, np.float32)
    v[i] = 1
    return v


def publish(root, name: str = "N2", edges=("e1", "e2"), refs: bool = True) -> str:
    """The lshape map (n1 -> n2 east, n2 -> n3 south; compass-referenced) published under root/t, with one
    reference image per REFS entry and their embeddings."""
    draft, cset, tracks = lshape()
    build_rel = f"builds/{BUILD.build_id}"
    m = finish_map(draft, cset, BUILD, build_rel, tracks)
    m.edges = [e for e in m.edges if e.id in edges]
    m.nodes[1].name = name
    d = root / m.map_id / build_rel
    (d / "refs").mkdir(parents=True, exist_ok=True)
    (d / "job").mkdir(exist_ok=True)
    (d / "job" / "claude.json").write_text("{}")
    if refs:
        vectors = np.zeros((len(REFS), DIM), np.float32)
        for row, (rid, (node, role, heading, ax)) in enumerate(REFS.items()):
            m.references.append(ReferenceImage(id=rid, node=node, role=role, image=f"{build_rel}/refs/{rid}.jpg",
                                               rec_id="r1", t_ms=row * 1000.0, heading_deg=heading,
                                               embedding_row=row))
            (d / "refs" / f"{rid}.jpg").write_bytes(JPEG)
            vectors[row] = axis(ax)
        np.save(d / "embeddings.npy", vectors)
        m.embedding = EmbeddingInfo(model=MODEL, dim=DIM, file=f"{build_rel}/embeddings.npy")
    (d / "map.json").write_text(m.model_dump_json())
    (root / m.map_id / "map.json").write_text(m.model_dump_json())
    return m.map_id


class Places:
    """Stand-in for the inference host: the next answer is set by the test."""

    def __init__(self):
        self.answer: np.ndarray | Exception = axis(2)[None]
        self.calls = []

    def __call__(self, images, model):
        self.calls.append((len(images), model))
        if isinstance(self.answer, Exception):
            raise self.answer
        return np.atleast_2d(self.answer).astype(np.float32)


@pytest.fixture
def nav(tmp_path):
    places = Places()
    map_id = publish(tmp_path / "maps")
    app = create_nav_app(MapCache(tmp_path / "maps", reload_s=0), place_fn=places, api_token=None, cors_origins=[],
                         cors_origin_regex=None, frontend_url=None, thresholds=THRESHOLDS,
                         infer_health_fn=lambda: {"ok": True, "where": "stub"})
    with TestClient(app) as c:
        yield c, places, map_id


def localize(c, map_id, n=1, **form):
    files = [("images", (f"f{i}.jpg", JPEG, "image/jpeg")) for i in range(n)]
    return c.post(f"{API}/maps/{map_id}/localize", files=files, data=form)


def test_a_clear_match_is_confirmed_with_its_references(nav):
    c, places, map_id = nav
    places.answer = unit(0.1, 0, 1, 0.3)  # mostly n2's "at" view, a bit of its approach and of n1
    r = localize(c, map_id, expected="n3", heading_deg="100")
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["status"] == "confirmed" and j["best"] == "n2" and j["frames"] == 1 and j["model"] == MODEL
    assert [n["node"] for n in j["candidates"]] == ["n2", "n1", "n3"]
    best = j["candidates"][0]
    assert best["name"] == "N2" and [x["id"] for x in best["refs"]] == ["r3", "r4"]
    assert best["refs"][0]["image"] == f"builds/{BUILD.build_id}/refs/r3.jpg" and best["refs"][0]["role"] == "at"
    # compass-referenced map (heading_zero_abs_deg 0): the phone faces 100, the reference was taken facing 90
    assert j["heading_map_deg"] == 100 and best["refs"][0]["heading_diff_deg"] == 10
    assert j["expected_rank"] == 3 and j["expected_score"] == 0
    assert j["margin"] > 0.5 and set(j["took_ms"]) == {"embed", "match", "total"}
    assert places.calls == [(1, MODEL)]


def test_close_or_weak_matches_are_not_confirmed(nav):
    c, places, map_id = nav
    places.answer = unit(0, 1, 1)  # as much n1 as n2
    j = localize(c, map_id).json()
    assert j["status"] == "uncertain" and "margin" in j["reason"]
    places.answer = unit(0, 0, 0, 0, 0.5, 0, 0, 1)  # mostly nothing the map knows
    j = localize(c, map_id).json()
    assert j["status"] == "lost" and j["best"] == "n3"
    places.answer = unit(0, 0, 1, 0, 0, 0, 0, 1)  # 0.71 n2, nothing else close
    assert localize(c, map_id).json()["status"] == "confirmed"
    places.answer = unit(0, 0, 1, 0, 0, 0, 0, 1.5)  # 0.55: between lost and confirmed
    assert localize(c, map_id).json()["status"] == "uncertain"


def test_a_jump_away_from_the_last_confirmed_node_stays_uncertain(nav):
    c, places, map_id = nav
    places.answer = axis(4)[None]  # n3
    assert localize(c, map_id, previous="n2").json()["status"] == "confirmed"  # n2 -> n3 is an edge
    j = localize(c, map_id, previous="n1").json()  # e3 (n1 - n3) is not in this map
    assert j["status"] == "uncertain" and "N1" in j["reason"] and j["previous"] == "n1"


def test_a_burst_scores_the_mean_of_its_frames(nav):
    c, places, map_id = nav
    places.answer = np.stack([axis(2), unit(0, 0, 1, 0, 0, 0, 0, 1)])
    j = localize(c, map_id, n=2).json()
    assert j["frames"] == 2 and places.calls[-1] == (2, MODEL)
    assert j["best"] == "n2" and j["candidates"][0]["score"] == pytest.approx((1 + 0.7071) / 2, abs=1e-3)


def test_inference_problems_are_reported_not_hidden(nav):
    c, places, map_id = nav
    places.answer = InferenceUnavailable("inference host unreachable or busy")
    r = localize(c, map_id)
    assert r.status_code == 503 and r.headers["retry-after"] == "2" and "unreachable" in r.json()["detail"]
    places.answer = InferenceError("POST /infer/v1/place: 409 place model here is 'x'")
    r = localize(c, map_id)
    assert r.status_code == 502 and "409" in r.json()["detail"]


def test_bad_requests(nav):
    c, places, map_id = nav
    assert localize(c, map_id, n=5).status_code == 422
    assert localize(c, "nope").status_code == 404
    assert c.post(f"{API}/maps/{map_id}/localize", data={"top_k": "3"}).status_code == 422
    assert places.calls == []


def lines(capsys) -> list[dict]:
    """nav-api's decision log (nav/nav_api/log.py) since the last call: the JSON lines on stdout."""
    return [json.loads(x) for x in capsys.readouterr().out.splitlines() if x.startswith("{")]


def test_each_localization_is_one_structured_log_line(nav, capsys, monkeypatch):
    c, places, map_id = nav
    monkeypatch.setattr("nav.config.COMMIT", "abc123")
    places.answer = unit(0.1, 0, 1, 0.3)
    r = c.post(f"{API}/maps/{map_id}/localize", files=[("images", ("f.jpg", JPEG, "image/jpeg"))],
               data={"expected": "n3"}, headers={"X-Client-Id": "c2cd8407-aaaa", "X-Session-Id": "s1"})
    assert r.status_code == 200
    [e] = lines(capsys)
    assert e["kind"] == "nav_localize" and e["service"] == "nav-api" and e["severity"] == "INFO"
    assert e["mapId"] == map_id and e["status"] == "confirmed" and e["best"] == "n2" and e["bestName"] == "N2"
    assert e["clientId"] == "c2cd8407-aaaa" and e["sessionId"] == "s1" and "requestId" not in e
    assert e["commit"] == "abc123" and e["version"] and e["expectedRank"] == 3 and set(e["tookMs"]) >= {"total"}
    assert e["message"].startswith(f"[nav-api] localize {map_id} · confirmed N2 ")
    assert " · expected n3 #3 · " in e["message"] and e["message"].endswith(" ms")

    places.answer = unit(0, 0, 0, 0, 0.5, 0, 0, 1)
    [e] = (localize(c, map_id), lines(capsys))[1]
    assert e["status"] == "lost" and e["severity"] == "WARNING" and "lost N3" in e["message"]
    assert "clientId" not in e


def test_failed_localizations_and_routes_are_logged_too(nav, capsys):
    c, places, map_id = nav
    places.answer = InferenceUnavailable("inference host unreachable or busy")
    assert localize(c, map_id).status_code == 503
    [e] = lines(capsys)
    assert e["kind"] == "nav_localize" and e["severity"] == "ERROR" and e["httpStatus"] == 503
    assert e["message"].startswith(f"[nav-api] localize {map_id} · error 503: inference host unavailable")
    assert localize(c, "nope").status_code == 404
    [e] = lines(capsys)
    assert e["severity"] == "WARNING" and e["httpStatus"] == 404 and e["mapId"] == "nope"
    assert localize(c, map_id, n=5).status_code == 422
    assert lines(capsys)[0]["frames"] == 5

    assert c.post(f"{API}/maps/{map_id}/route", json={"start": "n1", "goal": "n9"}).status_code == 404
    [e] = lines(capsys)
    assert e["kind"] == "nav_route" and e["severity"] == "WARNING" and e["goal"] == "n9" and "no node" in e["error"]


def test_each_route_is_one_structured_log_line(nav, capsys):
    c, _, map_id = nav
    c.post(f"{API}/maps/{map_id}/route", json={"start": "n1", "goal": "n3"})
    [e] = lines(capsys)
    assert e["kind"] == "nav_route" and e["found"] is False and e["hops"] == 0 and e["severity"] == "WARNING"
    assert e["message"] == f"[nav-api] route {map_id} · N1 → N3 (verified) · no route" and "lengthM" not in e
    c.post(f"{API}/maps/{map_id}/route", json={"start": "n1", "goal": "n3", "trust": "observed"})
    [e] = lines(capsys)
    assert e["found"] and e["hops"] == 2 and e["severity"] == "INFO" and e["lengthM"] > 0
    assert e["message"].endswith(f"(observed) · 2 hops, {e['lengthM']:.1f} m")
    c.get(f"{API}/maps/{map_id}")
    c.get(f"{API}/health")
    assert lines(capsys) == []  # reads aren't decisions


def test_a_map_without_references_is_lost_without_calling_the_host(tmp_path):
    places = Places()
    map_id = publish(tmp_path / "maps", refs=False)
    with TestClient(create_nav_app(MapCache(tmp_path / "maps"), place_fn=places, api_token=None,
                                   thresholds=THRESHOLDS)) as c:
        j = localize(c, map_id).json()
    assert j["status"] == "lost" and "no reference images" in j["reason"] and places.calls == []


def test_maps_routes_and_health(nav):
    c, _, map_id = nav
    assert [m["map_id"] for m in c.get(f"{API}/maps").json()] == [map_id]
    assert c.get(f"{API}/maps/{map_id}").json()["nodes"][1]["name"] == "N2"
    assert c.get(f"{API}/schema/map").json()["title"] == "PlaceMap"
    assert c.get(f"{API}/health").json() == {"ok": True, "service": "nav-api", "schema_version": 1}
    deep = c.get(f"{API}/health", params={"deep": 1}).json()
    assert deep["maps"] == 1 and deep["inference"]["where"] == "stub"

    r = c.post(f"{API}/maps/{map_id}/route", json={"start": "n1", "goal": "n3"}).json()
    assert r["found"] is False and r["hops"] == [] and r["trust"] == "verified"  # nothing verified yet
    r = c.post(f"{API}/maps/{map_id}/route", json={"start": "n1", "goal": "n3", "trust": "observed"}).json()
    assert r["found"] and [(h["edge"], h["forward"]) for h in r["hops"]] == [("e1", True), ("e2", False)]
    assert r["hops"][1]["bearing_deg"] == 180 and r["hops"][1]["instruction"] == "back e2"
    assert r["length_m"] == pytest.approx(sum(h["length_m"] for h in r["hops"]), abs=0.1)
    assert c.post(f"{API}/maps/{map_id}/route", json={"start": "n1", "goal": "n9"}).status_code == 404


def test_only_the_published_builds_images_and_embeddings_are_served(nav, tmp_path):
    c, _, map_id = nav
    build = f"builds/{BUILD.build_id}"
    (tmp_path / "maps" / map_id / "builds" / "20250101T000000Z" / "refs").mkdir(parents=True)
    (tmp_path / "maps" / map_id / "builds" / "20250101T000000Z" / "refs" / "old.jpg").write_bytes(JPEG)
    assert c.get(f"{API}/maps/{map_id}/files/{build}/refs/r1.jpg").content == JPEG
    assert c.get(f"{API}/maps/{map_id}/files/{build}/embeddings.npy").status_code == 200
    for path in (f"{build}/job/claude.json", f"{build}/map.json", "builds/20250101T000000Z/refs/old.jpg",
                 f"{build}/refs/../job/claude.json", f"{build}/refs/missing.jpg", "review.json"):
        assert c.get(f"{API}/maps/{map_id}/files/{path}").status_code == 404, path


def test_a_newly_published_map_is_picked_up(tmp_path):
    now = [0.0]
    cache = MapCache(tmp_path / "maps", reload_s=30, clock=lambda: now[0])
    map_id = publish(tmp_path / "maps")
    assert cache.get(map_id).nodes[1].name == "N2"
    path = tmp_path / "maps" / map_id / "map.json"
    m = json.loads(path.read_text())
    m["nodes"][1]["name"] = "Hall"
    path.write_text(json.dumps(m))
    os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 10**9))
    assert cache.get(map_id).nodes[1].name == "N2"  # checked less than reload_s ago
    now[0] = 31
    assert cache.get(map_id).nodes[1].name == "Hall" and len(cache.index(map_id).refs) == len(REFS)


def test_a_distance_measured_on_site_reaches_the_live_api(tmp_path):
    """The measurement goes into the review on map-api, which republishes; nav-api serves the rescaled map."""
    map_id = publish(tmp_path / "maps")

    def live() -> TestClient:  # a fresh cache: reloading on a newer map.json is tested above
        return TestClient(create_nav_app(MapCache(tmp_path / "maps"), place_fn=Places(), api_token=None,
                                         cors_origins=[], cors_origin_regex=None, frontend_url=None,
                                         thresholds=THRESHOLDS))

    route = {"start": "n1", "goal": "n3", "trust": "observed"}
    with live() as c:
        before = c.post(f"{API}/maps/{map_id}/route", json=route).json()["length_m"]
    maps = create_app(RecordingStore(tmp_path / "recordings"), MapStore(tmp_path / "maps"), api_token=None,
                      cache_dir=tmp_path / "cache", web_dir=None)
    with TestClient(maps) as c:
        r = c.put(f"{API}/maps/{map_id}/review", json={"distances": [{"between": ["n1", "n2"], "meters": 7}]})
        assert r.status_code == 200 and r.json()["frame"]["distance_scale"] == pytest.approx(0.7)  # e1 walked 10 m
        assert c.get(f"{API}/maps/{map_id}/review").json()["distances"][0]["meters"] == 7
    with live() as c:
        m = c.get(f"{API}/maps/{map_id}").json()
        assert next(e for e in m["edges"] if e["id"] == "e1")["distance_m"] == pytest.approx(7)
        assert m["frame"]["measured_distances"][0]["residual_m"] == pytest.approx(0, abs=0.01)
        after = c.post(f"{API}/maps/{map_id}/route", json=route).json()["length_m"]
        assert after == pytest.approx(0.7 * before, abs=0.1)

def test_token_cors_and_what_moved_to_map_api(tmp_path):
    publish(tmp_path / "maps")
    app = create_nav_app(MapCache(tmp_path / "maps"), place_fn=Places(), api_token="s3cret",
                         cors_origins=["https://map-api.example"], cors_origin_regex=None,
                         frontend_url="https://map-api.example", thresholds=THRESHOLDS)
    with TestClient(app) as c:
        assert c.get(f"{API}/health").status_code == 200
        r = c.get(f"{API}/maps", headers={"Origin": "https://map-api.example"})
        assert r.status_code == 401 and r.headers["access-control-allow-origin"] == "https://map-api.example"
        assert c.get(f"{API}/maps", headers={"Authorization": "Bearer s3cret"}).status_code == 200
        assert c.get(f"{API}/maps", params={"token": "s3cret"}).status_code == 200
        pre = c.options(f"{API}/maps/t/localize", headers={"Origin": "https://map-api.example",
                                                           "Access-Control-Request-Method": "POST",
                                                           "Access-Control-Request-Headers": "authorization"})
        assert pre.status_code == 200
        r = c.get(f"{API}/recordings", headers={"Authorization": "Bearer s3cret"})
        assert r.status_code == 404 and "map-api (https://map-api.example)" in r.json()["detail"]
        for path in ("/", "/viz/", "/dash", "/loc/"):
            r = c.get(path, follow_redirects=False)
            assert r.status_code == 307 and r.headers["location"].startswith("https://map-api.example/"), path
