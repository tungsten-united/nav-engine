"""map-api rebuilds a place's map when one of its walks finalizes; walks that finalize during a build queue one more."""

import json
import threading
import time
from types import SimpleNamespace

from fastapi.testclient import TestClient
from test_recorder import API, create, finish, upload, webm  # noqa: F401 (webm is a fixture)

from nav.map_api.app import create_app
from nav.mapping.store import MapStore
from nav.recorder.store import RecordingStore


class FakeBuild:
    """Stands in for build_map: records each call, and holds it until `release` is set."""

    def __init__(self, hold: bool = False):
        self.calls = []
        self.release = threading.Event()
        if not hold:
            self.release.set()

    def __call__(self, place, *, recordings_root, build_id, **kw):
        self.calls.append((place, build_id, sorted(p.name for p in recordings_root.iterdir())))
        self.release.wait(10)
        return SimpleNamespace(map=None, candidates=SimpleNamespace(candidates=[]))


def statuses(tmp_path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted((tmp_path / "maps").glob("*/builds/*/status.json"))]


def wait_for(cond, timeout=10.0):
    end = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < end, "timed out"
        time.sleep(0.05)


def app(tmp_path, fake, auto_build=True):
    return create_app(RecordingStore(tmp_path / "recordings"), MapStore(tmp_path / "maps"), api_token=None,
                      web_dir=None, cache_dir=tmp_path / "cache", auto_build=auto_build, build_map_fn=fake)


def walk(c, webm) -> str:
    rec_id = create(c)
    upload(c, rec_id, webm)
    assert finish(c, rec_id)["status"] == "complete"
    return rec_id


def test_a_finished_walk_rebuilds_its_places_map(tmp_path, webm):
    fake = FakeBuild()
    with TestClient(app(tmp_path, fake)) as c:
        rec_id = walk(c, webm)
        wait_for(lambda: statuses(tmp_path) and statuses(tmp_path)[0]["status"] == "done")
        [s] = statuses(tmp_path)
        assert s["map_id"] == "test-floor" and s["trigger"] == f"auto: {rec_id} finalized"
        assert [(place, recs) for place, _, recs in fake.calls] == [("Test floor", [rec_id])]
        # a walk finalized again (late uploads) rebuilds again
        c.post(f"{API}/recordings/{rec_id}/finalize", params={"wait": True})
        wait_for(lambda: len(statuses(tmp_path)) == 2 and statuses(tmp_path)[1]["status"] == "done")


def test_walks_finalized_during_a_build_queue_one_more(tmp_path, webm):
    fake = FakeBuild(hold=True)
    with TestClient(app(tmp_path, fake)) as c:
        first = walk(c, webm)
        wait_for(lambda: len(fake.calls) == 1)
        second, third = walk(c, webm), walk(c, webm)  # finalize while the first build runs
        assert c.post(f"{API}/maps/build", json={"place": "test floor"}).status_code == 409
        assert len(fake.calls) == 1
        fake.release.set()
        wait_for(lambda: len(statuses(tmp_path)) == 2 and all(s["status"] == "done" for s in statuses(tmp_path)))
        time.sleep(0.3)
        assert len(fake.calls) == 2  # one build for both queued walks, then nothing
        assert fake.calls[1][2] == sorted([first, second, third])
        assert statuses(tmp_path)[1]["trigger"] == f"auto: {second}, {third} finalized"


def test_off_and_unusable_walks_build_nothing(tmp_path, webm):
    fake = FakeBuild()
    with TestClient(app(tmp_path, fake, auto_build=False)) as c:
        walk(c, webm)
    with TestClient(app(tmp_path / "b", fake)) as c:
        rec_id = create(c)
        r = c.post(f"{API}/recordings/{rec_id}/finish",
                   json={"duration_ms": 3000, "video_chunks": 3, "sample_batches": 0, "voice_clips": 0})
        assert r.status_code == 202 and c.get(f"{API}/recordings/{rec_id}").json()["video"] is None
    time.sleep(0.3)
    assert fake.calls == [] and statuses(tmp_path) == [] and statuses(tmp_path / "b") == []


def test_a_published_build_warms_the_hosts_pose_cache_at_once(tmp_path, webm):
    class Warmer:
        kicks = 0

        def kick(self):
            self.kicks += 1

    published_map = SimpleNamespace(map=object(), candidates=SimpleNamespace(candidates=[]))
    warmer = Warmer()
    a = create_app(RecordingStore(tmp_path / "recordings"), MapStore(tmp_path / "maps"), api_token=None, web_dir=None,
                   cache_dir=tmp_path / "cache", auto_build=True, pose_warm=warmer,
                   build_map_fn=lambda place, **kw: published_map)
    with TestClient(a) as c:
        walk(c, webm)
        wait_for(lambda: statuses(tmp_path) and statuses(tmp_path)[0]["status"] == "done")
    assert warmer.kicks == 1 and a.state.pose_warmer is warmer
