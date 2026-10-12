"""The visual memory (nav/mapping/memory.py) and localization against it (nav/navigation/localize.py MemoryIndex)."""

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from nav.mapping.build import republish
from nav.nav_api.app import create_nav_app
from nav.nav_api.maps import MapCache
from nav.infer.errors import InferenceUnavailable
from nav.navigation.localize import MemoryIndex, PoseParams
from nav.schemas.map import MemoryFrame, MemoryInfo, PlaceMap
from test_nav_api import API, DIM, JPEG, MODEL, THRESHOLDS, Places, axis, localize, publish
from test_navigation import BUILD

# memory frame -> (x, y, embedding axis). The lshape map's nodes: n1 (0, 0), n2 (10, 0), n3 (10, -10).
MEMORY = {"m0001": (0.5, 0.0, 5), "m0002": (9.0, 0.5, 6), "m0003": (5.0, -1.0, 7)}


def add_memory(root, map_id: str, scale: float = 1.0, refs: bool = True, geometry: bool = False) -> None:
    """Give the published lshape map a visual memory (MEMORY); `scale`: as a review with a measured distance would
    leave the published map (positions x scale), the memory staying in the build's frame. `geometry`: its frames
    posed by a 3D model (geometry.npz: two observations per frame), for live poses."""
    build_rel = f"builds/{BUILD.build_id}"
    d = root / map_id / build_rel
    (d / "memory").mkdir(parents=True, exist_ok=True)
    frames, vectors = [], np.zeros((len(MEMORY), DIM), np.float32)
    for row, (fid, (x, y, ax)) in enumerate(MEMORY.items()):
        frames.append(MemoryFrame(id=fid, rec_id="r1", t_ms=row * 5000.0, image=f"{build_rel}/memory/{fid}.jpg",
                                  x_m=x, y_m=y, heading_deg=90.0, posed=geometry,
                                  row=row if geometry else None).model_dump())
        (d / "memory" / f"{fid}.jpg").write_bytes(JPEG)
        vectors[row] = axis(ax)
    (d / "memory.json").write_text(json.dumps({"frames": frames}))
    np.save(d / "memory.npy", vectors)
    if geometry:
        n = len(MEMORY)
        np.savez(d / "geometry.npz", obs_frame=np.repeat(np.arange(n), 2).astype(np.int32),
                 obs_xy=np.arange(4 * n, dtype=np.float32).reshape(2 * n, 2), obs_point=np.arange(2 * n, dtype=np.int32),
                 points=np.arange(6 * n, dtype=np.float32).reshape(2 * n, 3))
    for path in (d / "map.json", root / map_id / "map.json"):
        m = PlaceMap.model_validate_json(path.read_text())
        m.memory = MemoryInfo(model=MODEL, dim=DIM, frames=len(MEMORY), file=f"{build_rel}/memory.json",
                              embeddings=f"{build_rel}/memory.npy",
                              geometry=f"{build_rel}/geometry.npz" if geometry else None,
                              posed=len(MEMORY) if geometry else 0)
        if not refs:
            m.references, m.embedding = [], None
        if path.parent == root / map_id and scale != 1.0:
            for n in m.nodes:
                n.position.x_m, n.position.y_m = n.position.x_m * scale, n.position.y_m * scale
            m.frame.distance_scale = scale
        path.write_text(m.model_dump_json())


def client(root, pose_fn=None, **kw):
    places = Places()
    app = create_nav_app(MapCache(root, reload_s=0, **kw), place_fn=places, api_token=None, cors_origins=[],
                         cors_origin_regex=None, frontend_url=None, thresholds=THRESHOLDS, capture=None,
                         pose_fn=pose_fn, pose_params=PoseParams(k=2, confirm_inliers=40))
    return TestClient(app), places


class Poser:
    """Stand-in for the inference host's live pose: one answer for every live frame (the last frame's is best)."""

    def __init__(self, centre=(9.5, 0.2), inliers: int = 120, fail: Exception | None = None):
        self.centre, self.inliers, self.fail, self.calls = centre, inliers, fail, []

    def __call__(self, images, refs):
        self.calls.append((len(images), [k for k, _, _ in refs], [obs().shape for _, _, obs in refs]))
        if self.fail:
            raise self.fail
        out = [None] * (len(images) - 1)
        return out + [{"inliers": self.inliers, "matches": 300, "centre": [*self.centre, 0.1], "heading_deg": 91.0,
                       "focal": 900.0, "R": [1, 0, 0, 0, 1, 0, 0, 0, 1], "t": [0, 0, 0], "size": [576, 1024],
                       "per_ref": [self.inliers, 0]}]


def test_a_frame_like_a_memory_frame_gives_its_position_and_the_node_whose_region_it_is_in(tmp_path):
    map_id = publish(tmp_path / "maps")
    add_memory(tmp_path / "maps", map_id)
    c, places = client(tmp_path / "maps")
    with c:
        places.answer = axis(6)[None]  # m0002, 1 m from n2: no reference looks like it
        j = localize(c, map_id).json()
        assert j["best"] == "n2" and j["status"] == "confirmed"
        assert j["position"] == {"x_m": 9.0, "y_m": 0.5, "heading_deg": 90.0, "frame": "m0002", "rec_id": "r1",
                                 "t_ms": 5000.0, "score": 1.0, "node": "n2"}
        assert j["pose"] is None  # no 3D model
        top = j["candidates"][0]["refs"][0]
        assert top["role"] == "memory" and top["id"] == "m0002"
        assert top["image"] == f"builds/{BUILD.build_id}/memory/m0002.jpg"
        assert c.get(f"{API}/maps/{map_id}/files/{top['image']}").content == JPEG  # nav-api serves memory frames

        places.answer = axis(7)[None]  # m0003, more than 5 m from every node: a position, but no node's
        j = localize(c, map_id).json()
        assert j["position"]["frame"] == "m0003" and j["position"]["node"] is None and j["status"] == "lost"

        places.answer = axis(2)[None]  # a reference still counts as before (n2's "at" image)
        assert localize(c, map_id).json()["best"] == "n2"


def test_regions_follow_the_published_map_not_the_build(tmp_path):
    map_id = publish(tmp_path / "maps")
    add_memory(tmp_path / "maps", map_id, scale=2.0)  # a review measured the place twice as big as dead reckoning
    c, places = client(tmp_path / "maps", memory_radius_m=3.0)
    with c:
        places.answer = axis(6)[None]
        pos = localize(c, map_id).json()["position"]
        assert (pos["x_m"], pos["y_m"], pos["node"]) == (18.0, 1.0, "n2")  # n2 is now at (20, 0): 2.2 m away


def test_a_map_with_only_a_memory_localizes(tmp_path):
    map_id = publish(tmp_path / "maps")
    add_memory(tmp_path / "maps", map_id, refs=False)
    c, places = client(tmp_path / "maps")
    with c:
        places.answer = axis(5)[None]
        j = localize(c, map_id).json()
        assert j["best"] == "n1" and j["model"] == MODEL and places.calls == [(1, MODEL)]


def test_a_walk_left_out_leaves_its_frames_out_and_unplaced_walks_use_their_visits():
    from test_navigation import lshape
    from nav.mapping.build import finish_map

    draft, cset, tracks = lshape()
    m = finish_map(draft, cset, BUILD, "builds/b", tracks)
    frames = [MemoryFrame(id="m1", rec_id="r1", t_ms=10_500, image="x", x_m=None, y_m=None, heading_deg=None),
              MemoryFrame(id="m2", rec_id="r1", t_ms=15_000, image="x", x_m=None, y_m=None, heading_deg=None),
              MemoryFrame(id="m3", rec_id="r2", t_ms=0, image="x", x_m=0.0, y_m=0.0, heading_deg=None)]
    v = np.eye(3, DIM, dtype=np.float32)
    idx = MemoryIndex.build(m, frames, v)
    assert idx.nodes == ["n2", None, "n1"]  # m1: r1 was at n2 at 10 s; m2: 5 s from any visit
    idx = MemoryIndex.build(m, frames, v, exclude_rec="r1")
    assert [f.id for f in idx.frames] == ["m3"] and idx.vectors.shape == (1, DIM) and idx.vectors[0, 2] == 1


def test_republishing_keeps_the_memory(tmp_path):
    map_id = publish(tmp_path / "maps")
    add_memory(tmp_path / "maps", map_id)
    out = republish(tmp_path / "maps" / map_id)
    assert out.memory and out.memory.frames == 3


def test_a_pose_confirms_the_node_it_is_in_over_what_the_embeddings_say(tmp_path):
    map_id = publish(tmp_path / "maps")
    add_memory(tmp_path / "maps", map_id, geometry=True)
    poser = Poser(centre=(9.5, 0.2), inliers=120)
    c, places = client(tmp_path / "maps", pose_fn=poser)
    with c:
        places.answer = np.stack([axis(5), axis(5)])  # looks like m0001, at n1 ...
        j = localize(c, map_id, n=2).json()
        assert j["candidates"][0]["node"] == "n1"  # ... so the embeddings lead with n1 (the belief still sees that)
        assert j["best"] == "n2" and j["status"] == "confirmed" and j["reason"].startswith("pose: 120 inliers at")
        assert j["pose"] == {"x_m": 9.5, "y_m": 0.2, "heading_deg": 91.0, "inliers": 120, "matches": 300,
                             "focal_px": 900.0, "node": "n2", "frames": j["pose"]["frames"]}
        n, keys, shapes = poser.calls[0]
        assert n == 2 and j["pose"]["frames"][0] == "m0001" and len(keys) == 2  # the best posed frames
        assert keys == [f"{map_id}/{BUILD.build_id}/{f}" for f in j["pose"]["frames"]]
        assert shapes == [(2, 5), (2, 5)] and "pose" in j["took_ms"]


def test_a_weak_pose_or_one_between_nodes_confirms_nothing(tmp_path):
    map_id = publish(tmp_path / "maps")
    add_memory(tmp_path / "maps", map_id, geometry=True)
    c, places = client(tmp_path / "maps", pose_fn=Poser(centre=(9.5, 0.2), inliers=20))
    with c:
        places.answer = axis(5)[None]
        j = localize(c, map_id).json()
        assert j["pose"]["inliers"] == 20 and j["best"] == "n1" and not j["reason"].startswith("pose")
    c, places = client(tmp_path / "maps", pose_fn=Poser(centre=(5.0, -5.0), inliers=200))
    with c:
        places.answer = axis(6)[None]
        j = localize(c, map_id).json()
        assert j["pose"]["node"] is None and j["status"] == "uncertain" and j["reason"] == "pose between nodes"


def test_a_pose_follows_the_review_scale(tmp_path):
    map_id = publish(tmp_path / "maps")
    add_memory(tmp_path / "maps", map_id, scale=2.0, geometry=True)
    c, places = client(tmp_path / "maps", pose_fn=Poser(centre=(9.5, 0.2)), memory_radius_m=3.0)
    with c:
        places.answer = axis(6)[None]
        j = localize(c, map_id).json()
        assert (j["pose"]["x_m"], j["pose"]["y_m"], j["pose"]["node"]) == (19.0, 0.4, "n2")


def test_without_the_pose_host_the_answer_has_no_pose(tmp_path):
    map_id = publish(tmp_path / "maps")
    add_memory(tmp_path / "maps", map_id, geometry=True)
    c, places = client(tmp_path / "maps", pose_fn=Poser(fail=InferenceUnavailable("host down")))
    with c:
        places.answer = axis(6)[None]
        j = localize(c, map_id).json()
        assert j["best"] == "n2" and j["pose"] is None and j["took_ms"]["pose_failed"] == 1


def unit_of(*axes: int) -> np.ndarray:
    v = sum(axis(a) for a in axes)
    return v / np.linalg.norm(v)
