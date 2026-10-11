"""Episodes stitched from pieces of walks (nav/navigation/episodes.py), on the Itnig map of 2026-10-10
(tests/fixtures/itnig) with its four walks' visits (the map's evidence times) and synthetic frames: one every
667 ms, embedding = the node the walk was last at, steps every 500 ms."""

from pathlib import Path

import numpy as np
import pytest

from nav.navigation.episodes import MIN_PIECE_S, reward, sample_episodes, stitch
from nav.navigation.pdr import walk_visits
from nav.navigation.replay import OrchestratorResult, Track
from nav.navigation.route import hops
from nav.schemas.map import PlaceMap
from nav.schemas.navigation import LiveMotion

PMAP = PlaceMap.model_validate_json((Path(__file__).parent / "fixtures" / "itnig" / "map.json").read_text())
NODES = [n.id for n in PMAP.nodes]


def track(rec_id: str) -> Track:
    visits = walk_visits(PMAP, rec_id)
    t = np.arange(0.0, visits[-1][0] + 10_000, 667.0)
    at = [max((v for v in visits if v[0] <= x), default=visits[0])[1] for x in t]
    emb = np.eye(len(NODES), dtype=np.float32)[[NODES.index(n) for n in at]]
    return Track(rec_id, t, emb, [LiveMotion(step_count=int(x // 500)) for x in t], visits)


TRACKS = [track(r) for r in PMAP.recordings]


@pytest.mark.parametrize("seed", range(20))
def test_stitched_episodes_are_routes_along_the_map_with_frames_and_steps_that_go_on(seed):
    walkable = {(h.source, h.target) for e in PMAP.edges for h in hops(e)}
    episodes = sample_episodes(PMAP, TRACKS, 10, seed)
    assert len(episodes) == 10
    for ep in episodes:
        nodes = [n for n, _ in ep.truth]
        times = [t for _, t in ep.truth]
        assert len(nodes) >= 3 and ep.goal == nodes[-1] and nodes.count(ep.goal) == 1
        assert all(pair in walkable for pair in zip(nodes, nodes[1:]))
        assert times[0] == -np.inf and all(b - a >= MIN_PIECE_S * 1000 for a, b in zip(times[1:], times[2:]))
        assert np.all(np.diff(ep.t_ms) > 0) and ep.t_ms[0] >= 0
        steps = [m.step_count for m in ep.motions]
        assert steps[0] == 0 and all(b >= a for a, b in zip(steps, steps[1:]))
        assert len(ep.t_ms) == len(ep.emb) == len(ep.motions) == len(ep.recs) == len(ep.keys)
        # each frame shows where the user was: the node of the piece it came from (the walk's last visit)
        for t, e in zip(ep.t_ms, ep.emb):
            assert NODES[int(np.argmax(e))] == [n for n, t0 in ep.truth if t0 <= t][-1]


def test_same_seed_same_episodes_and_a_walk_left_out_is_not_in_them():
    a, b = sample_episodes(PMAP, TRACKS, 5, 7), sample_episodes(PMAP, TRACKS, 5, 7)
    assert [e.keys for e in a] == [e.keys for e in b]
    left_out = PMAP.recordings[0]
    assert all(left_out not in e.recs for e in sample_episodes(PMAP, TRACKS, 20, 3, exclude_rec=left_out))


def test_stitch_joins_pieces_of_two_walks():
    from nav.navigation.episodes import pieces

    by_rec = {t.rec_id: t for t in TRACKS}
    ps = pieces(TRACKS)
    first = next(p for p in ps if (p.source, p.target) == ("n1", "n2"))
    second = next(p for p in ps if p.source == "n2" and p.rec_id != first.rec_id and p.target != "n1")
    ep = stitch(by_rec, [first, second])
    assert [n for n, _ in ep.truth] == ["n1", "n2", second.target]
    assert ep.truth[1][1] == first.t1 - first.t0 and set(ep.recs) == {first.rec_id, second.rec_id}


def test_reward_counts_nodes_in_time_and_charges_wrong_moves_and_restarts():
    nodes = [{"node": "n2", "delay_s": 0.0}, {"node": "n3", "delay_s": 4.0}, {"node": "n4", "delay_s": None}]
    ok = OrchestratorResult([], nodes, 10, "n4")
    assert reward(ok) == pytest.approx((1 + 0.8) / 3)
    bad = OrchestratorResult([{"kind": "reached", "ok": False}, {"kind": "lost", "ok": True}], nodes, 10, "n4")
    assert reward(bad) == pytest.approx((1 + 0.8 - 3.0 - 0.3) / 3)


@pytest.mark.parametrize("kind", ["belief", "votes"])
def test_with_a_clear_camera_the_loops_reach_nodes_and_never_move_the_user_wrong(kind):
    """Each node has one reference whose embedding is its one-hot; frames are their node's one-hot plus seeded noise,
    so the node the user is at always leads by a clear margin."""
    from types import SimpleNamespace

    from nav.navigation.episodes import evaluate
    from nav.navigation.localize import RefIndex
    from nav.schemas.navigation import Thresholds

    rng = np.random.default_rng(0)
    refs = [SimpleNamespace(node=n, id=f"r{n}", role="at", edge=None, toward=None, heading_deg=None, image="")
            for n in NODES]
    index = RefIndex(PMAP, np.eye(len(NODES), dtype=np.float32), refs)
    episodes = sample_episodes(PMAP, TRACKS, 15, 1)
    for ep in episodes:
        noisy = ep.emb * 0.6 + rng.random(ep.emb.shape).astype(np.float32) * 0.1
        ep.emb = noisy / np.linalg.norm(noisy, axis=1, keepdims=True)
    th = Thresholds(confirm_score=0.45, confirm_margin=0.05, lost_score=0.35, gate_slack_m=100.0)
    s = evaluate(PMAP, index, episodes, th, kind, None)
    assert s.wrong == 0 and s.reached >= 0.8 * s.nodes


@pytest.mark.parametrize("rec_id", PMAP.recordings)
def test_a_walk_replayed_stops_soon_after_its_goal(rec_id):
    """Itnig walk ddb1 reached its route's goal (Main entrance) at 10 s and went on for 160 s; the replay kept
    sending those frames with the user still "at" the goal, so a loop that missed the goal and then followed the
    walker to the Drinks area was scored wrong twice (2026-10-11)."""
    from nav.navigation.replay import GOAL_TAIL_S, walk_episode

    tr = track(rec_id)
    ep = walk_episode(PMAP, tr)
    goal_t = ep.truth[-1][1]
    after = [t for t, _ in tr.visits if t > goal_t]
    assert ep.t_ms[-1] < min(after[0] if after else np.inf, goal_t + GOAL_TAIL_S * 1000)
    assert ep.t_ms[-1] >= goal_t  # but long enough to arrive
