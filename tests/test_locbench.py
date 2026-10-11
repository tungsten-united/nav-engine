"""The localization benchmark (nav/navigation/locbench.py): what it counts, and captured sessions as queries."""

from datetime import datetime, timezone

import numpy as np

from nav.nav_api.capture import Capture
from nav.navigation.captured import sessions
from nav.navigation.locbench import Query, Row, score, session_queries
from nav.schemas.navigation import LivePosition, Localization, NodeMatch, Thresholds
from test_navigation import BUILD, lshape
from nav.mapping.build import finish_map

TH = Thresholds(confirm_score=0.6, confirm_margin=0.05, lost_score=0.45)


def lmap():
    draft, cset, tracks = lshape()
    return finish_map(draft, cset, BUILD, "builds/b", tracks)  # n1 (0, 0), n2 (10, 0), n3 (10, -10)


def answer(best: str, lead: float, x: float | None = None) -> Localization:
    others = [n for n in ("n1", "n2", "n3") if n != best]
    cands = [NodeMatch(node=best, name=best, score=0.6, refs=[])] + \
            [NodeMatch(node=n, name=n, score=0.6 - lead, refs=[]) for n in others]
    pos = None if x is None else LivePosition(x_m=x, y_m=0, frame="m", rec_id="r", t_ms=0, score=0.6)
    return Localization(map_id="t", model="m", frames=1, status="uncertain", reason="", best=best, margin=lead,
                        candidates=cands, thresholds=TH, took_ms={}, position=pos)


def test_score_counts_what_the_belief_would_do():
    m = lmap()
    rows = [Row(Query("r1", i * 1000.0, np.zeros((1, 2)), None, walked_m=0.0, x_m=0.5, y_m=0.0, node="n1"),
                answer("n1", 0.2, x=1.0), 5.0) for i in range(6)]
    rows.append(Row(Query("r1", 6000.0, np.zeros((1, 2)), None, walked_m=0.0, x_m=0.5, y_m=0.0, node="n1"),
                    answer("n2", 0.01), 5.0))  # not decisive
    s = score("x", m, [rows])
    assert s.queries == 7 and s.node_right == 6 and s.decisive == 6 and s.decisive_right == 6
    assert s.pos_err == [0.5] * 6 and s.located_s == [1.0] and s.commits == 1 and s.commits_right == 1
    assert s.commit_err == [0.5]
    line = s.line()
    assert "node right  86%" in line and "belief located 1/1" in line and "within" not in line


def test_live_sessions_become_queries_with_the_phones_steps(tmp_path, monkeypatch):
    clock = iter(datetime(2026, 10, 11, 18, 0, s, tzinfo=timezone.utc) for s in range(10))
    cap = Capture(tmp_path / "live", clock=lambda: next(clock))
    m = lmap()
    for steps, n in ((10, 2), (20, 1), (None, 1)):
        motion = [{"stepCount": steps}] * n if steps is not None else []
        cap.localize("t", {"sessionId": "s1"}, [b"jpeg"] * n, {"motion": motion}, response={"best": "n1"})
    cap.route("t", {"sessionId": "s1"}, {"start": "n1", "goal": "n3"}, response={"found": True})
    assert cap.flush()
    seen = []

    def fake_place(paths, model):
        seen.append((len(paths), model))
        return np.eye(len(paths), 4, dtype=np.float32)

    monkeypatch.setattr("nav.mapping.inference.place_files", fake_place)
    m.embedding = type("E", (), {"model": "megaloc"})()
    [s] = sessions(tmp_path / "live", "t")
    qs = session_queries(m, s)
    assert seen == [(4, "megaloc")] and [q.q.shape[0] for q in qs] == [2, 1, 1]
    step = m.frame.live_step_length_m or m.frame.step_length_m
    assert [q.walked_m for q in qs] == [0.0, 10 * step, 10 * step] and [q.t_ms for q in qs] == [0, 1000, 2000]
    assert all(q.node is None and isinstance(q.image, list) for q in qs)
