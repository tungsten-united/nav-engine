"""frontend/nav/arrival.js is a 1:1 port of nav/navigation/pdr.py + arrival.py: the same inputs give the same steps and
the same events. Needs node (skipped without it)."""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from synthetic import walking_accel
from test_arrival import TH, loc, route, two

from nav.navigation.arrival import ArrivalParams, NodeConfirmer
from nav.navigation.pdr import GraphTracker, detect_steps

JS = Path(__file__).resolve().parent.parent / "frontend" / "nav" / "arrival.js"
NODE = shutil.which("node")

RUNNER = r"""
const A = require(process.argv[1]);
const s = JSON.parse(require("fs").readFileSync(0, "utf8"));
const det = new A.StepDetector();
const steps = [];
s.accel.forEach(([t, ax, ay, az]) => { if (det.update(t, [ax, ay, az])) steps.push(t); });
const tracker = new A.GraphTracker(s.route, { stepLength: 0.7 });
const nb = Object.fromEntries(Object.entries(s.neighbours).map(([k, v]) => [k, new Set(v)]));
const c = new A.NodeConfirmer(s.route, tracker, s.params, nb);
const out = [];
for (const op of s.ops) {
  let ev = null;
  if (op[0] === "step") ev = c.onStep(op[1], op[2]);
  else if (op[0] === "loc") ev = c.onLocalization(op[1], op[2]);
  else if (op[0] === "manual") ev = c.confirmManual(op[1]);
  if (ev) out.push([ev.kind, ev.node, ev.hop_index, ev.t]);
}
console.log(JSON.stringify({ steps, events: out, stepLength: tracker.stepLength, expected: c.expected, done: c.done }));
"""


def scenario() -> tuple[list, list]:
    """A walk along route(): a node off the route (a warning) and one next to it (no vote), far sightings, steps,
    an arrival on scores under nav-api's confirm score and the camera pointed back, a pass-through with nav-api's
    walked-distance gate closed once, a gap, a manual confirm."""
    ops: list = []
    t = 0.0

    def steps(n, heading):
        nonlocal t
        for _ in range(n):
            t += 0.8
            ops.append(["step", heading, t])

    def locs(*items):
        nonlocal t
        for it in items:
            t += 0.5
            ops.append(["loc", t, it])

    locs(loc("z"), loc("x"), loc("z"))  # z: off the route (warned once); x: next to b, no vote
    locs(loc("b", "approach", heading=90), loc("b", heading=90))  # b from afar: far votes
    steps(9, 90.0)
    locs(loc("b", score=0.35, margin=0.08, heading=90), loc(None), loc("b", score=0.35, margin=0.08, heading=270),
         loc("b", heading=90))  # arrives at b
    steps(8, 180.0)
    locs(loc("c", "outgoing", toward="d", heading=180), loc("d", heading=180, plausible=False),
         loc("d", "approach", heading=180), loc("d", heading=180))  # passes c: its outgoing views, then d's
    steps(30, 180.0)  # walks far past d without seeing it: uncertain once
    ops.append(["manual", t + 1])
    return ops, [loc_dict(op) for op in ops]


def loc_dict(op):
    return [op[0], op[1], op[2].model_dump(mode="json")] if op[0] == "loc" else op


NEIGHBOURS = {"b": {"x"}, "x": {"b"}}


def run_python(ops, accel):
    t, acc = accel
    steps = detect_steps(t, acc)
    hops = route()
    tracker = GraphTracker(hops, step_length=0.7)
    c = NodeConfirmer(hops, tracker, ArrivalParams(), NEIGHBOURS)
    events = []
    for op in ops:
        ev = None
        if op[0] == "step":
            ev = c.on_step(op[1], op[2])
        elif op[0] == "loc":
            ev = c.on_localization(op[1], op[2])
        else:
            ev = c.confirm_manual(op[1])
        if ev:
            events.append([ev.kind, ev.node, ev.hop_index, ev.t])
    return {"steps": [float(x) for x in steps], "events": events, "stepLength": tracker.step_length,
            "expected": c.expected, "done": c.done}


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_js_port_matches_python():
    ops, json_ops = scenario()
    t, acc = walking_accel(seconds=12)
    py = run_python(ops, (t, acc))
    p = ArrivalParams()
    payload = {"route": [h.__dict__ | {"steps": []} for h in route()], "ops": json_ops,
               "accel": np.c_[t, acc].tolist(),
               "neighbours": {k: sorted(v) for k, v in NEIGHBOURS.items()},
               "params": {"k": p.k, "n": p.n, "score": p.vote_score, "margin": p.vote_margin,
                          "runnerUp": p.runner_up_margin, "runnerUpExtra": p.runner_up_extra_m, "gateP": p.gate_p,
                          "gateSigmas": p.gate_sigmas, "looseFrac": p.loose_frac, "headingTol": p.heading_tol_deg}}
    out = subprocess.run([NODE, "-e", RUNNER, str(JS)], input=json.dumps(payload), capture_output=True, text=True,
                         check=True)
    js = json.loads(out.stdout)
    assert [e[:3] for e in js["events"]] == [e[:3] for e in py["events"]]
    assert [e[0] for e in py["events"]] == ["off_route", "arrived", "passed", "uncertain", "manual"]
    assert js["steps"] == pytest.approx(py["steps"]) and len(py["steps"]) > 10
    assert js["stepLength"] == pytest.approx(py["stepLength"])
    assert (js["expected"], js["done"]) == (py["expected"], py["done"])
    assert TH.confirm_score == 0.6  # the scenario's localizations use these thresholds


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_js_port_matches_python_on_the_runner_up_rule():
    """The source keeps a close lead over the target while the user walks the hop: both ports arrive at the same call."""
    ops = [["loc", float(t), two("a", "b", 0.62, 0.61, walked=w, dist2=7.0)] for t, w in enumerate([5, 7.5, 8.1, 8.5])]
    ops += [["loc", 4.0 + t, two("b", "c", 0.6, 0.59, walked=w, dist2=7.0)] for t, w in enumerate([3, 9])]
    json_ops = [loc_dict(op) for op in ops]
    t, acc = walking_accel(seconds=2)
    py = run_python(ops, (t, acc))
    p = ArrivalParams()
    payload = {"route": [h.__dict__ | {"steps": []} for h in route()], "ops": json_ops, "accel": np.c_[t, acc].tolist(),
               "neighbours": {k: sorted(v) for k, v in NEIGHBOURS.items()},
               "params": {"k": p.k, "n": p.n, "score": p.vote_score, "margin": p.vote_margin,
                          "runnerUp": p.runner_up_margin, "runnerUpExtra": p.runner_up_extra_m, "gateP": p.gate_p,
                          "gateSigmas": p.gate_sigmas,
                          "looseFrac": p.loose_frac, "headingTol": p.heading_tol_deg}}
    out = subprocess.run([NODE, "-e", RUNNER, str(JS)], input=json.dumps(payload), capture_output=True, text=True,
                         check=True)
    js = json.loads(out.stdout)
    assert [e[:3] for e in js["events"]] == [e[:3] for e in py["events"]] == [["arrived", "b", 0]]
