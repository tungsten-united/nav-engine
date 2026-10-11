"""Where the user is, as a probability per node of the map (a Bayes filter, docs/navigation_stack.md 4.2), updated by
each /localize result and the steps walked since the last one. It replaces the orchestrator's votes (replay.py
OrchestratorLoop: k of the last n results with a clear leader): a weak lead that keeps up adds up, a node next to
the last one needs less evidence than one across the map, and nothing restarts on a reset.

- Motion: walking w metres, the user leaves node a for its neighbour b (over the map's links, either way) at a rate
  w / length(a, b); without steps, `walk_m_per_call` stands in for w. A little of the belief (`jump`) spreads over
  every node each call: a move the camera and steps missed, or a wrong start.
- Seeing: a result with node scores s multiplies node n by exp((s_n - s_best) / temperature) (a node not among the
  candidates scores as the lowest one, at most the leader's score minus the margin); a node nav-api marks not plausible (farther than walked from `previous`) also
  by `gate`. Live scores are low and the leader's lead is what tells (CLAUDE.md, Itnig 2026-10-10): this uses only
  differences.

BeliefLoop is OrchestratorLoop with that belief for votes: located, a node ahead on the route reached, or the route
dropped when one node is at least `reach` / `leave` likely. Its parameters come from `nav map tune` (episodes.py).
"""

from dataclasses import dataclass

import numpy as np

from nav.navigation.pdr import map_step_length
from nav.navigation.replay import LoopEvent, OrchestratorLoop
from nav.navigation.route import UNKNOWN_LENGTH_M
from nav.schemas.map import PlaceMap
from nav.schemas.navigation import Localization


@dataclass
class BeliefParams:
    """From `nav map tune itnig` (2026-10-11, build 20261010T131002Z, its four walks each matched against the other
    three's references, 150 stitched episodes), rounded. On 300 other episodes (966 nodes): 293 reached, 0 wrong
    moves, 4 restarts; the votes (LoopParams) 289 reached, 430 wrong, 286 restarts. On the four walks as they went
    (18 nodes): 5 reached, 1 wrong; the votes 5 and 5. `walk_scale` is a fit, not a step length: the motion model is
    memoryless and has to move belief on by the time the camera shows the next node. Replays always send steps, so
    `walk_m_per_call` is not fitted: about the steps of a call at that scale."""

    temperature: float = 0.25  # score difference that makes a node e times less likely than the leader
    gate: float = 0.8  # likelihood of a node nav-api marks farther than walked
    jump: float = 0.1  # belief spread over every node each call
    walk_scale: float = 3.7  # metres walked = phone steps x the map's live step length x this
    walk_m_per_call: float = 1.5  # metres walked per call when the phone sends no steps
    reach: float = 0.7  # locate / reach a node ahead this likely
    leave: float = 0.7  # drop the route for a node off it this likely

class NodeBelief:
    def __init__(self, pmap: PlaceMap, params: BeliefParams | None = None):
        self.p_ = params or BeliefParams()
        self.nodes = [n.id for n in pmap.nodes]
        self.ix = {n: i for i, n in enumerate(self.nodes)}
        n = len(self.nodes)
        self.inv_len = np.zeros((n, n))  # 1 / metres between linked nodes
        for e in pmap.links or pmap.edges:
            if e.source in self.ix and e.target in self.ix and e.source != e.target:
                a, b = self.ix[e.source], self.ix[e.target]
                length = max(e.distance_m if e.distance_m is not None else UNKNOWN_LENGTH_M, 0.5)
                self.inv_len[a, b] = self.inv_len[b, a] = max(self.inv_len[a, b], 1 / length)
        self.p = np.full(n, 1 / n) if n else np.zeros(0)

    def predict(self, walked_m: float) -> None:
        rate = max(walked_m, 0.0) * self.inv_len  # [from, to]
        total = rate.sum(axis=1)
        leave = 1 - np.exp(-total)
        trans = np.divide(rate * leave[:, None], total[:, None], out=np.zeros_like(rate), where=total[:, None] > 0)
        trans[np.diag_indices_from(trans)] = 1 - leave
        p = self.p @ trans
        self.p = (1 - self.p_.jump) * p + self.p_.jump / len(p)

    def update(self, loc: Localization) -> None:
        scores = {c.node: c.score for c in loc.candidates if c.node in self.ix}
        if not scores:
            return
        best = max(scores.values())
        floor = min([*scores.values(), best - (loc.margin or 0.0)])  # a node not listed: at most the second's score
        s = np.full(len(self.nodes), floor)
        gate = np.ones(len(self.nodes))
        for c in loc.candidates:
            if c.node in self.ix:
                s[self.ix[c.node]] = c.score
                if c.plausible is False:
                    gate[self.ix[c.node]] = self.p_.gate
        p = self.p * np.exp((s - best) / self.p_.temperature) * gate
        self.p = p / p.sum() if p.sum() > 0 else np.full(len(p), 1 / len(p))

    def prob(self, node: str) -> float:
        return float(self.p[self.ix[node]]) if node in self.ix else 0.0

    def best(self) -> tuple[str | None, float]:
        if not self.nodes:
            return None, 0.0
        i = int(np.argmax(self.p))
        return self.nodes[i], float(self.p[i])


class BeliefLoop(OrchestratorLoop):
    """One session's navigation on the belief: `request`, `on_localization`, `path`, `hop`, `arrived` as
    OrchestratorLoop. Asks nav-api for every node's score (top_k)."""

    def __init__(self, goal: str, path_fn, pmap: PlaceMap, params: BeliefParams | None = None):
        super().__init__(goal, path_fn)
        self.bp = params or BeliefParams()
        self.belief = NodeBelief(pmap, self.bp)
        self.step_length = map_step_length(pmap)
        self.top_k = max(1, min(20, len(pmap.nodes)))  # nav-api's maximum
        self.last_steps: int | None = None

    def request(self) -> dict:
        return {**super().request(), "top_k": self.top_k}

    def on_localization(self, loc: Localization, step_count: int | None, t_s: float) -> LoopEvent | None:
        if self.arrived:
            return None
        if step_count is None or (self.last_steps is not None and step_count < self.last_steps):
            walked = self.bp.walk_m_per_call  # no steps, or the phone restarted counting
        else:
            walked = (step_count - (self.last_steps or step_count)) * self.step_length * self.bp.walk_scale
        self.last_steps = step_count
        self.belief.predict(walked)
        self.belief.update(loc)
        node, prob = self.belief.best()
        if self.path is None:
            return self._located(node, step_count, t_s) if node is not None and prob >= self.bp.reach else None
        source, ahead = self.path[self.hop].source, self.ahead()
        target = max(ahead, key=self.belief.prob)
        if self.belief.prob(target) >= self.bp.reach:
            return self._reach(target, step_count, t_s)
        if node != source and node not in ahead and prob >= self.bp.leave:
            self.path = None
            return LoopEvent("lost", self.node, t_s)
        return None

