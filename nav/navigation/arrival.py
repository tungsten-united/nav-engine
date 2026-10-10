"""Node confirmation along a route (docs/navigation_stack.md 4.2-4.3, 6, 8): has the user reached, or walked through,
the next checkpoint? Causal and small, so it ports to the app (frontend/nav/arrival.js mirrors it 1:1).

Inputs, all optional but the first:
- `on_localization(t, loc)`: each nav-api `Localization` (localize.py) of the live frames, made with
  previous = `confirmer.previous` and expected = `confirmer.expected`. The app sends a burst of 2 frames every
  second (localize averages them, which smooths blur and passers-by); each burst is one vote. How often a vote
  comes matters more than frames per vote: at a walk's ~0.5 m/s the user is near a node for a few seconds
  (Itnig, nav-replay: 2 every 1 s confirmed 7/9 nodes, 3 every 2 s only 3/9, both with 0 false);
- `on_step(heading_map_deg)`: each detected step (pdr.StepDetector), with the phone heading in the map frame;
- `confirm_manual(t)`: the user says or taps "I'm here".

For the current hop A -> T (and the next hop T -> U), each localization votes. Votes use the confirmer's own
thresholds, not nav-api's `confirm_score` (live phone frames of the right node score 0.36-0.48 at Itnig; the
leader and its margin matter, not the absolute score): the best node X votes when its score >= vote_score and its
margin over the second >= vote_margin.
- arrive: X = T;
- pass: X = T through an outgoing reference toward U, or X = U through an approach / at reference (in an open space
  the next spot is often what the camera sees from T): the user is at or past T;
- elsewhere: X is none of A, T, U nor next to any of them (`neighbours`, the map's edges): a node next to the hop is
  often in view from it, and is no sign of being lost.
A vote counts only if the user had walked far enough when the frame was taken, so T seen from afar doesn't add up to
an arrival once the user gets near:
- with the phone's motion, nav-api says so: the best node's `plausible` (walked since A from the phone's step count
  >= its distance - slack, localize.py; no compass involved);
- without it, the dead reckoning (GraphTracker, which projects steps on the compass): approach references need
  near T (P(arrival) >= gate_p or |remaining| <= gate_sigmas sigma); at references and pass votes need at least
  (1 - loose_frac) of the hop walked.
Events, preferring a late confirmation to a false one (recognising a place from afar is not being there):
- arrived T: counted arrive votes >= k of the last n. The heading is not checked: the camera is pointed around and
  phone compasses are off by tens of degrees indoors;
- passed T: counted pass votes >= k of n and the heading agrees with U's direction (heading_tol_deg). Never at the
  goal (there you arrive);
- off_route: elsewhere votes >= k of n, at most once per hop. A warning: confirmation goes on;
- uncertain: the user walked gate_sigmas sigma past T without a confirmation (once per hop): ask for a slow pan or a
  manual confirmation;
- manual: always accepted for T.
Only T can be confirmed, so no checkpoint is skipped. A confirmation calls `tracker.confirm_node(T)` (re-learning the
step length) and moves to the next hop. Without any gate for a hop (no motion, no tracker, no length) arrival needs n
of n votes. Defaults from the /nav test at Itnig on 2026-10-10 (nav-api logs); the /nav page takes ?k=&n=&score=&margin=
to tune them on site.
"""

from collections import deque
from dataclasses import dataclass, field

from nav.navigation.localize import angle_diff

CONFIRM_KINDS = ("arrived", "passed", "manual")


@dataclass
class ArrivalParams:
    k: int = 2  # votes needed ...
    n: int = 3  # ... among the last n localizations (one burst a second: about 2-3 s to confirm)
    vote_score: float = 0.30  # the best node votes when it scores at least this ...
    vote_margin: float = 0.04  # ... and leads the second by at least this
    gate_p: float = 0.2  # P(within the arrival radius) that opens the dead-reckoning gate ...
    gate_sigmas: float = 2.0  # ... or |remaining| within this many sigma
    loose_frac: float = 0.5  # loose gate: at least (1 - loose_frac) of the hop walked
    heading_tol_deg: float = 90.0  # passing only


@dataclass
class ArrivalEvent:
    kind: str  # arrived | passed | manual | uncertain | off_route
    node: str | None  # the confirmed node; for off_route the node recognised instead
    hop_index: int  # the hop it happened on
    t: float
    evidence: dict = field(default_factory=dict)


def neighbours(pmap) -> dict[str, set[str]]:
    """node -> the nodes an edge joins it to, either direction (NodeConfirmer's `neighbours`)."""
    out: dict[str, set[str]] = {}
    for e in pmap.edges:
        out.setdefault(e.source, set()).add(e.target)
        out.setdefault(e.target, set()).add(e.source)
    return out


def _wrap_abs(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else abs(angle_diff(a, b))


class NodeConfirmer:
    def __init__(self, route: list, tracker=None, params: ArrivalParams | None = None,
                 neighbours: dict[str, set[str]] | None = None):
        """`route`: hops (route.Hop or nav-api RouteHop: source, target, length_m, bearing_deg, arrival_deg).
        `tracker`: a pdr.GraphTracker over the same hops, or None. `neighbours`: node -> the nodes an edge joins it
        to (`neighbours(pmap)`), or None (every node off the hop is elsewhere)."""
        self.route = list(route)
        self.tracker = tracker
        self.p = params or ArrivalParams()
        self.neighbours = neighbours or {}
        self.hop_index = 0
        self.votes: deque = deque(maxlen=self.p.n)
        self.heading: float | None = None  # last known phone heading, map frame
        self.walk_gate = False  # nav-api has said whether nodes are plausible (the phone sends its motion)
        self.done = not self.route
        self._uncertain_sent = False
        self._off_route_sent = False
        self.events: list[ArrivalEvent] = []

    # --- where we are -------------------------------------------------------------------------------------------
    @property
    def hop(self):
        return None if self.done else self.route[self.hop_index]

    @property
    def next_hop(self):
        return self.route[self.hop_index + 1] if not self.done and self.hop_index + 1 < len(self.route) else None

    @property
    def previous(self) -> str | None:
        """The last confirmed node (send as `previous` to /localize)."""
        if not self.route:
            return None
        return self.route[-1].target if self.done else self.route[self.hop_index].source

    @property
    def expected(self) -> str | None:
        """The next checkpoint (send as `expected` to /localize)."""
        return None if self.done else self.route[self.hop_index].target

    def _pdr(self) -> dict | None:
        """The tracker's state on the current hop, or None when it can't tell (no tracker, no hop length)."""
        if self.tracker is None or self.hop is None or not self.hop.length_m:
            return None
        s = self.tracker.state()
        return {"p": round(s.arrival_probability, 3), "progress_m": round(s.progress_m, 2),
                "remaining_m": round(s.remaining_m, 2), "sigma_m": round(s.sigma_m, 2)}

    def _near(self, pdr: dict | None) -> bool:
        return pdr is None or pdr["p"] >= self.p.gate_p or abs(pdr["remaining_m"]) <= self.p.gate_sigmas * pdr["sigma_m"]

    def _walked_enough(self, pdr: dict | None) -> bool:
        return pdr is None or pdr["progress_m"] >= (1 - self.p.loose_frac) * self.hop.length_m

    # --- inputs -------------------------------------------------------------------------------------------------
    def on_step(self, heading_map_deg: float | None, t: float = 0.0) -> ArrivalEvent | None:
        if heading_map_deg is not None:
            self.heading = heading_map_deg % 360
        if self.done or self.tracker is None:
            return None
        self.tracker.on_step(self.heading if self.heading is not None else (self.hop.bearing_deg or 0.0))
        pdr = self._pdr()
        if pdr and not self._uncertain_sent and pdr["remaining_m"] < -self.p.gate_sigmas * pdr["sigma_m"]:
            self._uncertain_sent = True
            return self._emit("uncertain", self.hop.target, t, {"pdr": pdr, "why": "walked past it, not recognised"})
        return None

    def _vote(self, loc) -> tuple[str | None, dict]:
        hop, nxt = self.hop, self.next_hop
        if not loc.candidates or loc.best is None:
            return None, {}
        top = loc.candidates[0]
        clear = top.score >= self.p.vote_score and (loc.margin or 0.0) >= self.p.vote_margin
        ref = top.refs[0] if top.refs else None
        info = {"best": loc.best, "score": top.score, "margin": loc.margin, "role": ref.role if ref else None,
                "toward": ref.toward if ref else None, "plausible": top.plausible}
        if not clear:
            return None, info
        if loc.best == hop.target:
            if ref and ref.role == "outgoing":
                return ("pass" if nxt is not None and ref.toward == nxt.target else None), info
            return "arrive", info
        if nxt is not None and loc.best == nxt.target:
            return ("pass" if ref and ref.role in ("approach", "at") else None), info
        on_hop = [hop.source, hop.target] + ([nxt.target] if nxt is not None else [])
        if loc.best in on_hop or any(loc.best in self.neighbours.get(x, ()) for x in on_hop):
            return None, info
        return "elsewhere", info

    def _gate(self, vote: str, info: dict, pdr: dict | None) -> bool:
        """Had the user walked far enough for this vote when the frame was taken?"""
        if info.get("plausible") is not None:  # nav-api's walked-distance gate (the phone's steps, no compass)
            return info["plausible"]
        if vote == "arrive":
            return self._near(pdr) or (info.get("role") == "at" and self._walked_enough(pdr))
        return self._walked_enough(pdr)

    def on_localization(self, t: float, loc) -> ArrivalEvent | None:
        if loc.heading_map_deg is not None:
            self.heading = loc.heading_map_deg % 360
        if self.done:
            return None
        vote, info = self._vote(loc)
        if info.get("plausible") is not None:
            self.walk_gate = True
        pdr = self._pdr()
        # A vote counts only if the user had walked far enough when the frame was taken: T seen from afar, before
        # the gate opened, must not add up to an arrival once the user gets near.
        if vote in ("arrive", "pass") and not self._gate(vote, info, pdr):
            vote = "far"
        self.votes.append(vote)
        n_arrive, n_pass, n_else = (sum(v == x for v in self.votes) for x in ("arrive", "pass", "elsewhere"))
        nxt = self.next_hop
        evidence = {"pdr": pdr, "last": info, "votes": list(self.votes), "heading_map_deg": self.heading}

        gated = self.walk_gate or pdr is not None
        need = self.p.k if gated else self.p.n  # without any gate, ask the camera for more
        if n_arrive >= need:
            return self._confirm("arrived", t, evidence)
        if nxt is not None and n_pass >= need:
            leave_diff = _wrap_abs(self.heading, nxt.bearing_deg)
            evidence["heading_diff_deg"] = leave_diff
            if leave_diff is None or leave_diff <= self.p.heading_tol_deg:
                return self._confirm("passed", t, evidence)
        if n_else >= self.p.k and not self._off_route_sent:
            self._off_route_sent = True  # a warning: confirmation goes on
            return self._emit("off_route", info.get("best"), t, evidence)
        return None

    def confirm_manual(self, t: float = 0.0) -> ArrivalEvent | None:
        if self.done:
            return None
        return self._confirm("manual", t, {"pdr": self._pdr()})

    def reroute(self, route: list, tracker=None) -> None:
        """A new route (after off_route and relocalization): start over on it."""
        self.__init__(route, tracker, self.p, self.neighbours)

    # --- outputs ------------------------------------------------------------------------------------------------
    def _emit(self, kind: str, node: str | None, t: float, evidence: dict) -> ArrivalEvent:
        ev = ArrivalEvent(kind, node, self.hop_index, t, evidence)
        self.events.append(ev)
        return ev

    def _confirm(self, kind: str, t: float, evidence: dict) -> ArrivalEvent:
        target = self.hop.target
        ev = self._emit(kind, target, t, evidence)
        if self.tracker is not None:
            try:
                self.tracker.confirm_node(target)
            except ValueError:  # a tracker over a different route: keep confirming, it just won't re-learn
                pass
        if self.hop_index + 1 >= len(self.route):
            self.done = True
        else:
            self.hop_index += 1
        self.votes.clear()
        self._uncertain_sent = False
        self._off_route_sent = False
        return ev
