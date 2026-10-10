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

For the current hop A -> T (and the next hop T -> U), each localization votes:
- arrive: T is recognised (score >= confirm_score, margin >= confirm_margin) through an approach / at reference;
- pass: T is recognised through an outgoing reference toward U, or U is recognised through an approach / at
  reference (in an open space the next spot is often what the camera sees from T): the user is at or past T;
- elsewhere: a node other than A, T, U is recognised.
A vote counts only if the dead reckoning agreed when the frame was taken, so T seen from afar doesn't add up to an
arrival once the user gets near:
- strict gate (approach references: they are taken from a distance): near T, P(arrival) >= gate_p or
  |remaining| <= gate_sigmas sigma;
- loose gate (at references, the spot itself, and pass votes): at least (1 - loose_frac) of the hop walked. Map
  lengths can be off by a few metres (Itnig: the camera recognised a node from its "at" photos 3-4 m before the
  dead reckoning got there), and requiring the strict gate there loses most confirmations.
Events, preferring a late confirmation to a false one (recognising a place from afar is not being there):
- arrived T: counted arrive votes >= k of the last n and the heading agrees with the hop's arrival direction when
  both are known;
- passed T: counted pass votes >= k of n and the heading agrees with U's direction. Never at the goal (there you
  arrive);
- off_route: elsewhere votes >= k of n. Confirmation stops until `reroute`;
- uncertain: the user walked gate_sigmas sigma past T without a confirmation (once per hop): ask for a slow pan or a
  manual confirmation;
- manual: always accepted for T.
Only T can be confirmed, so no checkpoint is skipped. A confirmation calls `tracker.confirm_node(T)` (re-learning the
step length) and moves to the next hop. Without dead reckoning for a hop (no tracker, no length) the gate counts as
open but arrival needs n of n votes. Thresholds are placeholders until tuned (`nav map nav-replay`).
"""

from collections import deque
from dataclasses import dataclass, field

from nav.navigation.localize import angle_diff

CONFIRM_KINDS = ("arrived", "passed", "manual")


@dataclass
class ArrivalParams:
    k: int = 2  # votes needed ...
    n: int = 3  # ... among the last n localizations (one burst a second: about 2-3 s to confirm)
    gate_p: float = 0.2  # P(within the arrival radius) that opens the dead-reckoning gate ...
    gate_sigmas: float = 2.0  # ... or |remaining| within this many sigma
    loose_frac: float = 0.5  # loose gate: at least (1 - loose_frac) of the hop walked
    heading_tol_deg: float = 45.0


@dataclass
class ArrivalEvent:
    kind: str  # arrived | passed | manual | uncertain | off_route
    node: str | None  # the confirmed node; for off_route the node recognised instead
    hop_index: int  # the hop it happened on
    t: float
    evidence: dict = field(default_factory=dict)


def _wrap_abs(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else abs(angle_diff(a, b))


class NodeConfirmer:
    def __init__(self, route: list, tracker=None, params: ArrivalParams | None = None):
        """`route`: hops (route.Hop or nav-api RouteHop: source, target, length_m, bearing_deg, arrival_deg).
        `tracker`: a pdr.GraphTracker over the same hops, or None."""
        self.route = list(route)
        self.tracker = tracker
        self.p = params or ArrivalParams()
        self.hop_index = 0
        self.votes: deque = deque(maxlen=self.p.n)
        self.heading: float | None = None  # last known phone heading, map frame
        self.off_route = False
        self.done = not self.route
        self._uncertain_sent = False
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
        if self.done or self.off_route or self.tracker is None:
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
        th = loc.thresholds
        clear = top.score >= th.confirm_score and (loc.margin or 0.0) >= th.confirm_margin
        ref = top.refs[0] if top.refs else None
        info = {"best": loc.best, "score": top.score, "margin": loc.margin, "role": ref.role if ref else None,
                "toward": ref.toward if ref else None}
        if not clear:
            return None, info
        if loc.best == hop.target:
            if ref and ref.role == "outgoing":
                return ("pass" if nxt is not None and ref.toward == nxt.target else None), info
            return "arrive", info
        if nxt is not None and loc.best == nxt.target:
            return ("pass" if ref and ref.role in ("approach", "at") else None), info
        if loc.best != hop.source:
            return "elsewhere", info
        return None, info

    def on_localization(self, t: float, loc) -> ArrivalEvent | None:
        if loc.heading_map_deg is not None:
            self.heading = loc.heading_map_deg % 360
        if self.done or self.off_route:
            return None
        vote, info = self._vote(loc)
        pdr = self._pdr()
        # A vote counts only if the dead reckoning agreed when the frame was taken: T seen from afar, before the
        # gate opened, must not add up to an arrival once the user gets near.
        role = info.get("role")
        if vote == "arrive" and not (self._near(pdr) or (role == "at" and self._walked_enough(pdr))):
            vote = "far"
        elif vote == "pass" and not self._walked_enough(pdr):
            vote = "far"
        self.votes.append(vote)
        n_arrive, n_pass, n_else = (sum(v == x for v in self.votes) for x in ("arrive", "pass", "elsewhere"))
        hop, nxt = self.hop, self.next_hop
        evidence = {"pdr": pdr, "last": info, "votes": list(self.votes), "heading_map_deg": self.heading}

        if n_else >= self.p.k:
            self.off_route = True
            return self._emit("off_route", info.get("best"), t, evidence)

        need = self.p.k if pdr is not None else self.p.n  # without dead reckoning, ask the camera for more
        arrive_diff = _wrap_abs(self.heading, getattr(hop, "arrival_deg", None))
        evidence["heading_diff_deg"] = arrive_diff
        if n_arrive >= need and self._near(pdr) and (arrive_diff is None or arrive_diff <= self.p.heading_tol_deg):
            return self._confirm("arrived", t, evidence)
        if nxt is not None and n_pass >= need and self._walked_enough(pdr):
            leave_diff = _wrap_abs(self.heading, nxt.bearing_deg)
            evidence["heading_diff_deg"] = leave_diff
            if leave_diff is None or leave_diff <= self.p.heading_tol_deg:
                return self._confirm("passed", t, evidence)
        return None

    def confirm_manual(self, t: float = 0.0) -> ArrivalEvent | None:
        if self.done:
            return None
        self.off_route = False
        return self._confirm("manual", t, {"pdr": self._pdr()})

    def reroute(self, route: list, tracker=None) -> None:
        """A new route (after off_route and relocalization): start over on it."""
        self.__init__(route, tracker, self.p)

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
        return ev
