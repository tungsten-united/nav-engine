"""More episodes than walks, and parameters picked on them (`nav map tune`).

A place has a handful of walks, so tuning the loop on their replays alone fits those few routes. Each walk is cut at
the map's evidence times into pieces, one per pair of nodes it went between (arriving at one, until arriving at the
next); a random walk over those pieces, each from any walk that went that way, makes a new route made of real
frames and steps. Every frame is matched against the other walks' references only (the user is never one of the
mapping walks), so a map needs two walks or more (single-walk maps fall back to leaving out frames near its own
references, which flatters).

The reward per episode, over the nodes the user got to: each node reached while there scores 1 minus a little per
second late; each move to a node the user wasn't at costs `wrong_cost` (a wrong instruction is worse than a missed
one); each restart costs `restart_cost`. Tuning maximises its mean over a seeded set of episodes (scipy's
differential evolution, deterministic for a seed) and reports it on another set and on the walks as they went.
"""

import random
from dataclasses import asdict, dataclass, fields

import numpy as np

from nav.navigation.replay import Episode, OrchestratorResult, Track, loop_factory, run_episode
from nav.navigation.route import hops
from nav.schemas.map import PlaceMap

WRONG_COST = 3.0
RESTART_COST = 0.3
LATE_COST_PER_S = 0.05  # per second after the user got there, up to 10 s
GOAL_TAIL_S = 6.0  # an episode goes on this long past its goal (or until the walk's next node), to arrive
MIN_PIECE_S = 1.0  # shorter pieces are two nodes seen at once (Itnig 7f42 at 38 s: Drinks area, Hackathon tables)


@dataclass
class Piece:
    rec_id: str
    source: str
    target: str
    t0: float  # the walk got to `source` (recording ms)
    t1: float  # ... and to `target`
    t_end: float  # until here the walk was still at `target` (its next visit, or the tail)


def pieces(tracks: list[Track]) -> list[Piece]:
    out = []
    for tr in tracks:
        v = tr.visits
        for k in range(len(v) - 1):
            (t0, a), (t1, b) = v[k], v[k + 1]
            t_end = min(v[k + 2][0] if k + 2 < len(v) else float("inf"), t1 + GOAL_TAIL_S * 1000)
            out.append(Piece(tr.rec_id, a, b, t0, t1, t_end))
    return out


def stitch(tracks: dict[str, Track], route: list[Piece]) -> Episode:
    """The pieces one after the other: each from its arrival at the source to its arrival at the target, the last
    one on to its tail; step counts and time continue across the joins."""
    t_out, emb, motions, recs, keys, truth = [], [], [], [], [], [(route[0].source, -float("inf"))]
    clock, steps = 0.0, 0
    for k, pc in enumerate(route):
        tr = tracks[pc.rec_id]
        end = pc.t_end if k == len(route) - 1 else pc.t1
        sel = np.flatnonzero((tr.t_ms >= pc.t0) & (tr.t_ms < end))
        if k:
            truth.append((pc.source, clock))
        base = tr.motions[sel[0]].step_count if len(sel) else 0
        for i in sel:
            m = tr.motions[i]
            t_out.append(clock + tr.t_ms[i] - pc.t0)
            emb.append(tr.emb[i])
            motions.append(m.model_copy(update={"step_count": steps + (m.step_count or 0) - (base or 0)}))
            recs.append(pc.rec_id)
            keys.append((pc.rec_id, float(tr.t_ms[i])))
        if len(sel):
            steps = motions[-1].step_count
        clock += pc.t1 - pc.t0
    truth.append((route[-1].target, clock))
    dim = next(iter(tracks.values())).emb.shape[1]
    return Episode(np.asarray(t_out, float), np.asarray(emb, np.float32).reshape(-1, dim), motions, recs, keys,
                   truth, route[-1].target)


def sample_episodes(pmap: PlaceMap, tracks: list[Track], n: int, seed: int, min_hops: int = 2, max_hops: int = 7,
                    exclude_rec: str | None = None) -> list[Episode]:
    """`n` random routes over the walks' pieces (seeded): only along the map's edges (so the route a user is given
    is the episode's own), never straight back unless it is the only way on, and ending at the first visit of the
    goal."""
    rng = random.Random(seed)
    walkable = {(h.source, h.target) for e in pmap.edges for h in hops(e)}
    by_source: dict[str, list[Piece]] = {}
    for pc in pieces([t for t in tracks if t.rec_id != exclude_rec]):
        if (pc.source, pc.target) in walkable and pc.t1 - pc.t0 >= MIN_PIECE_S * 1000:
            by_source.setdefault(pc.source, []).append(pc)
    starts = sorted(by_source)
    by_rec = {t.rec_id: t for t in tracks}
    out, tries = [], 0
    while len(out) < n and starts and tries < 50 * n:
        tries += 1
        node, route, length = rng.choice(starts), [], rng.randint(min_hops, max_hops)
        while len(route) < length and by_source.get(node):
            options = by_source[node]
            onward = [pc for pc in options if not route or pc.target != route[-1].source]
            pc = rng.choice(onward or options)
            route.append(pc)
            node = pc.target
        visited = [route[0].source] + [pc.target for pc in route] if route else []
        if len(route) < min_hops or visited.index(visited[-1]) != len(visited) - 1:
            continue  # too short, or the goal was passed on the way
        out.append(stitch(by_rec, route))
    return out


def reward(res: OrchestratorResult, wrong_cost: float = WRONG_COST, restart_cost: float = RESTART_COST) -> float:
    got = sum(1 - LATE_COST_PER_S * min(max(x["delay_s"], 0.0), 10.0) for x in res.nodes if x["delay_s"] is not None)
    return (got - wrong_cost * len(res.wrong) - restart_cost * res.relocations) / max(len(res.nodes), 1)


@dataclass
class Summary:
    reward: float
    reached: int
    nodes: int
    wrong: int
    restarts: int

    def __str__(self) -> str:
        return (f"reward {self.reward:+.3f} · reached {self.reached}/{self.nodes} · wrong {self.wrong} · "
                f"restarts {self.restarts}")


def evaluate(pmap: PlaceMap, index, episodes: list[Episode], thresholds, kind: str, params,
             cache: dict | None = None) -> Summary:
    make = loop_factory(pmap, kind, params)
    results = [run_episode(pmap, index, ep, thresholds, make, cache=cache) for ep in episodes]
    return Summary(float(np.mean([reward(r) for r in results])) if results else 0.0, sum(r.found for r in results),
                   sum(len(r.nodes) for r in results), sum(len(r.wrong) for r in results),
                   sum(r.relocations for r in results))


# What tuning searches, per loop: (low, high, integer).
SPACES = {
    "belief": {"temperature": (0.005, 0.1, False), "gate": (0.0, 1.0, False), "jump": (0.001, 0.2, False),
               "walk_scale": (0.2, 2.0, False), "walk_m_per_call": (0.1, 3.0, False), "reach": (0.4, 0.98, False),
               "leave": (0.4, 0.99, False)},
    "votes": {"k": (1, 6, True), "n": (1, 8, True), "margin": (0.0, 0.15, False), "lost_margin": (0.0, 0.2, False),
              "lost_s": (2.0, 20.0, False)},
}


def make_params(kind: str, x) -> object:
    from nav.navigation.belief import BeliefParams
    from nav.navigation.replay import LoopParams

    space = SPACES[kind]
    vals = {name: (round(v) if integer else float(v)) for (name, (_, _, integer)), v in zip(space.items(), x)}
    if kind == "votes":
        vals["n"] = max(vals["n"], vals["k"])
        return LoopParams(**vals)
    return BeliefParams(**vals)


def tune(pmap: PlaceMap, index, episodes: list[Episode], thresholds, kind: str, seed: int = 0, maxiter: int = 15,
         popsize: int = 8, log=None):
    """The params of loop `kind` with the best mean reward on `episodes`, starting from its defaults."""
    from scipy.optimize import differential_evolution

    from nav.navigation.belief import BeliefParams
    from nav.navigation.replay import LoopParams

    space = SPACES[kind]
    default = BeliefParams() if kind == "belief" else LoopParams()
    x0 = [min(max(getattr(default, name), lo), hi) for name, (lo, hi, _) in space.items()]
    cache: dict = {}
    best = {"r": -np.inf}

    def objective(x) -> float:
        r = evaluate(pmap, index, episodes, thresholds, kind, make_params(kind, x), cache).reward
        if r > best["r"]:
            best["r"] = r
            if log:
                log(f"  {r:+.3f}  {asdict(make_params(kind, x))}")
        return -r

    res = differential_evolution(objective, [(lo, hi) for lo, hi, _ in space.values()], x0=x0, seed=seed,
                                 maxiter=maxiter, popsize=popsize, polish=False, tol=1e-6, init="sobol")
    return make_params(kind, res.x)


def describe(params) -> str:
    return ", ".join(f"{f.name}={getattr(params, f.name):.4g}" if isinstance(getattr(params, f.name), float)
                     else f"{f.name}={getattr(params, f.name)}" for f in fields(params))
