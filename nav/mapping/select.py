"""Candidate selection: from ~1 frame per second of every walk to a few dozen frames worth showing the LLM.

1. Quality: drop frames that are blurry for this recording, too dark or bright, or look at the floor / ceiling.
2. Per recording, walking in time order, keep a frame when
   - it is the first or last usable frame (start, end),
   - a tag was pressed or a voice note started around it (mark, voice: the person said something is here),
   - it looks new compared to the last kept frame and the person walked or turned enough since (novel, turn),
   - or the person walked far since the last kept frame (distance), so long corridors keep their length.
3. Across all recordings: frames that show the same place are folded into one candidate with several views
   (complete linkage, so chains of slightly-similar frames don't collapse). "Same place" (verify.py) means MegaLoc
   retrieved the pair and ALIKED+LightGlue found enough geometrically consistent matches. Within one walk only
   revisits (>= revisit_min_s apart) are folded; a near-duplicate (DINOv2, dup_sim) of a recent frame with
   nothing said or tagged is simply dropped. Pairs that look alike (MegaLoc cosine >= hint_sim) but weren't
   folded are reported as possibly the same place.
4. If there are still more than max_candidates, the most redundant unanchored ones are dropped.
"""

from dataclasses import asdict, dataclass, field

import numpy as np

from nav.mapping.motion import MotionTrack, angle_diff


@dataclass
class SelectParams:
    fps: float = 1.0  # frames sampled per second of video
    image_long_side: int = 768  # px, frames stored and shown to the LLM
    min_sharpness_rel: float = 0.35  # x the recording's median sharpness
    min_brightness: float = 25
    max_brightness: float = 235
    max_abs_pitch_deg: float = 55  # camera elevation; beyond this it looks at the floor or ceiling
    novelty_sim: float = 0.80  # a frame less similar than this to the last kept one looks new ...
    min_gap_m: float = 2.0  # ... and is kept once the person walked this far ...
    turn_deg: float = 60  # ... or turned this much since the last kept frame
    max_gap_m: float = 10.0  # keep a frame at least every this many metres walked
    anchor_before_s: float = 2.0  # marks and voice notes look for a frame from this long before ...
    anchor_after_s: float = 4.0  # ... to this long after
    dup_sim: float = 0.92  # DINOv2 cosine: a recent frame this similar is a near-duplicate
    hint_sim: float = 0.45  # MegaLoc cosine from which two frames may be the same place (checked, else a hint)
    retrieve_k: int = 10  # ... and only each frame's most similar ones are checked
    # LightGlue matches fitting one fundamental matrix to fold two frames as one spot. One open hall (Itnig):
    # near-identical views 490-870, different spots of the same room 100-300, so this asks for the same spot.
    min_inliers: int = 300
    revisit_min_s: float = 30  # within one walk, only frames this far apart in time count as a revisit
    max_candidates: int = 60
    calibrate: bool = True  # align frame times with the sensors and measure the field of view (calibrate.py)
    odometry: bool = True  # walked metres from the floor seen in the video where possible (odometry.py)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Frame:
    rec_id: str
    t_ms: float
    sharpness: float
    brightness: float
    image_path: str  # absolute, in the cache
    emb: np.ndarray  # DINOv2: novelty within a walk
    place: np.ndarray  # MegaLoc: same place across walks
    motion: dict  # MotionTrack.at(t_ms)
    usable: bool = True
    reasons: list[str] = field(default_factory=list)
    marks: list[str] = field(default_factory=list)
    voice: list[str] = field(default_factory=list)

    @property
    def anchored(self) -> bool:
        return bool(self.marks or self.voice) or any(r in ("start", "end") for r in self.reasons)


def mark_usable(frames: list[Frame], p: SelectParams) -> None:
    if not frames:
        return
    floor = p.min_sharpness_rel * float(np.median([f.sharpness for f in frames]))
    for f in frames:
        pitch = f.motion.get("pitch_deg")
        f.usable = (f.sharpness >= floor and p.min_brightness <= f.brightness <= p.max_brightness
                    and (pitch is None or abs(pitch) <= p.max_abs_pitch_deg))


def _best_in(frames: list[Frame], t0: float, t1: float, near: float) -> Frame | None:
    """Sharpest usable frame in [t0, t1], favouring frames close to `near`."""
    pool = [f for f in frames if f.usable and t0 <= f.t_ms <= t1]
    if not pool:
        return None
    return max(pool, key=lambda f: f.sharpness / (1 + abs(f.t_ms - near) / 2000))


def select_in_recording(frames: list[Frame], track: MotionTrack, marks: list[dict], voice: list[dict],
                        p: SelectParams) -> list[Frame]:
    """Keyframes of one recording, in time order. `marks`: [{t_ms, tag}], `voice`: [{t_ms, t_end_ms, text}]."""
    usable = [f for f in frames if f.usable]
    if not usable:
        return []
    usable[0].reasons.append("start")
    usable[-1].reasons.append("end")
    for m in marks:
        f = _best_in(frames, m["t_ms"] - p.anchor_before_s * 1000, m["t_ms"] + p.anchor_after_s * 1000, m["t_ms"])
        if f:
            f.reasons.append("mark")
            f.marks.append(m["tag"])
    for v in voice:
        end = min(v["t_end_ms"], v["t_ms"] + p.anchor_after_s * 1000)
        f = _best_in(frames, v["t_ms"] - p.anchor_before_s * 1000, end, v["t_ms"])
        if f:
            f.reasons.append("voice")
            if v.get("text"):
                f.voice.append(v["text"])

    kept: list[Frame] = []
    for f in usable:
        if kept:
            last = kept[-1]
            walked = f.motion["walked_m"] - last.motion["walked_m"]
            h0, h1 = last.motion.get("heading_deg"), f.motion.get("heading_deg")
            turn = abs(angle_diff(h0, h1)) if h0 is not None and h1 is not None else 0.0
            if walked >= p.max_gap_m:
                f.reasons.append("distance")
            elif float(f.emb @ last.emb) < p.novelty_sim:
                if walked >= p.min_gap_m:
                    f.reasons.append("novel")
                elif turn >= p.turn_deg:
                    f.reasons.append("turn")
        if f.reasons:
            kept.append(f)
    return kept


@dataclass
class Group:
    frames: list[Frame]  # the first one is the representative

    @property
    def rep(self) -> Frame:
        return self.frames[0]

    @property
    def anchored(self) -> bool:
        return any(f.anchored for f in self.frames)


def _rep_first(frames: list[Frame]) -> list[Frame]:
    rep = max(frames, key=lambda f: (f.anchored, f.sharpness))
    return [rep] + sorted((f for f in frames if f is not rep), key=lambda f: (f.rec_id, f.t_ms))


def group_frames(kept: list[Frame], p: SelectParams, sim: np.ndarray, same: np.ndarray
                 ) -> tuple[list[Group], dict[tuple[int, int], float]]:
    """Fold keyframes of the same place into groups. `sim`: [n, n] place similarity, `same`: [n, n] bool, the
    pair is verified as one place (verify.match_pairs). Returns the groups (in first-seen order) and the
    similarity between groups that look alike but were not folded (possible same place), keyed by group index pairs."""
    n = len(kept)
    if not n:
        return [], {}
    emb = np.stack([f.emb for f in kept])
    dup = emb @ emb.T >= p.dup_sim
    close_in_time = np.array([[a.rec_id == b.rec_id and abs(a.t_ms - b.t_ms) < p.revisit_min_s * 1000
                               for b in kept] for a in kept])

    # Recent near-duplicates within a walk: drop the unanchored one (or fold if both are anchored).
    alive = np.ones(n, bool)
    members: list[list[int]] = [[i] for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if alive[i] and alive[j] and close_in_time[i, j] and dup[i, j]:
                if kept[j].anchored and kept[i].anchored:
                    members[i] += members[j]
                    alive[j] = False
                elif kept[j].anchored:
                    alive[i] = False
                    members[j] += [m for m in members[i] if m != i]
                else:
                    alive[j] = False
                    members[i] += [m for m in members[j] if m != j]

    # Revisits and other walks: complete-linkage merge of the remaining groups; every cross pair must be a
    # verified same place and not just the next steps of one walk. The most similar such pair of groups goes first.
    groups = [members[i] for i in range(n) if alive[i]]
    same = same & ~close_in_time
    while True:
        best, pair = -np.inf, None
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                ix = np.ix_(groups[a], groups[b])
                if same[ix].all() and sim[ix].mean() > best:
                    best, pair = sim[ix].mean(), (a, b)
        if pair is None:
            break
        a, b = pair
        groups[a] += groups.pop(b)

    groups.sort(key=lambda g: min((kept[i].rec_id, kept[i].t_ms) for i in g))
    hints = {}
    for a in range(len(groups)):
        for b in range(a + 1, len(groups)):
            cross = [(i, j) for i in groups[a] for j in groups[b] if not close_in_time[i, j]]
            s = max((float(sim[i, j]) for i, j in cross), default=0.0)
            if s >= p.hint_sim:
                hints[(a, b)] = round(s, 3)
    return [Group(_rep_first([kept[i] for i in g])) for g in groups], hints


def cap_groups(groups: list[Group], hints: dict, p: SelectParams) -> tuple[list[Group], dict]:
    """Drop the most redundant unanchored groups until at most max_candidates remain."""
    if len(groups) <= p.max_candidates:
        return groups, hints
    emb = np.stack([g.rep.emb for g in groups])
    sim = emb @ emb.T
    np.fill_diagonal(sim, -1)
    keep = list(range(len(groups)))
    while len(keep) > p.max_candidates:
        droppable = [i for i in keep if not groups[i].anchored] or keep
        worst = max(droppable, key=lambda i: max(sim[i, j] for j in keep if j != i))
        keep.remove(worst)
    remap = {old: new for new, old in enumerate(keep)}
    hints = {(remap[a], remap[b]): s for (a, b), s in hints.items() if a in remap and b in remap}
    return [groups[i] for i in keep], hints
