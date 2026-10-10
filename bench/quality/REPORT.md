# Map quality check: no check vs keyword check vs Jev

Run: `uv run python bench/quality/bench.py run` (needs `TYPESAFE_API_KEY`). Data: `dataset.json` (73 spoken items from
the 4 Itnig maps built so far: 2 from the older prompt, rich in visual wording, 2 from the current one),
`labels.json` (reference labels), `results.json` (every number and every error below).

| set | n (defects) | no check F1 | keyword check F1 | Jev F1 | Jev AUROC |
|---|---|---|---|---|---|
| visual, real map text | 73 (37) | 0.00 | **1.00** | 0.99 | 1.00 |
| visual, synthetic (unseen phrasing + keyword traps) | 36 (18) | 0.00 | 0.70 | **0.83** | 0.92 |
| turn contradicts the walk, synthetic | 64 (32) | 0.00 | 0.72 (recall 0.56) | **1.00** | 1.00 |

Cost of Jev for all 173 judgments: 83,774 input tokens = **$0.0035**, **0.85 s** wall (parallel requests).
A whole map is ~14k tokens, about $0.0006. For scale, the reference labelling of the 73 items by Opus (Claude Code,
high effort, one batched call) cost $0.26 at API list price and took 41 s.

## What it shows
- **Turn check: this is where Jev earns its place.** The keyword check ("does the instruction contain the given
  side?") misses 14 of 32 contradictions, because instructions name both sides ("with the cooler on your right, turn
  left…"). Jev caught all 32 with no false alarms.
  *Correction:* an earlier version of this report cited e3 on the live map ("Turn right…" where the walk turned 71°
  left) as a catch. The 71° was wrong: the pipeline measured the turn around the node's centre and booked the left
  turn made while arriving at the tables onto leaving them. Measured from the heading at the end of the pass, the
  departure is a 20° bear right, as the instruction says. Jev answered the question it was given correctly; the
  question was wrong.
- **Visual wording, real outputs: the keyword list is already enough.** It scores 1.00, but the list was written
  after reading these same maps, so that number is optimistic. Jev scores 0.99 without having seen them (its one miss,
  "long white work tables", sits at p = 0.59, just under the 0.6 threshold).
- **Visual wording, new phrasing: Jev generalises better** (F1 0.83 vs 0.70). The keyword check misses visual cues
  outside its list ("a poster of a mountain", "you can spot the exit") and fires on 12 traps ("Look out for a low
  bench", "sign-in desk", "screen door"). Jev's errors are 6 traps that still contain colour or darkness words
  ("red-hot radiator", "painted plaster, smooth to the touch", "dark at night", arguably visual) and 1 miss
  ("you can spot the exit door").

## Caveats
- Reference labels for the real set come from one LLM judge (Opus, high effort) given the same definition as Jev's
  question. Every disagreement with either detector was read by hand; there was one, and it agreed with the judge.
- Synthetic clauses were written by hand (10 visual, 10 traps); the turn set pairs each real instruction whose first
  turn the judge extracted with a measured turn to the same and to the other side. Small sets: treat differences of a
  few points as noise.
- Arrival / followability Scores are not benchmarked: there is no objective truth for them yet.

## Threshold
Re-run with every probability stored (`results.json → visual thresholds`), Jev's sight question scores F1
real / synthetic: 1.00 / 0.77 at 0.5, 0.99 / 0.81 at 0.6, **0.96 / 0.87 at 0.7**, 0.94 / 0.85 at 0.8. The flag
threshold is 0.7.

## Decision (updated after edges became structured steps)
Edges now carry `steps` with an action enum, and code checks the first step against the walk's turn. So Jev's turn
question checks the words instead: does the spoken instruction turn the same way as its first step? That is the
"text vs a given side" judgment benchmarked above (Jev 1.00, keyword 0.72), and it now applies to every edge whose
first step turns, not only to clear measured turns. Visual wording stays keyword first, Jev as backstop at 0.7.
On the latest Itnig map (new rules) Jev flags nothing at 0.7; cost per map stays around $0.0005.


# Generation rules: before and after structured steps (bench/prompt/compare.py)

Same 30 candidates, one Opus run (medium effort) each, both post-processed with the same code:

| | old rules | new rules |
|---|---|---|
| spoken distance vs walked, mean gap | 2.3 m | **0.2 m** |
| spoken text with visual words (keyword) | 5 | **0** |
| flagged by Jev (sight, threshold 0.6 at the time) | 3 | 1 (a false positive: "tiled splashback") |
| edges presented as walked that weren't | 2 (found by code; no field to say so) | **0** (the one inferred edge is declared) |
| clear turns (≥45°) whose first step contradicts the walk | n/a (no steps) | **0 of 3** |
| candidates unused | 4 | 2 |

One run per side, so read small differences as noise; the distance gap and the visual wording are the clear wins.
