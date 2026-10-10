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
- **Turn check: this is where Jev earns its place.** The keyword check ("does the instruction contain the measured
  side?") misses 14 of 32 contradictions, because instructions name both sides ("with the cooler on your right, turn
  left…"). Jev caught all 32 with no false alarms. On the live map it flagged e3: "Turn right…" where the walk turned
  71° left. No free check can read that.
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

## Decision
Keep both. The keyword check stays as the free first line for visual wording; Jev adds the turn check (which nothing
cheap can do) and a semantic backstop for visual wording the list doesn't know, for well under a cent per map.
