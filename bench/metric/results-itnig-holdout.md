## itnig: held-out walks (1124 model frames, 14807 pairs, global mapping)

All-walks model: [{"registered": 1123, "points": 54443, "walks": {"20261009T202101Z_7f42": 227, "20261010T124519Z_6b88": 159, "20261010T110802Z_601a": 310, "20261010T130629Z_ddb1": 351, "20261010T140614Z_b6a0": 76}}]
Model -> map frame: {"frames": 1047, "inliers": 652, "scale": 2.2401, "track_vs_model_median_m": 1.27, "track_vs_model_p90_m": 2.99}

| walk | queries | truth | posed ≥20 | posed ≥40 | posed ≥80 | ≤1 m & ≤10° | error median / p90 | rotation median / p90 | first fix |
|---|---|---|---|---|---|---|---|---|
| 20261009T202101Z_7f42 | 89 | 89 | 100% | 99% | 96% | 98% | 0.07 / 0.26 m | 0.4 / 1.0° | 0.0 s |
| 20261010T110802Z_601a | 122 | 122 | 88% | 81% | 61% | 76% | 0.1 / 0.62 m | 0.4 / 2.66° | 0.0 s |
| 20261010T124519Z_6b88 | 61 | 60 | 98% | 97% | 93% | 97% | 0.02 / 0.09 m | 0.2 / 0.52° | 1.2 s |
| 20261010T130629Z_ddb1 | 138 | 138 | 99% | 96% | 86% | 93% | 0.1 / 0.3 m | 0.4 / 1.49° | 0.0 s |
| 20261010T140614Z_b6a0 (not in the map) | 29 | 29 | 100% | 100% | 90% | 93% | 0.1 / 0.31 m | 0.5 / 1.1° | 0.0 s |
| **all** | 439 | 438 | 96% | 93% | 82% | **90%** | 0.07 / 0.33 m | 0.4 / 1.6° | |

Current localizer on the same degraded frames, and PnP's node (the region its pose is in):

| walk | method | node right | position error, map frame (median / p90) | its memory frame's pose vs truth (median / p90) | belief located | commits right |
|---|---|---|---|---|---|---|
| 20261009T202101Z_7f42 | PnP (≥40) | 93% | - | - | - | - |
| 20261009T202101Z_7f42 | memory | 46% | 1.47 / 3.43 m | 1.01 / 3.0 m | never | 0/0 |
| 20261009T202101Z_7f42 | verify | 45% | 1.55 / 3.43 m | 1.01 / 3.0 m | never | 0/0 |
| 20261010T110802Z_601a | PnP (≥40) | 75% | - | - | - | - |
| 20261010T110802Z_601a | memory | 40% | 2.38 / 5.55 m | 2.23 / 6.92 m | 138.519 | 1/1 |
| 20261010T110802Z_601a | verify | 40% | 2.38 / 5.55 m | 2.23 / 6.92 m | 138.519 | 1/1 |
| 20261010T124519Z_6b88 | PnP (≥40) | 88% | - | - | - | - |
| 20261010T124519Z_6b88 | memory | 60% | 1.28 / 2.63 m | 0.64 / 2.23 m | 3.481 | 2/2 |
| 20261010T124519Z_6b88 | verify | 60% | 1.19 / 2.6 m | 0.59 / 2.19 m | 3.481 | 3/4 |
| 20261010T130629Z_ddb1 | PnP (≥40) | 91% | - | - | - | - |
| 20261010T130629Z_ddb1 | memory | 56% | 1.97 / 4.42 m | 1.39 / 4.28 m | 2.466 | 1/1 |
| 20261010T130629Z_ddb1 | verify | 54% | 1.97 / 4.42 m | 1.35 / 4.28 m | 2.466 | 3/3 |
| 20261010T140614Z_b6a0 | PnP (≥40) | 93% | - | - | - | - |
| 20261010T140614Z_b6a0 | memory | 31% | 4.05 / 5.93 m | 2.71 / 5.24 m | never | 0/0 |
| 20261010T140614Z_b6a0 | verify | 31% | 4.05 / 5.93 m | 2.71 / 5.24 m | never | 0/0 |

Latency per frame (median): extract 29 ms, retrieve 26 ms, match 173 ms, pnp 6 ms; total 243 ms
