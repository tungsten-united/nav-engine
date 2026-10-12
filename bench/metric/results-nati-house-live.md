## nati-house: captured live calls against the walk's model (719 model frames, 13003 pairs)

Model: [{"registered": 308, "points": 8436, "walks": {"20261010T195025Z_bf0e": 308}}, {"registered": 72, "points": 3008, "walks": {"20261010T195025Z_bf0e": 72}}, {"registered": 58, "points": 2802, "walks": {"20261010T195025Z_bf0e": 58}}]
Model -> map frame: [{"registered": 308, "frames": 308, "inliers": 175, "scale": 1.1356, "track_vs_model_median_m": 1.27, "track_vs_model_p90_m": 3.93}, {"registered": 72, "frames": 72, "inliers": 48, "scale": 0.1716, "track_vs_model_median_m": 0.76, "track_vs_model_p90_m": 3.18}, {"registered": 58, "frames": 58, "inliers": 57, "scale": 0.6363, "track_vs_model_median_m": 0.51, "track_vs_model_p90_m": 0.91}, {"registered": 55, "frames": 55, "inliers": 43, "scale": 0.5903, "track_vs_model_median_m": 0.6, "track_vs_model_p90_m": 2.24}, {"registered": 54, "frames": 54, "inliers": 46, "scale": 0.3187, "track_vs_model_median_m": 0.52, "track_vs_model_p90_m": 2.03}, {"registered": 52, "frames": 52, "inliers": 52, "scale": 0.1561, "track_vs_model_median_m": 0.47, "track_vs_model_p90_m": 1.07}, {"registered": 49, "frames": 49, "inliers": 42, "scale": 0.2252, "track_vs_model_median_m": 1.01, "track_vs_model_p90_m": 1.75}, {"registered": 32, "frames": 32, "inliers": 32, "scale": 0.0476, "track_vs_model_median_m": 0.2, "track_vs_model_p90_m": 0.39}]

| session | calls | frames | posed ≥20 | posed ≥40 | posed ≥80 | calls posed (≥40) | implausible jumps | live answers then |
|---|---|---|---|---|---|---|---|---|---|
| 03c24e20 | 126 | 130 | 93% | 92% | 75% | 91% | 3/114 | lost 100, uncertain 26 |
| anonymou | 23 | 23 | 100% | 100% | 100% | 100% | 0/0 | lost 23 |
| b542ab5a | 7 | 8 | 38% | 38% | 38% | 43% | 0/2 | lost 7 |

Estimated focal length of posed live frames: median 737 px at 1024 px (p10 698, p90 985).

Current localizer, memory: `memory          queries  156 | node right   -  | decisive (lead>=0.08)   9%, right   -  | position - | belief located 2/3 (median 11 s), commits 9 right   -  | 7 ms`

Current localizer, verify: `verify          queries  156 | node right   -  | decisive (lead>=0.08)   9%, right   -  | position - | belief located 2/3 (median 11 s), commits 9 right   -  | 195 ms`

Latency per frame (median): extract 51 ms, retrieve 30 ms, match 210 ms, pnp 9 ms; total 307 ms
