# nav-engine

Navigation engine of Orient, a voice-first indoor guidance web app for blind users (tungsten-united/project-description).
The **recorder** (Orient's app at `/map`, project-description `apps/web/src/mapping/`, + map-api) captures free walks
through a place; the **map pipeline** turns all walks of a place into one semantic topological map; **nav-api** serves
the live side (maps, routes, localization) to Orient's orchestrator (tungsten-united/orient-orchestrator), its only
live caller. For guidance Orient's phone app talks only to the orchestrator; its `/map` page (mapping, for the team)
calls map-api directly. `frontend/` here is a debugging frontend, not Orient's app.

## Layout
- `nav/schemas/recording.py` — the recording format (schema_version 1). Source of truth for what is stored.
- Two HTTP services, one image (`nav serve --app map-api|nav-api`, env `NAV_APP`); shared bits (token, CORS,
  no-cache, files under Cloud Run's cap, cache-busted pages) in `nav/web.py`:
  - `nav/map_api/app.py` — **map-api**: recordings, map builds, review, map files + the debugging frontend; no-delete
    for the public deployment.
  - `nav/nav_api/app.py` — **nav-api**: the lean, read-only live API for Orient's orchestrator: published maps,
    routes, localization (`maps.py`: published maps + reference embeddings cached, reloaded when map.json changes).
    Formats in `nav/schemas/navigation.py`. `log.py`: one JSON line per localize/route call for Cloud Logging, `message`
    prefixed `[nav-api]`, the orchestrator's field names (`clientId` from `X-Client-Id`), `commit` from `NAV_COMMIT`.
    `capture.py`: with `NAV_CAPTURE_DIR` every localize/route call is recorded whole (frames, motion, answer or error)
    under `<map_id>/<date>/<sessionId>/`, on a background thread; on Cloud Run that is the bucket's `live/` prefix,
    the only place nav-api may write (`deploy/cloudrun.sh capture-setup`). Read back with `nav/navigation/captured.py`
    (`nav live pull|sessions|logs`): the test set for live localization.
- `nav/recorder/store.py` — recording storage, idempotent uploads, `finalize` (chunks → `video.mp4`, batches →
  parquet/jsonl).
- `frontend/` — the debugging frontend, for the team and for testing the APIs by hand (not Orient's app):
  an index at `/` (links, and the token-carrying link to Orient's `/map`), `dash/` (recordings, uPlot from
  jsdelivr), `viz/` (map viewer, SVG), `loc/` and `nav/` (nav-api debuggers; `loc/loc-core.js` is shared). Vanilla JS, no build step. `config.js`: `NAV` = map-api, `NAV.nav` =
  nav-api, each with its base (`?api=`, `?navapi=`) and token (`?token=`, `?navtoken=`).
- `nav/schemas/map.py` — the map format: `MapDraft` (what the LLM writes; its JSON Schema must stay valid for
  Anthropic structured outputs) and `PlaceMap` (stored: draft + evidence, positions, distances, build info).
- `nav/mapping/` — map pipeline: `prepare.py` (per-recording cache: frames, embeddings, transcripts), `frames.py`,
  `motion.py`, `models.py` (DINOv2, MegaLoc, ALIKED+LightGlue, Whisper), `geometry.py` (the 3D model: COLMAP global
  mapping of every walk's video, or incremental when global flings frames; put in the map frame and checked against
  each walk's motion; the GPU stage a job on helium's queue via gpu-run; live `solve_pose`), `topology.py` (code's
  layout: spots by position, nodes kept to nearby spots, visits, edges where the walks went, steps from the paths,
  node relations), `assemble.py` (layout + the two LLM passes -> MapDraft -> PlaceMap), `select.py` (candidates),
  `verify.py` (same-place pairs: MegaLoc retrieval + LightGlue check, only for builds without a model), `llm.py`
  (Claude Code job, two passes that only add meaning: places, routes), `build.py` (orchestration; `finish_map` is the
  pre-metric-1 finish for refinishing old builds), `quality.py` (TypeSafe Jev check), `store.py`, `calibrate.py` (video yaw,
  sensor-to-video offset, FOV), `odometry.py` (metres walked from the floor seen in the video, fused with steps),
  `references.py` (reference images + embeddings for live VPR), `memory.py` (the visual memory, localization's
  "where": the model's frames with their poses and MegaLoc embeddings, or for builds without a model every usable
  frame placed by the tracks; nodes stay the "what" and get frames by position at load, so a review moves regions
  with it; `GeometryObs`: posed frames' 3D observations for live poses; `nav map memory` adds a memory to an old build), `review.py` (human review overlay
  `maps/<id>/review.json`, incl. distances measured on site that set the scale, and `merge_into` to fold nodes into
  one area), `clean.py` (only what the walks back up is published: ≥ 2 walks, weaker edges only to keep the map
  walkable; nodes and edges kept or left out whole; `pruned` says why; `links` keeps the connections from before it
  for localization; published map.json = latest build + review, cleaned).
- `nav/navigation/` — the live side, small and causal: `route.py` (routes over verified edges by default) and
  `localize.py` (VPR against reference embeddings and the visual memory → ranked nodes, confirmed/uncertain/lost,
  and `position`: where the best memory frame was taken; a memory frame scores for the nearest node within
  `NAV_LOC_MEMORY_RADIUS_M`; with a 3D model, `pose`: nav-api has the inference host pose the camera frames against
  the best `NAV_LOC_POSE_K` posed memory frames (`/infer/v1/pose`, live lane, features cached there by
  map/build/frame key, images and 3D observations sent only when it asks), and 40+ inliers confirm the pose's node),
  both served by nav-api;
  `pdr.py` (causal step detector + GraphTracker, `nav map pdr-replay`) and `arrival.py` (node confirmer, ported 1:1
  to `frontend/nav/arrival.js`: /nav's auto-confirm, where a rule is tried before the orchestrator copies it),
  reference code that nothing live runs. `replay.py` replays walks through the orchestrator's loop (`Track`,
  `Episode`, `run_episode`); `belief.py` is that loop on a probability per node (a Bayes filter: steps move it along
  the links, each /localize result's score differences reweigh it) instead of votes; `episodes.py` stitches pieces
  of walks into more routes than were walked, scores a loop on them (nodes reached in time; wrong moves cost) and
  picks its parameters (`nav map tune`, scipy differential evolution, each walk matched against the others'
  references only).
- `docs/` — the navigation-side design (`main_idea.md`, `navigation_stack.md`, by LPA, from
  tungsten-united/demo-repository@navigation, aligned with Orient's architecture) and `map.schema.json` (regenerate: `nav map schema --out ...`; a test checks it).
- `nav/navigation/locbench.py` — `nav map loc-bench`: localization method by method (refs / memory / verify) on the
  map's walks (each against the others; a single walk minus `--exclude-s` around the query) with the truth, or on
  captured live sessions (`--live`) without it; every answer sequence through the belief at the orchestrator's live
  temperature: when it locates, how often it commits, how far from the truth. The scoreboard for localization changes.
- `nav/mapping/inference.py` — every model call on files: local models, or the inference host when `NAV_INFER_URL` is set.
- `nav/infer/` — inference host: `server.py` (`nav infer serve`: models, live poses, the Claude Code map job (a pass
  per call) and geometry jobs over HTTP, one GPU slot where nav-api's live frames go ahead of queued batch calls,
  bounded queues, 503 + Retry-After; a geometry job's GPU stage is `nav infer geometry-gpu` queued with gpu-run, since
  nav-infer is a resident service of helium's GPU queue and batch work must wait its turn), `client.py` (chunking,
  retries, deadlines, Cloudflare service token).
- `nav/cli.py` — `nav serve [--app nav-api]`, `nav infer serve`, `nav rec list|show|finalize|import|push|pull`,
  `nav live pull|sessions|logs` (captured live calls; `logs` saves the session's Cloud Logging entries, the
  orchestrator's included, as `logs.jsonl`),
  `nav map setup|build|llm|refinish|check|review|clean|republish|memory|loc-bench|measure|route|pdr-replay|orchestrator-replay|tune|schema|list|show|push|pull`.
- `Dockerfile` + `deploy/cloudrun.sh` + `deploy/{map-api,nav-api}.env.yaml` — the two Cloud Run services;
  `deploy/nav-infer.service`, `deploy/nav-recorder.service` (helium-nav: map-api serving the debugging pages) — helium units.
- `data/recordings/<rec_id>/` — recordings (gitignored). Shared copy for the team: `gs://tungsten-united-nav-recordings/recordings/<rec_id>/`
  (project `tungsten-united`, not the active gcloud project `kotie-app`), via `nav rec push|pull` (`gcloud storage rsync`, no deletes, skips `raw/`).
- `data/live/<map_id>/<date>/<session>/` — live calls nav-api recorded (gitignored), from `gs://…/live/` via
  `nav live pull`. They show people at the venue: team use only, never in the repo.
- `data/maps/<map_id>/` — maps (gitignored), shared under `gs://tungsten-united-nav-recordings/maps/<map_id>/` via
  `nav map push|pull`; `data/cache/<rec_id>/` — preprocessing cache, safe to delete.

## Run / test
```sh
uv sync && uv run pytest
deploy/cloudrun.sh [deploy] [map-api|nav-api|all]   # build the image, push, deploy (rerun after every change)
systemctl --user restart nav-infer         # inference host, 127.0.0.1:8451 (deploy/nav-infer.service)
journalctl --user -u nav-infer -f
```
Deployment (Cloud Run, project `tungsten-united`, `europe-southwest1`; the bucket mounted at `/data`, GCS FUSE):
- **map-api** — https://map-api-613464313064.europe-southwest1.run.app: one instance, bucket read-write, deleting off,
  serves the debugging frontend (coworkers use it). Token: secret `map-api-token`.
- **nav-api** — https://nav-api-613464313064.europe-southwest1.run.app (the URL of the old all-in-one service): 1–3
  instances, bucket read-only, service account `nav-api` (bucket viewer). Token: secret `nav-api-token` (the token
  from before the split; map-api's started as a copy). Page paths redirect to map-api.
- `NAV_API_TOKEN` gates `/api/*` on each (`deploy/cloudrun.sh token map-api|nav-api` prints it). Secrets come from
  Secret Manager (`deploy/cloudrun.sh secrets` reloads them from this machine). Logs:
  `gcloud run services logs read <service> --region europe-southwest1 --project tungsten-united`.
- **nav-infer** on helium, published as https://helium-nav-infer.oscarmartinez.dev (`tunnel-expose nav-infer 8451`).
  Both services send the `helium-clients` service token. This machine only does inference for the cloud.
- **nav-recorder** on helium-nav.oscarmartinez.dev (`tunnel-expose nav 8450`): `nav serve` (map-api code, local
  data) whose pages call the Cloud Run services; restart it after pulling to get new pages. Helium only pulls maps
  (`nav map push` would overwrite the cloud's map.json).
Cloudflare Access gates every `helium-*` host: browser login for people, the `helium-clients` service token for
workers (`~/.cloudflared/access-service-token.env`, headers `CF-Access-Client-Id/Secret`). nav-infer has no auth of its own.

Browser end-to-end check: headless `/usr/bin/chromium` via `uvx --from playwright python` with
`--use-fake-device-for-media-stream --use-fake-ui-for-media-stream`, sensor permissions granted, synthetic
`devicemotion` / `deviceorientationabsolute` events dispatched from the page. Run it against a throwaway
`NAV_DATA_DIR` on other ports (open the page with `?api=http://127.0.0.1:<map-api port>&navapi=...<nav-api port>`),
never the live services: recordings made there land in the team bucket. The recorder is Orient's: run its dev server
with `VITE_MAP_API_URL=http://127.0.0.1:<map-api port>` and open `/map?token=...` (the throwaway map-api's
`NAV_CORS_ORIGIN_REGEX` must allow the dev origin). For /loc and /nav, copy a published map
(map.json + its build folder) and a recording into the throwaway dir; a throwaway `nav infer serve --port ...` keeps
the live inference host out of it.

## Things learned the hard way
- Chrome's MP4 MediaRecorder only emits data at keyframes (in practice at stop); WebM honours the timeslice.
  Keep WebM first in `VIDEO_MIMES` and `videoKeyFrameIntervalDuration: 1000` (now in Orient's `mapping/recorder.ts`).
- Cloudflare edge-caches `.js/.css` and the zone forces a 4 h browser TTL. `index.html` is served with
  `?v=<content hash>` on assets and `Cache-Control: no-cache`; keep it that way or phones get a stale script.
- Chrome now has `DeviceMotionEvent.requestPermission`; a denial must not abort recording (video + voice still useful).
- Chrome (Android) reports `rotationRate` alpha/beta/gamma around device x/y/z, not z/x/y as the spec says; the
  recorder stores gx/gy/gz = beta/gamma/alpha. `motion.py` picks the layout whose turns match the compass.
- Handheld walking can be very smooth (|a| std 0.1–0.4 m/s²): fixed step thresholds find nothing; use adaptive ones.
- Jev Noul wording decides its calibration: "relies on sight anywhere" fired on glass doors and tables (0.5-0.8);
  naming what only sight can notice, and listing materials/furniture as fine, separated 0.9 from <0.45.
- Measure an edge's turn from the heading at the end of the source pass vs the next seconds of walking, not from
  the path geometry around the node centre: that booked the turn made while arriving at a spot onto leaving it,
  and made a correct "bear right" look like a contradiction (and fooled the Jev check that compared against it).
- map-api rebuilds a place's map when one of its walks finalizes (`NAV_AUTO_BUILD`, default on; tests/conftest.py
  turns it off so test recordings never start a real build). The queue is in memory: a restart mid-build loses it.
- `nav map refinish` reuses the saved Claude answer: use it to iterate on post-processing without paying again.
- Generic DINOv2 similarity says "same kind of scene", not "same place" (two ends of one hall: 0.81). Use MegaLoc
  for place identity. LightGlue inliers in one open hall: near-identical views 490–870, different spots 100–300, so
  inliers alone fold a whole room; within-walk near-duplicates stay a DINOv2 call. Tune with `pairs.json`.
- LightGlue needs `opencv-python`; `pyproject.toml` overrides it (and matplotlib) away to keep the headless build.
- Cloud Run cuts any HTTP/1 response with a Content-Length over 32 MiB, and the browser sees a 500. Files over
  8 MiB are served in ranges of at most 8 MiB, or streamed without a length (`send_file` in app.py).
- Cloud Build's default service account has no rights in tungsten-united, so `deploy/cloudrun.sh` builds the image
  with local docker and pushes it to Artifact Registry.
- Behind Cloud Run IAM, `Authorization` carries Google's identity token, so the app's token is also accepted as `?token=`.
- A cold inference host loads MegaLoc on its first place call (~10 s+), longer than nav-api's live deadline (8 s):
  the first frame after nav-infer restarts gets a 503; the next ones take ~30 ms on the GPU.
- Two step detectors disagree on Itnig (offline band-pass in motion.py ~153 steps; the phone's causal detector,
  navigation/pdr.py StepDetector = Orient's tracker.ts, also its /map recorder, 137, hand sway included). The map's
  `live_step_length_m` is metres per phone step: nav-api turns the phone's stepCount into metres walked with it.
- Steps x the recorder's 0.7 m overstated Itnig 3x (107 m vs 32 m from floor odometry; Main entrance -> Drinks area
  measured 7 m on site, odometry 5.9 m): the bout gate passes hand sway while standing and talking (moving 94% of
  the walk), and filming makes slow ~0.3 m steps. Distances now come from odometry.py; a measured distance fixes
  the scale (assumed phone height). Rescaling a step-based build is uniform and can't fix per-edge errors: rebuild.
- Floor odometry measures each half second as soon as its frames are decoded: holding the whole walk's frames took
  ~5 MB per second of video (Itnig 854 MB peak, now 373 MB) on map-api, one 4 GiB instance that also serves the API.
- On macOS, opencv-python-headless and av each bundle libavdevice ("Class AVFFrameReceiver is implemented in
  both"): harmless warning; the Linux builds don't have it.
- Status is per direction: an edge walked target -> source has `status` inferred and `reverse_status` observed.
- Walks in the opposite direction don't fold with the first walk's frames (the other side of a spot looks
  different), so each walk's dead-reckoned x/y is in its own frame: compare across walks only via map_x_m/map_y_m
  (aligned through folded candidates) or after finish (aligned through nodes). The LLM joins walks by words, order
  and distances; continuity keeps edge orientation so per-direction review stays put.
- `nvidia-smi` may report "Driver/library version mismatch" after a driver update without reboot; CUDA still works.
- Live phone frames score lower than the mapping walks' replays (Itnig, 2026-10-10: 0.17-0.49 against a confirm score
  of 0.45), so a single absolute `confirmed` rarely fires after the first node. Hop following votes on which node
  leads (margin), over several frames, not on the absolute score.
- /nav's confirmer does the same (2026-10-10 /nav test, nav-api logs: the right node scored 0.36-0.48, leading by
  0.08-0.16): it votes on the leader and its lead with its own thresholds (0.30 / 0.04), gated by nav-api's
  `plausible` (steps, no compass). The phone compass was 35-45° off the map's bearings, so headings don't gate
  arrival, and neighbours of the hop seen from it are not "off route".
- A merged node (review `merge_into`) is an area whose edges lead to different parts of it: routes and the walked-
  distance gate cross it only in and out of one part (`*_part` on edges). Merging Itnig's three table nodes without
  that left the stage and kitchen unreachable over observed edges from the entrance, or gave 13 m shortcuts.
- The LLM's `confidence` does not follow the evidence (Itnig: Bathroom "high", seen once by one walk; 10 of 19 edges
  walked once, 2 never): the cleaner (clean.py) judges by walks instead. Connectivity has to be checked in the walked
  direction (route.py's `observed`): undirected, it kept Bathroom's edge walked only out of it, and dropped a
  "shortcut" whose way around was walkable only backwards.
- Cleaning must not change localization or what is said (Itnig, 2026-10-10, live test after the cleaned map went
  out): folding a one-walk spot into one edge joined two instructions into 263 characters; Orient speaks 240, up to
  the last whole sentence, so the user stopped at the folded spot, 4 m short, and the hop never confirmed ("Please
  hold still" in a loop). Pruned edges also moved spots apart for the walked-distance gate (kitchen -> centre tables
  2.9 -> 5.5 m). Folding is gone; localization measures over `links`. tests/test_clean.py checks both on 200 random
  maps and on tests/fixtures/itnig.

- iOS Safari reports accelerationIncludingGravity pointing down (Nati House, 2026-10-10, iPhone: y -9.4 held
  upright; four Android walks +9.1 to +9.4). Taken as up, it put the floor above the camera: the floor odometry
  measured nothing (0 of 323 segments), every distance fell back to steps (158 m for a 3-minute walk in a flat,
  69 m with the floor measured), the calibration failed, pitch read +16° and turns integrated backwards.
  `motion.py` turns gravity to agree with the phone's tilt (deviceorientation beta/gamma). The live side uses
  |acceleration| and the tilt angles, so it was never affected.
- Replays must stop at the goal: a walk replayed to its route's goal kept sending the frames of the rest of the walk
  with the user still "at" the goal (Itnig ddb1: goal at 10 s, frames to 170 s), so following the walker on was
  scored wrong. Walk episodes end GOAL_TAIL_S after the goal, as stitched ones do. Replays against a walk's own
  references flatter (frames 0.4 s from a reference are near copies): tune and judge with the other walks' only.
- Live frames separate nodes less than replays (Itnig median lead 0.05 live vs 0.068 replayed; Glories walk-back
  0.024, walked against its one walk's direction), and the belief's tuned optimum is a cliff (temperature 0.25: 0
  wrong of 966 episode nodes, 0.18: 203). The orchestrator runs 0.18 (0.25 scaled by the live/replay lead ratio).
  With walk_scale 3.7 a steady 0.1 lead doesn't gather while walking 3 m links: the belief commits when the user
  pauses. Tune against live traces (top_k 20 answers) once staging has them; replay noise matching is next.
- Area nodes can't be told apart by a picture: in an open hall a view belongs to several of them, and the map gave
  one view different nodes in different walks (Itnig: of 84 confident cross-walk LightGlue matches that "disagreed"
  with the node, 78 were neighbours, and the 9 checked by eye were the same spot). Leads stayed ~0.03 live and the
  belief, with the leader flipping between neighbours, never reached 0.7. Matching to the frame instead (any frame
  of the other walks) put the camera a median 1.5 m from where it was: the visual memory (memory.py) gives
  `position`, and nodes get frames by where they were taken, not by what they show.
- LightGlue proves "same scene", not "same spot": verified frames are often a neighbour's (seen from the tables,
  the stage is the stage). As a strong vote (gain 0.5 from 60 inliers) it made the belief commit 4x as often and
  wrongly (Itnig replays: 56% of commits right vs 73% without it). Conservative (0.3, from 150 to 500), on degraded
  frames, it is what makes every walk locate (memory alone: 3 of 4 Itnig walks; Nati House after 31 s instead of
  2 s) with 95-100% of commits within 3 m. `nav map loc-bench` measures this; tune there, then on captured sessions.
- `cv2.findFundamentalMat` raises on degenerate matches (a few on one line) instead of returning no model:
  `models.geometric_inliers` turns that into 0 inliers, else a blurred live frame is a 500.
- Orient sends live frames at up to 1280 px (JPEG 0.7); the memory is 768 px and ALIKED isn't scale-invariant, so
  the verifier shrinks live frames to 768 first.
- A problem found live becomes a test made from its real data, not only a fix: `tests/fixtures/itnig` (the map
  that stalled guidance after cleaning), `tests/fixtures/glories-basement` (a place walked once: no way back over
  `observed`), `tests/fixtures/voice-languages.json` (Whisper's guesses for short notes that alone came out Spanish
  or Icelandic), `tests/fixtures/sensors` (a minute of an iPhone and an Android walk). Each says in its docstring or
  `about` where it came from.

- The GPU on helium is shared through a queue (managed with `gpu-run` and `gpuq`): agent sessions see no GPU,
  nav-infer is a resident service next to the one queued job, and an agent's exclusive service can hold the job slot
  for hours. Batch GPU work (a build's geometry stage) goes through `gpu-run`; CPU work (COLMAP mapping) stays outside
  the queue so the slot isn't held. `tests/conftest.py` sets `NAV_GEOMETRY=0`: a test build must never queue a job.
- Global mapping (GLOMAP) joins Itnig's 5 walks into one model (1123 of 1124 frames, 2.5 min) where incremental
  mapping didn't finish in an hour; on Nati House's one night walk it joined everything wrongly (frames flung 20 m,
  a stretch glued elsewhere) while incremental gave consistent pieces (544 of 719 frames at 4 fps). So: global first,
  incremental when the global model flings frames (`_jumpy`), every piece kept, each walk checked against its own
  motion (a 1 m jump between consecutive offsets splits stretches; only the main stretch and those within drift of it
  stay posed). A low mean reprojection error (1 px) says nothing about such joins.
- COLMAP's similarity estimator rejects a flat target (the walks' tracks have z = 0): fit the map frame with Umeyama.
- With the model, spots fold by position, so the LightGlue same-place check (and its 300-inlier threshold) is only
  for builds without one. A node's spots must be within 6 m of each other; the LLM no longer decides what is one
  place, nor edges, turns or distances.
- Measure an edge's departure turn from how the walk arrived at the source: by the time it leaves the node's radius
  it already faces the new way, so "heading when leaving the radius" read every turn at a node as straight.

## Product decisions
- A recording is a **free walk through a place**, not a route. `place` (required, whitespace-normalised,
  case-insensitive grouping) is the map key: every walk of a place feeds that place's single map.
- The map pipeline must therefore merge many walks and detect revisits (embedding similarity + tags + voice),
  not assume a linear path.
- Map pipeline MVP (built): 1 fps sharpest frames → quality filter → keyframes per walk (novelty by DINOv2
  embedding + walked distance / turn from the IMU, anchors at tags and voice notes) → fold the same spot across
  walks and revisits (MegaLoc retrieval, ALIKED+LightGlue geometric check; weaker look-alikes become hints)
  → a Claude Code job writes the graph from candidates.json + frames (+ previous.json: keep the last map where it
  holds) → code assigns stable ids by evidence overlap with the last map (continuity.py), checks ids and adds
  evidence, positions and distances. Voice is transcribed locally (Whisper turbo; no OpenAI key here). Later: Anthropic API with
  structured outputs instead of the Claude Code job; code proposing node/edge spans itself.
- nav-api is for Orient's orchestrator, server to server; nav-api is stateless and the orchestrator keeps the user's
  state. Its loop: `localize` (votes over results) until the user is located, `route` once, then along it, each
  `localize` (previous = hop source, expected = hop target, the phone's motion) a vote on reaching the target or a
  node further on (one the camera missed: the route goes on from there; nav-api's walked-distance gate keeps a node
  seen from afar from counting); lost
  (other nodes win, or nothing recognised for a while), it starts over. `nav map orchestrator-replay` models that loop
  (replay.py `OrchestratorLoop`, `--loop votes`) and the proposed replacement for its votes, a probability per node
  (belief.py `BeliefLoop`, the default; the orchestrator's `pipeline::Belief` is its port, orient-orchestrator#16); `nav map tune` picks either's
  parameters on stitched episodes. The phone never calls nav-api. map-api and
  `frontend/` are the team's (mapping, review, debugging); Orient's `/map` page records walks into map-api with its
  token (asked once per phone, never built into the public app). The contract lives in project-description
  `docs/contracts.md` §2: change it there first, then here.
- There is no navigation state machine: route state lives in the orchestrator's session, not here.

## Conventions
- Branch from `main` as `omm/<topic>` and open a PR; push with `git push` (gh credential helper is configured for this repo).
- Data sharing: the server does not upload by itself. After new recordings are finalized, run `uv run nav rec push`
  (from the main checkout, so it sees the live `data/`) to keep the team bucket current; check with `--dry-run` first.
  Teammates with access: lpall98@gmail.com (objectAdmin). Grant others per README → "Shared dataset".
- Propose before large changes; the user reviews plans first.
