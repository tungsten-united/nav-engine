# nav-engine

Navigation engine for a blind-navigation app. The **recorder** (phone web app + API) captures free walks through a
place; the **map pipeline** turns all walks of a place into one semantic topological map. Next: localization.

## Layout
- `nav/schemas/recording.py` — the recording format (schema_version 1). Source of truth for what is stored.
- `nav/recorder/app.py` — FastAPI app factory: `/api/v1` (+ map builds) + the debugging frontend; optional token,
  CORS, no-delete for the public deployment; `store.py` — filesystem storage, idempotent uploads, `finalize`
  (chunks → `video.mp4`, batches → parquet/jsonl).
- `frontend/` — the debugging frontend, kept here for reference and direct API testing (the frontend team copies it):
  phone app at `/`, `dash/` (recordings, uPlot from jsdelivr), `viz/` (map viewer, SVG). Vanilla JS, no build step.
  `config.js` sets the API base (default: the Cloud Run worker; `?api=` overrides) and the token.
- `nav/schemas/map.py` — the map format: `MapDraft` (what the LLM writes; its JSON Schema must stay valid for
  Anthropic structured outputs) and `PlaceMap` (stored: draft + evidence, positions, distances, build info).
- `nav/mapping/` — map pipeline: `prepare.py` (per-recording cache: frames, embeddings, transcripts), `frames.py`,
  `motion.py`, `models.py` (DINOv2, MegaLoc, ALIKED+LightGlue, Whisper), `select.py` (candidates), `verify.py`
  (same-place pairs: MegaLoc retrieval + LightGlue check), `llm.py` (Claude Code job + prompt),
  `build.py` (orchestration + finish), `quality.py` (TypeSafe Jev check), `store.py`, `calibrate.py` (video yaw,
  sensor-to-video offset, FOV), `references.py` (reference images + embeddings for live VPR), `review.py`
  (human review overlay `maps/<id>/review.json`; published map.json = latest build + review).
- `nav/navigation/` — reference code for the live side: `route.py` (routes over verified edges), `pdr.py`
  (causal step detector + GraphTracker, `nav map pdr-replay`).
- `docs/` — the navigation-side design (`main_idea.md`, `navigation_stack.md`, by LPA, from
  tungsten-united/demo-repository@navigation) and `map.schema.json` (regenerate: `nav map schema --out ...`; a test checks it).
- `nav/mapping/inference.py` — every model call on files: local models, or the inference host when `NAV_INFER_URL` is set.
- `nav/infer/` — inference host: `server.py` (`nav infer serve`: models + Claude Code map job over HTTP, one GPU slot,
  bounded queues, 503 + Retry-After), `client.py` (chunking, retries, Cloudflare service token).
- `nav/cli.py` — `nav serve`, `nav infer serve`, `nav rec list|show|finalize|import|push|pull`,
  `nav map setup|build|llm|refinish|check|review|route|pdr-replay|schema|list|show|push|pull`.
- `Dockerfile` + `deploy/cloudrun.sh` — the Cloud Run worker; `deploy/nav-infer.service`, `deploy/nav-recorder.service` — helium units.
- `data/recordings/<rec_id>/` — recordings (gitignored). Shared copy for the team: `gs://tungsten-united-nav-recordings/recordings/<rec_id>/`
  (project `tungsten-united`, not the active gcloud project `kotie-app`), via `nav rec push|pull` (`gcloud storage rsync`, no deletes, skips `raw/`).
- `data/maps/<map_id>/` — maps (gitignored), shared under `gs://tungsten-united-nav-recordings/maps/<map_id>/` via
  `nav map push|pull`; `data/cache/<rec_id>/` — preprocessing cache, safe to delete.

## Run / test
```sh
uv sync && uv run pytest
deploy/cloudrun.sh                         # build the Dockerfile, push, deploy nav-api (rerun after every change)
systemctl --user restart nav-infer         # inference host, 127.0.0.1:8451 (deploy/nav-infer.service)
journalctl --user -u nav-infer -f
```
Deployment (2026-10-10):
- **nav-api** on Cloud Run, project `tungsten-united`, `europe-southwest1`:
  https://nav-api-613464313064.europe-southwest1.run.app. The bucket is mounted at `/data` (GCS FUSE). One instance;
  `NAV_API_TOKEN` gates `/api/*` (`deploy/cloudrun.sh token` prints it); deleting is off. Secrets come from Secret
  Manager (`deploy/cloudrun.sh secrets` reloads them from this machine). Logs:
  `gcloud run services logs read nav-api --region europe-southwest1 --project tungsten-united`.
- **nav-infer** on helium, published as https://helium-nav-infer.oscarmartinez.dev (`tunnel-expose nav-infer 8451`).
  The worker sends the `helium-clients` service token. This machine only does inference for the cloud.
- **nav-recorder** (the old all-in-one on helium-nav.oscarmartinez.dev, `tunnel-expose nav 8450`) is still running
  for now; after cutover helium only pulls maps (`nav map push` would overwrite the cloud's map.json).
Cloudflare Access gates every `helium-*` host: browser login for people, the `helium-clients` service token for
workers (`~/.cloudflared/access-service-token.env`, headers `CF-Access-Client-Id/Secret`). The app has no auth of its own.

Browser end-to-end check: headless `/usr/bin/chromium` via `uvx --from playwright python` with
`--use-fake-device-for-media-stream --use-fake-ui-for-media-stream`, sensor permissions granted, synthetic
`devicemotion` / `deviceorientationabsolute` events dispatched from the page. Run it against a throwaway
`NAV_DATA_DIR` on another port (open the page with `?api=http://127.0.0.1:<port>`), never the live service:
recordings made there land in the team bucket.

## Things learned the hard way
- Chrome's MP4 MediaRecorder only emits data at keyframes (in practice at stop); WebM honours the timeslice.
  Keep WebM first in `VIDEO_MIMES` and `videoKeyFrameIntervalDuration: 1000`.
- Cloudflare edge-caches `.js/.css` and the zone forces a 4 h browser TTL. `index.html` is served with
  `?v=<content hash>` on assets and `Cache-Control: no-cache`; keep it that way or phones get a stale app.js.
- Chrome now has `DeviceMotionEvent.requestPermission`; a denial must not abort recording (video + voice still useful).
- Chrome (Android) reports `rotationRate` alpha/beta/gamma around device x/y/z, not z/x/y as the spec says; the
  recorder stores gx/gy/gz = beta/gamma/alpha. `motion.py` picks the layout whose turns match the compass.
- Handheld walking can be very smooth (|a| std 0.1–0.4 m/s²): fixed step thresholds find nothing; use adaptive ones.
- Jev Noul wording decides its calibration: "relies on sight anywhere" fired on glass doors and tables (0.5-0.8);
  naming what only sight can notice, and listing materials/furniture as fine, separated 0.9 from <0.45.
- Measure an edge's turn from the heading at the end of the source pass vs the next seconds of walking, not from
  the path geometry around the node centre: that booked the turn made while arriving at a spot onto leaving it,
  and made a correct "bear right" look like a contradiction (and fooled the Jev check that compared against it).
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
- Two step detectors disagree on Itnig (offline band-pass in motion.py ~153 steps, causal StepDetector in
  navigation/pdr.py ~97); the tracker re-learns step length at each node, but a walk with counted steps is needed.
- Status is per direction: an edge walked target -> source has `status` inferred and `reverse_status` observed.
- `nvidia-smi` may report "Driver/library version mismatch" after a driver update without reboot; CUDA still works.

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

## Conventions
- Work on branch `omm/first_draft`; push with `git push` (gh credential helper is configured for this repo).
- Data sharing: the server does not upload by itself. After new recordings are finalized, run `uv run nav rec push`
  (from the main checkout, so it sees the live `data/`) to keep the team bucket current; check with `--dry-run` first.
  Teammates with access: lpall98@gmail.com (objectAdmin). Grant others per README → "Shared dataset".
- Propose before large changes; the user reviews plans first.
