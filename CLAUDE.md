# nav-engine

Navigation engine for a blind-navigation app. The **recorder** (phone web app + API) captures free walks through a
place; the **map pipeline** turns all walks of a place into one semantic topological map. Next: localization.

## Layout
- `nav/schemas/recording.py` — the recording format (schema_version 1). Source of truth for what is stored.
- `nav/recorder/app.py` — FastAPI app factory: `/api/v1` + the static phone app; `store.py` — filesystem storage,
  idempotent uploads, `finalize` (chunks → `video.mp4`, batches → parquet/jsonl).
- `nav/recorder/web/` — phone app (vanilla JS, no build step).
- `nav/recorder/web/dash/` — recordings dashboard at `/dash/` (vanilla JS + uPlot from jsdelivr, no build step).
- `nav/schemas/map.py` — the map format: `MapDraft` (what the LLM writes; its JSON Schema must stay valid for
  Anthropic structured outputs) and `PlaceMap` (stored: draft + evidence, positions, distances, build info).
- `nav/mapping/` — map pipeline: `prepare.py` (per-recording cache: frames, embeddings, transcripts), `frames.py`,
  `motion.py`, `models.py` (DINOv2, Whisper), `select.py` (candidates), `llm.py` (Claude Code job + prompt),
  `build.py` (orchestration + finish), `store.py`.
- `nav/recorder/web/viz/` — map viewer at `/viz/` (vanilla JS + SVG, no build step).
- `nav/cli.py` — `nav serve`, `nav rec list|show|finalize|push|pull`, `nav map setup|build|llm|refinish|list|show|push|pull`.
- `data/recordings/<rec_id>/` — recordings (gitignored). Shared copy for the team: `gs://tungsten-united-nav-recordings/recordings/<rec_id>/`
  (project `tungsten-united`, not the active gcloud project `kotie-app`), via `nav rec push|pull` (`gcloud storage rsync`, no deletes, skips `raw/`).
- `data/maps/<map_id>/` — maps (gitignored), shared under `gs://tungsten-united-nav-recordings/maps/<map_id>/` via
  `nav map push|pull`; `data/cache/<rec_id>/` — preprocessing cache, safe to delete.

## Run / test
```sh
uv sync && uv run pytest
systemctl --user restart nav-recorder      # live service, 127.0.0.1:8450 (deploy/nav-recorder.service)
journalctl --user -u nav-recorder -f
```
Public URL: https://helium-nav.oscarmartinez.dev (Cloudflare Tunnel "helium", route added with `tunnel-expose nav 8450`).
Cloudflare Access gates every `helium-*` host: browser login for people, the `helium-clients` service token for
workers (`~/.cloudflared/access-service-token.env`, headers `CF-Access-Client-Id/Secret`). The app has no auth of its own.

Browser end-to-end check: headless `/usr/bin/chromium` via `uvx --from playwright python` with
`--use-fake-device-for-media-stream --use-fake-ui-for-media-stream`, sensor permissions granted, synthetic
`devicemotion` / `deviceorientationabsolute` events dispatched from the page. Run it against a throwaway
`NAV_DATA_DIR` on another port, never the live service.

## Things learned the hard way
- Chrome's MP4 MediaRecorder only emits data at keyframes (in practice at stop); WebM honours the timeslice.
  Keep WebM first in `VIDEO_MIMES` and `videoKeyFrameIntervalDuration: 1000`.
- Cloudflare edge-caches `.js/.css` and the zone forces a 4 h browser TTL. `index.html` is served with
  `?v=<content hash>` on assets and `Cache-Control: no-cache`; keep it that way or phones get a stale app.js.
- Chrome now has `DeviceMotionEvent.requestPermission`; a denial must not abort recording (video + voice still useful).
- Chrome (Android) reports `rotationRate` alpha/beta/gamma around device x/y/z, not z/x/y as the spec says; the
  recorder stores gx/gy/gz = beta/gamma/alpha. `motion.py` picks the layout whose turns match the compass.
- Handheld walking can be very smooth (|a| std 0.1–0.4 m/s²): fixed step thresholds find nothing; use adaptive ones.
- `nav map refinish` reuses the saved Claude answer: use it to iterate on post-processing without paying again.
- `nvidia-smi` may report "Driver/library version mismatch" after a driver update without reboot; CUDA still works.

## Product decisions
- A recording is a **free walk through a place**, not a route. `place` (required, whitespace-normalised,
  case-insensitive grouping) is the map key: every walk of a place feeds that place's single map.
- The map pipeline must therefore merge many walks and detect revisits (embedding similarity + tags + voice),
  not assume a linear path.
- Map pipeline MVP (built): 1 fps sharpest frames → quality filter → keyframes per walk (novelty by DINOv2
  embedding + walked distance / turn from the IMU, anchors at tags and voice notes) → fold look-alikes across walks
  → a Claude Code job writes the graph from candidates.json + frames → code checks ids and adds evidence, positions
  and distances. Voice is transcribed locally (Whisper turbo; no OpenAI key here). Later: Anthropic API with
  structured outputs instead of the Claude Code job; code proposing node/edge spans itself.

## Conventions
- Work on branch `omm/first_draft`; push with `git push` (gh credential helper is configured for this repo).
- Data sharing: the server does not upload by itself. After new recordings are finalized, run `uv run nav rec push`
  (from the main checkout, so it sees the live `data/`) to keep the team bucket current; check with `--dry-run` first.
  Teammates with access: lpall98@gmail.com (objectAdmin). Grant others per README → "Shared dataset".
- Propose before large changes; the user reviews plans first.
