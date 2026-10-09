# nav-engine

Navigation engine for a blind-navigation app. Today: the **recorder** (phone web app + API) that captures free
walks through a place. Next: turn all walks of a place into one semantic topological map, then localization.

## Layout
- `nav/schemas/recording.py` — the recording format (schema_version 1). Source of truth for what is stored.
- `nav/recorder/app.py` — FastAPI app factory: `/api/v1` + the static phone app; `store.py` — filesystem storage,
  idempotent uploads, `finalize` (chunks → `video.mp4`, batches → parquet/jsonl).
- `nav/recorder/web/` — phone app (vanilla JS, no build step).
- `nav/cli.py` — `nav serve`, `nav rec list|show|finalize|push|pull`.
- `data/recordings/<rec_id>/` — recordings (gitignored). Shared copy for the team: `gs://tungsten-united-nav-recordings/recordings/<rec_id>/`
  (project `tungsten-united`, not the active gcloud project `kotie-app`), via `nav rec push|pull` (`gcloud storage rsync`, no deletes, skips `raw/`).

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
- `nvidia-smi` may report "Driver/library version mismatch" after a driver update without reboot; CUDA still works.

## Product decisions
- A recording is a **free walk through a place**, not a route. `place` (required, whitespace-normalised,
  case-insensitive grouping) is the map key: every walk of a place feeds that place's single map.
- The map pipeline must therefore merge many walks and detect revisits (embedding similarity + tags + voice),
  not assume a linear path.
- Planned map pipeline (agreed direction, not built): resample to ~1 fps picking the sharpest frame per second;
  transcribe voice clips (`whisper-1` for word timestamps); embed frames (DINOv2 default) *before* segmentation;
  code proposes node/edge spans from tags, voice and junction flags, and the LLM only names, merges and writes
  instructions; cache every stage and API call by content hash; interactive HTML viewer (Cytoscape.js).

## Conventions
- Work on branch `omm/first_draft`; push with `git push` (gh credential helper is configured for this repo).
- Propose before large changes; the user reviews plans first.
