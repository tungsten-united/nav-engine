# nav-engine

Navigation engine for blind users. First piece: the **recorder**, which captures a narrated walkthrough of an
indoor route (video, motion sensors, voice notes and tags) so it can be turned into a semantic topological map.

> Experimental. Not a substitute for a white cane or guide dog.

## Recorder

A phone web app plus an HTTP API, served from this machine and published through the Cloudflare Tunnel at
**https://helium-nav.oscarmartinez.dev**. It is protected by Cloudflare Access: people log in in the browser, and
other workers send the `helium-clients` service token.

While you walk, the phone records:

| what | how | stored as |
|---|---|---|
| rear camera | MediaRecorder (WebM/H.264 on Android), 1080p, 30 fps, 5 Mbps, keyframe every 1 s, uploaded in 2 s chunks | `video.mp4` |
| motion | `devicemotion`, native rate (~60 Hz): accel with/without gravity, rotation rate | `imu.parquet` |
| orientation / heading | `deviceorientationabsolute` (Android) or `webkitCompassHeading` (iOS) | `orientation.parquet` |
| voice | **Talk**: hold to talk, or tap to start and tap again to stop; one clip per press | `voice/NNN.webm` |
| tags | Door / Junction / Stairs / Lift / Destination buttons | `mark` events |
| live steps / speed | step peaks in \|accel\| × step length (display only; recomputed offline) | `step` events |

Every timestamp is `t_ms` since the start, on the phone's monotonic clock; `manifest.json → t0_epoch_ms`
anchors it to UTC, and the `video_start` event gives video time 0. The full format is in
[`nav/schemas/recording.py`](nav/schemas/recording.py).

```
data/recordings/<rec_id>/        rec_id = 20261009T194012Z_a3f9
  manifest.json                  Manifest: device, camera, status, counts, voice clips, gaps
  video.mp4
  voice/000.webm …
  imu.parquet  orientation.parquet  events.jsonl
  raw/                           uploads as received (video chunks are removed once video.mp4 is built)
```

Uploads are idempotent (one file per sequence number) and retried with backoff, so flaky Wi-Fi only delays them.
When the phone stops, the server assembles the files in the background, and the recording becomes `complete`, or
`incomplete` with its `gaps` listed.

### Run
```sh
uv sync
uv run nav serve                      # http://127.0.0.1:8450 (NAV_HOST / NAV_PORT / NAV_DATA_DIR)
tunnel-expose nav 8450                # once: https://helium-nav.oscarmartinez.dev
```
On helium it runs as a user service ([`deploy/nav-recorder.service`](deploy/nav-recorder.service), copied to
`~/.config/systemd/user/`): `systemctl --user {status,restart} nav-recorder`, logs with
`journalctl --user -u nav-recorder`.

```sh
uv run nav rec list                   # recordings
uv run nav rec show <rec_id>          # manifest
uv run nav rec finalize <rec_id>      # rebuild outputs from raw uploads
uv run pytest
```

### Shared dataset (Google Cloud Storage)
Finished recordings are shared with the team in the private bucket
**`gs://tungsten-united-nav-recordings/recordings/<rec_id>/`** (project `tungsten-united`, region `europe-southwest1`),
with the same layout as `data/recordings/`. `raw/` is not uploaded unless you ask for it.
```sh
gcloud auth login                     # once, with the Google account that was granted access
uv run nav rec push [<rec_id> …]      # upload (default: every finished recording); --raw, --dry-run
uv run nav rec pull [<rec_id> …]      # download into data/recordings (default: everything)
gcloud storage ls gs://tungsten-united-nav-recordings/recordings/   # or browse without the repo
```
Both only add or update files and never delete. `NAV_GCS_URL` overrides the location. To give a teammate
read/write access:
`gcloud storage buckets add-iam-policy-binding gs://tungsten-united-nav-recordings --member user:<email> --role roles/storage.objectAdmin`
(use `roles/storage.objectViewer` for read-only).

### Dashboard
**https://helium-nav.oscarmartinez.dev/dash/** (same Cloudflare Access login) lists the recordings grouped by place.
Opening one (`/dash/#<rec_id>`) shows its details, the video, and its tags, voice notes, steps, acceleration,
rotation, heading and live speed on one timeline. Click the timeline, a chart or a note to jump the video there;
voice notes play along with the video. Drag on a chart to zoom, double-click to reset.

### API (`/api/v1`, OpenAPI at `/docs`)
| method | path | |
|---|---|---|
| GET | `/health` | |
| POST | `/recordings` | create (`RecordingCreate`) → `Manifest` |
| GET | `/recordings` | list → `RecordingSummary[]` |
| GET / PATCH / DELETE | `/recordings/{id}` | manifest / edit place, notes, step length / delete |
| PUT | `/recordings/{id}/video/{seq}` | raw video chunk |
| POST | `/recordings/{id}/samples` | `SampleBatch` (imu, orientation, events) |
| PUT | `/recordings/{id}/voice/{clip}?t_start_ms&t_end_ms` | audio clip (Content-Type = its mime) |
| POST | `/recordings/{id}/finish` | `FinishRequest` (what the phone sent) → finalizes in background |
| POST | `/recordings/{id}/finalize?wait=true` | rebuild outputs |
| GET | `/recordings/{id}/files[/{path}]` | list / download any file (Range requests work, for video seeking) |
| GET | `/recordings/{id}/series/{imu,orientation}?max_points` | samples as JSON columns, for plotting |
| GET | `/maps` | built maps → `MapSummary[]` |
| GET | `/maps/{map_id}` | `PlaceMap` (the latest build) |
| GET | `/maps/{map_id}/builds` | build ids, newest first |
| GET | `/maps/{map_id}/files/{path}` | frames, `candidates.json`, job files of any build |

From another worker:
```sh
source ~/.cloudflared/access-service-token.env
curl -H "CF-Access-Client-Id: $CF_ACCESS_CLIENT_ID" -H "CF-Access-Client-Secret: $CF_ACCESS_CLIENT_SECRET" \
  https://helium-nav.oscarmartinez.dev/api/v1/recordings
```

### Known limits
- No barometer or OS step counter in browsers: floor changes come from video/voice, steps from accel peaks.
- Indoor compass readings are disturbed by steel and wiring; heading is stored raw for offline fusion with the gyro.
- Uploads waiting in the queue live in page memory: don't close or reload the tab until it says "Recording saved".
- The recorder writes to local disk; sharing through the bucket is a manual `nav rec push`.
- Maps: positions come from dead reckoning, so they drift over long walks; distances undercount very slow steps.

## Maps

Every walk of a place feeds one **semantic topological map** of that place: nodes are spots worth naming (entrance,
doors, junctions, rooms, areas, stairs, lifts, destinations, landmarks), and edges are walkable connections with
spoken, egocentric instructions both ways. The format is in [`nav/schemas/map.py`](nav/schemas/map.py).

```sh
uv run nav map setup                  # once: download + check the local models (GPU if available)
uv run nav map build "Itnig"          # all finished walks of the place -> data/maps/itnig/
uv run nav map build "Itnig" --no-llm # stop after selecting candidates (free; to tune the selection)
uv run nav map llm itnig              # rerun only the LLM with the latest build's candidates, as a new build
uv run nav map refinish itnig         # redo post-processing from the saved LLM answer (no new LLM call)
uv run nav map list | show <map_id>
uv run nav map push [<map_id> …]      # upload to gs://tungsten-united-nav-recordings/maps/<map_id>/ (--dry-run)
uv run nav map pull [<map_id> …]
```

### Pipeline
1. **Preprocess** each walk (cached per recording in `data/cache/<rec_id>/`, keyed by parameters and model):
   - frames at **1 fps**, the sharpest of each second (variance of the Laplacian), 768 px;
   - **motion** from the IMU: steps (adaptive peaks of vertical acceleration, in bouts), walked distance
     (steps × step length), heading (gyro yaw about gravity, anchored to the compass), camera pitch, and a
     dead-reckoned path;
   - voice notes transcribed locally with **Whisper** (`openai/whisper-large-v3-turbo`);
   - frame **embeddings** with **DINOv2** (`facebook/dinov2-base`, whole frame without cropping, [CLS] + mean patch,
     1536-d, cosine similarity).
2. **Select candidates**: drop blurry, dark or floor/ceiling frames; per walk keep the start and end, frames at tags
   and voice notes, frames that look new once the walker moved ≥ 2 m or turned ≥ 60°, and one every 10 m; fold
   near-identical frames (cosine ≥ 0.92) across walks and revisits into one candidate with several views, and flag
   look-alikes (≥ 0.80) as possibly the same place; at most 60. Thresholds: `SelectParams` in
   [`nav/mapping/select.py`](nav/mapping/select.py).
3. **LLM**: a headless **Claude Code** job (Opus, medium effort) runs in the build folder with read-only tools (`Read`, `Glob`), opens
   `candidates.json` and every frame, and must answer in the `MapDraft` JSON Schema (`claude -p --json-schema`).
   Everything the app will say (summary, descriptions, landmarks, instructions) is written without visual detail,
   for someone who cannot see; colours, signs, screens and the like go into `visual_cues`, for sighted helpers and
   camera-based localization. The prompt is in [`nav/mapping/llm.py`](nav/mapping/llm.py). The schema is written to also be valid for the
   Anthropic structured-outputs API (closed objects, all fields required, no numeric/length constraints), which can
   replace the job later.
4. **Finish**: check node, edge and candidate ids and flag visual words left in spoken text (both go into
   `warnings`), attach evidence (recording, time,
   frame) to every node, place nodes at their first pass by dead reckoning (walks aligned through shared nodes),
   bend each walk so every later pass of a node lands on it (loop closure: the drift is spread over the stretch in
   between; passes far off are flagged in `warnings`), measure each edge's walked distance and keep its walked path
   for drawing.

Defaults can be changed with `NAV_EMBED_MODEL`, `NAV_TRANSCRIBE_MODEL`, `NAV_MAP_LLM_MODEL` (default `opus`), `NAV_MAP_LLM_EFFORT` (default `medium`),
`NAV_MAP_LANGUAGE` (default `en`), `NAV_MAP_LLM_BUDGET_USD` (default 5) and `NAV_GCS_MAPS_URL`. A build of a
2-minute walk takes about 15 s of preprocessing on the GPU plus 4 minutes and about $1 of Claude Code.

```
data/maps/<map_id>/              map_id = slug of the place, e.g. "itnig"
  map.json                       latest build's PlaceMap
  builds/<build_id>/             immutable, one per build
    map.json  candidates.json  tracks.json  frames/c01.jpg …
    job/  prompt.md  schema.json  command.txt  claude.json  claude.log
```

### Viewer
**https://helium-nav.oscarmartinez.dev/viz/** shows each map as a navigation graph: nodes at their dead-reckoned
positions and edges along the walked path (with the whole walk as a faint track), the route in recording order
with turns, hover previews with the frame, and details pinned on click (description, landmarks, hazards, both
instructions, frames, the walks and times that show it). **Directions** picks two nodes and chains the
instructions along the shortest walked route.
