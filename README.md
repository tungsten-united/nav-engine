# nav-engine

Navigation engine for blind users. First piece: the **recorder**, which captures a narrated walkthrough of an
indoor route (video, motion sensors, voice notes and tags) so it can be turned into a semantic topological map.

> Experimental. Not a substitute for a white cane or guide dog.

## Recorder

A phone web app plus an HTTP API. It runs on Cloud Run at
**https://nav-api-613464313064.europe-southwest1.run.app**, on the team bucket, gated by an API token (see
[Deployment](#deployment)). The original copy on helium (https://helium-nav.oscarmartinez.dev, behind Cloudflare
Access) still runs until the cutover is complete.

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
uv run nav map check itnig            # rerun the cheap quality check (TypeSafe Jev) on the published map
uv run nav map review itnig           # write an editable review file; --apply <file> verifies / fixes / hides
uv run nav map route itnig "Main entrance" Stage [--trust verified|observed|any]
uv run nav map pdr-replay itnig <rec_id> [--step-scale 0.85]  # replay a walk through the live dead-reckoning tracker
uv run nav map schema [--out docs/map.schema.json]            # JSON Schema of the published map, for the app
uv run nav rec import walk.mp4 --place "Itnig"                # a recording from a video alone (no sensors)
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
   - **calibration** ([`calibrate.py`](nav/mapping/calibrate.py)): the video's own yaw (ORB shift between frames
     0.1 s apart) cross-correlated with the sensor heading gives the sensor-to-video offset (Itnig: +0.6 s, the
     encoder starting late) and the camera's horizontal field of view (38°); applied when the correlation is clear.
     A recording **without sensors** (`nav rec import`) takes its heading from that yaw and guesses distances from
     time (~1 m/s); the LLM is told, and its distances are not counted as measured;
   - frame **embeddings** with **DINOv2** (`facebook/dinov2-base`, whole frame without cropping, [CLS] + mean patch,
     1536-d, cosine similarity), for "does this look new";
   - **place embeddings** with [**MegaLoc**](https://github.com/gmberton/MegaLoc) (DINOv2 + SALAD head trained for
     visual place recognition, 8448-d, torch.hub at a pinned commit), for "is this the same place".
2. **Select candidates**: drop blurry, dark or floor/ceiling frames; per walk keep the start and end, frames at tags
   and voice notes, frames that look new once the walker moved ≥ 2 m or turned ≥ 60°, and one every 10 m; drop
   near-duplicates of a recent frame (DINOv2 cosine ≥ 0.92). Then **same place** ([`verify.py`](nav/mapping/verify.py)):
   MegaLoc retrieves each keyframe's look-alikes (cosine ≥ 0.45, top 10) and **ALIKED + LightGlue** checks each
   pair (matches fitting one fundamental matrix). Pairs with ≥ 300 inliers (the same spot, not just the same room)
   across walks or revisits fold into one candidate with several views; the other look-alikes are flagged as
   possibly the same place; at most 60. Every checked pair goes to `pairs.json` in the build, for tuning.
   Thresholds: `SelectParams` in [`nav/mapping/select.py`](nav/mapping/select.py).
3. **LLM**: a headless **Claude Code** job (Opus, medium effort) runs in the build folder with read-only tools (`Read`, `Glob`), opens
   `candidates.json` and every frame, and must answer in the `MapDraft` JSON Schema (`claude -p --json-schema`).
   Each edge comes as `steps` (an action enum: straight, bear/turn left/right, turn around, go through, step/stairs
   up/down, lift; with distance, what to keep beside you and the cue that ends the step) plus their spoken form, and
   says whether it was `walked` or `inferred`. Distances are code's: measured, or the steps' sum if not walked.
   Everything the app will say (summary, descriptions, landmarks, instructions) is written without visual detail,
   for someone who cannot see; colours, signs, screens and the like go into `visual_cues`, for sighted helpers and
   camera-based localization. The prompt is in [`nav/mapping/llm.py`](nav/mapping/llm.py). The schema is written to also be valid for the
   Anthropic structured-outputs API (closed objects, all fields required, no numeric/length constraints), which can
   replace the job later.
   When the place already has a map, it goes into the build as `previous.json` (each old node with the new
   candidates that show its frames) and the LLM is told to keep it where the evidence holds: same nodes, names and
   connections, adding what is new and explaining any merge, split or drop in `notes`.
4. **Finish**: **stable ids** ([`continuity.py`](nav/mapping/continuity.py)): code, not the LLM, assigns them. A node
   that shares the most evidence frames with a previous node (same recording, within 3 s) keeps its id, new nodes
   get the next unused number, edges keep theirs when they join the same two nodes, and ids that disappear go to
   `retired` (never reused). Then check node, edge and candidate ids and flag visual words left in spoken text (both go into
   `warnings`), attach evidence (recording, time,
   frame) to every node, place nodes at their first pass by dead reckoning (walks aligned through shared nodes),
   bend each walk so every later pass of a node lands on it (loop closure: the drift is spread over the stretch in
   between; passes far off are flagged in `warnings`), measure each edge's walked distance and keep its walked path
   for drawing. Edges no walk went along directly (it passed other nodes in between) are marked inferred. Code
   derives each edge's reverse steps and checks its first step against the turn the walk made leaving the source
   (heading at the end of the source pass vs the next 6 s of walking).
   For the navigation side ([`docs/navigation_stack.md`](docs/navigation_stack.md)) it also stores: each direction's
   **status** (`observed`: a walk went that way; `inferred`: deduced, e.g. the way back) and the walking heading
   when leaving the source and reaching the target (map frame, compass-referenced: `frame`); **relations** (the
   LLM says where other places are in a node's photos, `x` across the photo; code makes that a bearing with the
   calibrated field of view and the photo's heading); and `review_flags`, what a reviewer should check.
5. **Reference images** ([`references.py`](nav/mapping/references.py)): per node, up to 3 sharp frames each while
   approaching, at, and leaving along each edge, from every walk that passed (the 1 fps cache), with their camera
   heading, plus their **MegaLoc** place embeddings in `embeddings.npy`, for live visual place recognition.
6. **Quality check** (cheap, optional): TypeSafe's **Jev** asks a few narrow questions about every node and edge, in
   parallel: does the spoken text point to something only visible, can a blind person tell they arrived / follow
   the instruction, and does the spoken instruction turn the same way as its first step. Benchmark and the
   justification for it: [`bench/quality/REPORT.md`](bench/quality/REPORT.md). Raw answers are stored per item
   (`quality`), flagged items go into `warnings` and are marked in the viewer. A map costs ~13k input tokens
   (well under a cent) and a second. Needs `TYPESAFE_API_KEY` (in `.env`; also a repo secret); otherwise skipped.
   Questions and thresholds: [`nav/mapping/quality.py`](nav/mapping/quality.py).
7. **Review** ([`review.py`](nav/mapping/review.py)): a person walks the map and verifies what holds (per edge
   direction), renames, fixes instructions or hides what is wrong, with `nav map review` or the viewer's buttons
   (`PUT /api/v1/maps/<map_id>/review`). The review is `review.json` next to the builds, keyed by stable ids, so it
   survives rebuilds; the published `map.json` is the latest build with it applied (the build's own copy keeps
   everything). **Real users are routed over verified edges only** ([`route.py`](nav/navigation/route.py)).

### Navigation reference code
[`nav/navigation/`](nav/navigation/) has the pieces of the live side in Python, small and causal so they port to
the TypeScript app: `route.plan_route` (shortest route over trusted edges, in the walked direction) and
`pdr.GraphTracker` (a causal step detector + progress along the route's bearings, the uncertainty growing with
distance, the probability of being at the next node, and the user's step length re-learned at each confirmed
node). `nav map pdr-replay` plays a mapping walk back through it. Design: [`docs/main_idea.md`](docs/main_idea.md),
[`docs/navigation_stack.md`](docs/navigation_stack.md) (by LPA); the map's JSON Schema for the app is
[`docs/map.schema.json`](docs/map.schema.json) (also `GET /api/v1/schema/map`).

Defaults can be changed with `NAV_EMBED_MODEL`, `NAV_PLACE_MODEL`, `NAV_TRANSCRIBE_MODEL`, `NAV_MAP_LLM_MODEL` (default `opus`), `NAV_MAP_LLM_EFFORT` (default `medium`),
`NAV_MAP_LANGUAGE` (default `en`), `NAV_QUALITY_MODEL` (default `jev-latest`), `NAV_MAP_LLM_BUDGET_USD` (default 5) and `NAV_GCS_MAPS_URL`. A build of a
2-minute walk takes about 15 s of preprocessing on the GPU plus 4 minutes and about $1 of Claude Code.

```
data/maps/<map_id>/              map_id = slug of the place, e.g. "itnig"
  map.json                       latest build's PlaceMap, with review.json applied
  review.json                    human review (approvals, fixes, hidden items), keyed by stable ids
  builds/<build_id>/             one per build
    map.json  candidates.json  tracks.json  frames/c01.jpg …
    refs/r001.jpg …  embeddings.npy   reference images and their place embeddings
    job/  prompt.md  schema.json  command.txt  claude.json  claude.log
```

### Viewer
**https://helium-nav.oscarmartinez.dev/viz/** shows each map as a navigation graph: nodes at their dead-reckoned
positions and edges along the walked path (with the whole walk as a faint track), the route in recording order
with turns, hover previews with the frame, and details pinned on click (description, landmarks, hazards, both
instructions, frames, the walks and times that show it). **Directions** picks two nodes and chains the
instructions along the shortest route (any edge, or only observed or verified ones). Nodes and edges show their
status with **Verify** / **Hide** buttons (they save the review), reference images by role with their heading,
and relations.

## Deployment

```
phone / dash / viz ──HTTPS + API token──▶ nav-api (Cloud Run, tungsten-united, europe-southwest1)
                                            │  /data = the team bucket (GCS FUSE): recordings/ maps/ cache/
                                            │  finalize (ffmpeg), map builds, Jev quality check (TypeSafe API)
                                            ▼  Cloudflare Access service token
                  helium-nav-infer.oscarmartinez.dev ─tunnel─▶ nav-infer on helium (GPU): DINOv2, MegaLoc,
                                                               ALIKED+LightGlue, Whisper, the Claude Code map job
```

**nav-api** (Cloud Run): the root `Dockerfile` (no torch), deployed with `deploy/cloudrun.sh`, which builds the
image here, pushes it and deploys it. Rerun it after every change. It serves `/api/v1` and the debugging frontend
(`frontend/`).
- One instance, always on, so in-process locks and background work (finalize, builds) hold.
- `/api/*` needs the token, as `Authorization: Bearer <token>` or `?token=` (for `<video>`/`<img>` URLs).
  `deploy/cloudrun.sh token` prints it.
- Deleting recordings is off: the bucket is the team's shared data, and it keeps deleted objects for 7 days anyway.
- Secrets (API token, Cloudflare service token, `TYPESAFE_API_KEY`) live in Secret Manager.
  `deploy/cloudrun.sh secrets` reloads them from this machine.
- Map builds: `POST /api/v1/maps/build {"place": "Itnig", "llm": false}` → poll the `status_url` it returns.
  `llm: false` stops after the candidates (no LLM cost). One build at a time.

**nav-infer** (helium): `nav infer serve` on 127.0.0.1:8451 (`deploy/nav-infer.service`), published with
`tunnel-expose nav-infer 8451`. It serves the local models and the Claude Code map job, and nothing else (no data).
It shares one GPU with the desktop, so it protects itself:
- one model call at a time, at most 4 waiting, then `503` + `Retry-After`;
- at most 32 images or 64 pairs per request, so each call stays well under Cloudflare's 100 s;
- one map job at a time, at most 2 waiting; jobs are submitted, then polled.

The worker's client retries with backoff and fails clearly when helium is off. Recording doesn't need helium; map
builds do. Without `NAV_INFER_URL`, everything runs locally as before (`nav map build` on helium).

**Coexistence:** recordings get unique ids and pushes never delete, so helium and the cloud can both add. Maps are
built by the cloud now: on helium use `nav map pull`, not `push`. `nav rec pull` skips `raw/` unless `--raw` is given.
