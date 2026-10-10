# nav-engine

Navigation engine of [Orient](https://github.com/tungsten-united/project-description), a voice-first web app that
guides blind and visually impaired people indoors. Two sides:

- **Mapping**, for the team: the **recorder** captures free walks through a place (video, motion sensors, voice notes
  and tags), the **map pipeline** turns all walks of a place into one semantic topological map, and a person reviews
  it. **map-api** serves this, used through the [debugging frontend](frontend/README.md).
- **Live**: **nav-api** ([below](#nav-api-the-live-api)) says which node of the map the camera sees and gives the
  route to a destination. Its caller is Orient's orchestrator
  ([tungsten-united/orient-orchestrator](https://github.com/tungsten-united/orient-orchestrator)). Orient's phone app
  talks only to the orchestrator, never to nav-engine.

> Experimental. Not a substitute for a white cane or guide dog.

## Recorder

A phone page of the debugging frontend plus an HTTP API, both part of **map-api**, which runs on Cloud Run at
**https://map-api-613464313064.europe-southwest1.run.app** on the team bucket, gated by an API token (see [Deployment](#deployment)). helium serves the same
debugging pages at https://helium-nav.oscarmartinez.dev (behind Cloudflare Access); they call the Cloud Run services.
The live side, for Orient's orchestrator, is **nav-api** ([below](#nav-api-the-live-api)).

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
uv run nav serve                      # map-api on http://127.0.0.1:8450 (NAV_HOST / NAV_PORT / NAV_DATA_DIR)
uv run nav serve --app nav-api        # nav-api on http://127.0.0.1:8452 (NAV_APP picks the default)
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
**https://map-api-613464313064.europe-southwest1.run.app/dash/** (or helium-nav's `/dash/`) lists the recordings grouped by place.
Opening one (`/dash/#<rec_id>`) shows its details, the video, and its tags, voice notes, steps, acceleration,
rotation, heading and live speed on one timeline. Click the timeline, a chart or a note to jump the video there;
voice notes play along with the video. Drag on a chart to zoom, double-click to reset.

### map-api (`/api/v1`, OpenAPI at `/docs`)
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
| POST | `/maps/build`, `/maps/{map_id}/refinish` | build / refinish in the background → poll the `status_url` |
| GET / PUT | `/maps/{map_id}/review` | the human review; PUT republishes the map with it |

From a script (`deploy/cloudrun.sh token map-api` prints the token):
```sh
curl -H "Authorization: Bearer $MAP_API_TOKEN" https://map-api-613464313064.europe-southwest1.run.app/api/v1/recordings
```
helium-nav's own API (behind Cloudflare Access: send `CF-Access-Client-Id/Secret` from
`~/.cloudflared/access-service-token.env`) answers from helium's local data, not the team bucket.

### Known limits
- No barometer or OS step counter in browsers: floor changes come from video/voice, steps from accel peaks.
- Indoor compass readings are disturbed by steel and wiring; heading is stored raw for offline fusion with the gyro.
- Uploads waiting in the queue live in page memory: don't close or reload the tab until it says "Recording saved".
- map-api on Cloud Run writes recordings straight to the team bucket. A recorder run elsewhere (`nav serve`, e.g.
  helium-nav) writes to local disk, and sharing those is a manual `nav rec push`.
- Maps: positions come from dead reckoning, so they drift over long walks. Distances come from the floor seen in the
  video where possible; their scale rests on an assumed phone height until a distance measured on site is entered.

## Maps

Every walk of a place feeds one **semantic topological map** of that place (the phone app's **Place** dropdown picks
an existing place, so a new walk joins its map; walks may start anywhere and go in any direction: say where you
start): nodes are spots worth naming (entrance,
doors, junctions, rooms, areas, stairs, lifts, destinations, landmarks), and edges are walkable connections with
spoken, egocentric instructions both ways. The format is in [`nav/schemas/map.py`](nav/schemas/map.py).
On map-api a walk that finalizes rebuilds its place's map by itself (`NAV_AUTO_BUILD`, on by default; `0` turns it
off): the same map, improved, since the build starts from the published one (ids, the review and what still holds
carry over), and a failed build leaves the published map as it was. Walks that finalize during a build queue one more
build of their place. Each build's `status.json` says what triggered it (`auto: <rec_id> finalized`, or `api`).

```sh
uv run nav map setup                  # once: download + check the local models (GPU if available)
uv run nav map build "Itnig"          # all finished walks of the place -> data/maps/itnig/
uv run nav map build "Itnig" --no-llm # stop after selecting candidates (free; to tune the selection)
uv run nav map llm itnig              # rerun only the LLM with the latest build's candidates, as a new build
uv run nav map refinish itnig         # redo post-processing from the saved LLM answer (no new LLM call)
uv run nav map check itnig            # rerun the cheap quality check (TypeSafe Jev) on the published map
uv run nav map review itnig           # write an editable review file; --apply <file> verifies / fixes / hides
uv run nav map measure itnig "Main entrance" "Drinks area" 7 --sigma 0.3   # a distance measured on site: sets the scale
uv run nav map route itnig "Main entrance" Stage [--trust verified|observed|any]
uv run nav map pdr-replay itnig <rec_id> [--step-scale 0.85]  # replay a walk through the live dead-reckoning tracker
uv run nav map nav-replay itnig <rec_id> [--step-scale 0.85] [--k 3 --n 4]  # ... through the whole live loop
uv run nav map schema [--out docs/map.schema.json]            # JSON Schema of the published map, for nav-api's callers
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
   - **floor odometry** ([`odometry.py`](nav/mapping/odometry.py)), after the calibration below: floor features in
     the lower part of the frame tracked over 0.5 s and projected onto the floor (camera tilt from gravity, the
     calibrated field of view, phone height 1.35 m) give the metres actually walked; turning on the spot reads
     ~0 m. Where the floor can't be tracked, the steps fill in at a step length re-estimated from the video.
     Itnig: 32 m walked (steps × 0.7 m said 107 m: hand sway while talking counted as steps, and a slow walk while
     filming takes ~0.3 m steps), Main entrance → Drinks area 5.9 m against 7 m measured;
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
   Everything meant to be spoken (summary, descriptions, landmarks, instructions; Orient's orchestrator speaks a
   route hop's `instruction`) is written without visual detail, for someone who cannot see; colours, signs, screens and the like go into `visual_cues`, for sighted helpers and
   camera-based localization. The prompt is in [`nav/mapping/llm.py`](nav/mapping/llm.py). The schema is written to also be valid for the
   Anthropic structured-outputs API (closed objects, all fields required, no numeric/length constraints), which can
   replace the job later.
   When the place already has a map, it goes into the build as `previous.json` (each old node with the new
   candidates that show its frames) and the LLM is told to keep it where the evidence holds: same nodes, names and
   connections, adding what is new and explaining any merge, split or drop in `notes`.
   **More walks of a place** improve the same map rather than making a new one. Walks are aligned before the LLM
   (`map_x_m` / `map_y_m` on each view) through candidates folded across them; a walk in the opposite direction
   usually shares none (spots seen from the other side don't look alike), so the LLM joins its frames to the
   existing nodes from the walker's words, the reverse order of places on its timeline, distances and turns, and is
   told to keep every previous edge's direction (code turns an edge written the other way back).
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
   Edges use **every walk** that went directly between their nodes: `walked_forward` / `walked_back` counts, each
   direction `observed` once walked, the measured distance is the shortest pass, the path and headings come from
   the best one. A rebuild that drops a previous node says so in `warnings`.
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
   everything). **Routes use verified edges only, unless the caller asks for less**
   ([`route.py`](nav/navigation/route.py)). Orient's orchestrator asks for `observed` edges (its `NAV_TRUST`) until
   the demo route is verified, so for now it guides over edges nobody has checked.
   Distances measured on site (`distances` in the review, `nav map measure`) fit one scale factor, by least squares
   over all of them, to every distance, position, path and track of the published map; `frame.measured_distances`
   reports each one's residual. `nav map measure` edits a local map folder; for the team's maps, add the entry to
   `distances` with map-api's `PUT /api/v1/maps/<map_id>/review`, and nav-api serves the rescaled map. On a build whose distances came from steps it can only rescale uniformly (step
   errors aren't uniform), so rebuild first.

### Navigation code
[`nav/navigation/`](nav/navigation/) is the live side, small and causal. nav-api serves two of its pieces:
`route.plan_route` (shortest route over trusted edges, in the walked direction) and `localize.localize` (which node
live frames show). The others are reference code that nothing live runs; the app ports them.
`pdr.GraphTracker` (a causal step detector + progress along the route's bearings, the uncertainty growing with
distance, the probability of being at the next node, and the user's step length re-learned at each confirmed
node); `nav map pdr-replay` plays a mapping walk back through it.
`arrival.NodeConfirmer` decides that the user reached, or walked through, the next node of the route from
consecutive localizations (nav-api `/localize`, sent with the confirmer's `previous` / `expected`), the dead
reckoning and the compass: votes over the last n localizations (T recognised through its approach / at references
= arrive; through its outgoing references toward the next node, or the next node recognised = passed), each counted
only if the dead reckoning agreed when the frame was taken (approach photos: near T; at photos and pass votes: half
the hop walked), so a place seen from afar is never "reached"; the heading must agree with the hop's arrival
direction. It also reports `uncertain` (walked past without recognising it) and `off_route` (another node
recognised); nothing is ever skipped, and a manual confirmation is always accepted.
[`frontend/nav/arrival.js`](frontend/nav/arrival.js) is its 1:1 JS port (with the step detector and tracker; a test
checks they agree), used by the `/nav/` debugger's "auto-confirm". `nav map nav-replay` plays a walk back through the
whole loop (steps → tracker, video at 2 fps → localization → confirmer; frames near the walk's own references left
out). Itnig, its own mapping walk: 6 of 9 nodes confirmed by the camera, 0 false, 0 off-route alarms; the misses
are two spots 3 m apart that look alike (Corridor end / Kitchen) and a kitchen scoring under 0.6.
Design: [`docs/main_idea.md`](docs/main_idea.md), [`docs/navigation_stack.md`](docs/navigation_stack.md)
(by LPA); the map's JSON Schema for nav-api's callers is [`docs/map.schema.json`](docs/map.schema.json) (also
`GET /api/v1/schema/map`).

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
**https://map-api-613464313064.europe-southwest1.run.app/viz/** shows each map as a navigation graph: nodes at their dead-reckoned
positions and edges along the walked path (with the whole walk as a faint track), the route in recording order
with turns, hover previews with the frame, and details pinned on click (description, landmarks, hazards, both
instructions, frames, the walks and times that show it). **Directions** picks two nodes and chains the
instructions along the shortest route (any edge, or only observed or verified ones). Nodes and edges show their
status with **Verify** / **Hide** buttons (they save the review), reference images by role with their heading,
and relations.

## nav-api: the live API

The live API, for Orient's orchestrator: read-only on the bucket's published maps, with its own token. Code:
[`nav/nav_api/app.py`](nav/nav_api/app.py), formats in [`nav/schemas/navigation.py`](nav/schemas/navigation.py).
It runs on Cloud Run at **https://nav-api-613464313064.europe-southwest1.run.app** (before the split that URL served
everything; its page paths now redirect to map-api, and recordings and builds answer 404 with a pointer there).

**Who calls it.** Orient's orchestrator, server to server; the phone sends its frames to the orchestrator, never here.
Since 2026-10-10 (orient-orchestrator `main`), for every frame it evaluates, it calls `localize` with the session's
last 4 frames and the last confirmed node as `previous`, then `route` from that node to the destination with
`trust: observed`, and speaks the first hop's `instruction`. The contract is
[project-description `docs/contracts.md` §2](https://github.com/tungsten-united/project-description/blob/main/docs/contracts.md#2-orchestrator--navigation-engine):
change it there first. Its staging release uses a fake navigation engine unless `fake_nav` is unticked, and its
destinations must be node ids of the map (`n1` …). The debugging pages below also call
nav-api, to test it by hand.

| method | path | |
|---|---|---|
| GET | `/health[?deep=1]` | `deep`: also the map count and the inference host |
| GET | `/maps`, `/maps/{map_id}`, `/schema/map` | as on map-api: the published map (latest build + review) |
| GET | `/maps/{map_id}/files/{path}` | only the published build's `refs/`, `frames/` and its embeddings |
| POST | `/maps/{map_id}/route` | `{start, goal, trust}` → hops (verified edges by default; `observed` / `any` for testing) |
| POST | `/maps/{map_id}/localize` | 1–4 JPEG frames (multipart `images`), `previous?`, `expected?`, `heading_deg?`, `top_k?` → `Localization` |

**Localization** ([`localize.py`](nav/navigation/localize.py)): the frames are embedded with the map's own model
(MegaLoc) on the inference host, ahead of any queued map-build calls, and compared with the map's reference images. A
node scores as its best reference (a burst scores each reference by the mean over its frames); the margin is the best
node's score minus the second's. Status: `lost` under 0.45, `confirmed` at ≥ 0.60 with a margin ≥ 0.05 and, when
`previous` is given, the node being it or one edge away; `uncertain` otherwise. The thresholds are placeholders until
tuned on venue images (`NAV_LOC_CONFIRM_SCORE`, `NAV_LOC_CONFIRM_MARGIN`, `NAV_LOC_LOST_SCORE`). It is stateless: the
caller (the orchestrator) keeps the last confirmed node, combines consecutive results and confirms arrival itself
(docs/navigation_stack.md 4.2–4.3). Each result lists
the matched references (role, heading, and the difference with the live compass heading when given), and
`took_ms`. On Itnig, frames from the walk the map was built from land on the right node at ~27 ms per frame on the
GPU (plus the network). Those frames are in the map, so that shows the plumbing and the speed, not accuracy: no
held-out walk has been tested and the thresholds are untuned. The first frame after nav-infer restarts can get a 503 while MegaLoc loads (live calls give up after 8 s,
`NAV_LIVE_INFER_DEADLINE_S`).

**Debugging pages**, for testing nav-api by hand; they are not part of Orient (served by map-api and helium-nav,
calling nav-api): **`/loc/`** localizes frames from the
camera, a recording's video (with its compass heading) or an image, and shows ranked nodes with their reference
images, timings and a history; walks the map was built from match themselves, so held-out walks are the honest
test. **`/nav/`** walks a route roughly the way a caller would: it plans it, speaks its instructions, checks live
frames against the next checkpoint, and takes arrival as confirmed by hand.

## Deployment

```
recorder, /dash, /viz (debugging) ──HTTPS + map-api token──▶ map-api (Cloud Run)  read-write ┐ /data = the team bucket
Orient's orchestrator, /loc, /nav ──HTTPS + nav-api token──▶ nav-api (Cloud Run)  read-only  ┘ (GCS FUSE): recordings/ maps/
              (the debugging pages are served by map-api, and by helium-nav)          │
                       both, with the Cloudflare Access service token                  ▼
                  helium-nav-infer.oscarmartinez.dev ─tunnel─▶ nav-infer on helium (GPU): DINOv2, MegaLoc,
                                                               ALIKED+LightGlue, Whisper, the Claude Code map job
```

Both Cloud Run services (tungsten-united, europe-southwest1) run the root `Dockerfile` (one image, no torch; `NAV_APP`
picks the app), deployed with `deploy/cloudrun.sh [deploy] [map-api|nav-api|all]`, which builds the image here,
pushes it and deploys. Rerun it after every change.

**map-api** (https://map-api-613464313064.europe-southwest1.run.app): recordings, finalize (ffmpeg), map builds, the
Jev quality check (TypeSafe API), review, map files, and the debugging frontend (`frontend/`) for the team.
- One instance, always on, so in-process locks and background work (finalize, builds) hold. Runs as `nav-worker`
  (bucket read-write). Settings: [`deploy/map-api.env.yaml`](deploy/map-api.env.yaml).
- Deleting recordings is off: the bucket is the team's shared data, and it keeps deleted objects for 7 days anyway.
- Map builds: `POST /api/v1/maps/build {"place": "Itnig", "llm": false}` → poll the `status_url` it returns.
  `llm: false` stops after the candidates (no LLM cost). One build at a time. A walk that finalizes starts one by
  itself (or queues one, while another runs): see [Maps](#maps).

**nav-api** (https://nav-api-613464313064.europe-southwest1.run.app): the live API, for Orient's orchestrator ([above](#nav-api-the-live-api)).
Runs as `nav-api` (bucket viewer), the bucket mounted read-only; 1 to 3 instances, at least one always on.
Settings: [`deploy/nav-api.env.yaml`](deploy/nav-api.env.yaml) (CORS for map-api's and helium-nav's pages).

Both:
- `/api/*` needs the service's token, as `Authorization: Bearer <token>` or `?token=` (for `<video>`/`<img>` URLs).
  `deploy/cloudrun.sh token map-api|nav-api` prints it. nav-api's is the token from before the split; map-api's
  started as a copy of it, so they can be rotated apart (write a new value to its file, then `secrets` and `deploy`).
- Secrets (API tokens, Cloudflare service token, `TYPESAFE_API_KEY`) live in Secret Manager, each readable only by
  the services that use it. `deploy/cloudrun.sh secrets` reloads them from this machine.
- **CI deploy** ([`.github/workflows/deploy.yml`](.github/workflows/deploy.yml)): push a tag `v*` on a commit of
  `main` (`git tag v0.3.0 origin/main && git push origin v0.3.0`), or Actions → Deploy → Run workflow on `main`.
  It runs the tests, then `deploy/cloudrun.sh deploy all` (map-api first), then checks both `/api/v1/health`. Tags
  on commits outside `main` stop before touching Google Cloud. Auth is keyless: GitHub OIDC → the workload identity
  pool `github` (shared with orient-orchestrator) → `nav-deployer` (Cloud Run admin, Artifact Registry writer, acts
  as `nav-worker` and `nav-api`). One-time setup by a project owner: `deploy/cloudrun.sh setup` and `ci-setup`
  (only add). It doesn't touch helium: after a change to the map job, the prompt or the inference host, also update
  and restart `nav-infer` there.
- Rollback: `gcloud run services update-traffic <service> --to-revisions <revision>=100 --region europe-southwest1
  --project tungsten-united` (`gcloud run revisions list --service <service> ...` lists them).

**nav-infer** (helium): `nav infer serve` on 127.0.0.1:8451 (`deploy/nav-infer.service`), published with
`tunnel-expose nav-infer 8451`. It serves the local models and the Claude Code map job, and nothing else (no data).
It shares one GPU with the desktop, so it protects itself:
- one model call at a time, at most 4 waiting, then `503` + `Retry-After`; live frames from nav-api go ahead of
  waiting calls (at most 2 waiting of their own);
- at most 32 images or 64 pairs per request, so each call stays well under Cloudflare's 100 s;
- one map job at a time, at most 2 waiting; jobs are submitted, then polled.

The workers' client retries with backoff and fails clearly when helium is off. Recording doesn't need helium; map
builds and localization do. Without `NAV_INFER_URL`, everything runs locally as before (`nav map build` on helium).

**helium-nav** (`deploy/nav-recorder.service`: `nav serve`, i.e. map-api, on 127.0.0.1:8450 behind Cloudflare
Access) keeps serving the debugging pages; they call the Cloud Run services (`frontend/config.js`). After pulling new
code there, restart it to get the new pages.

**Coexistence:** recordings get unique ids and pushes never delete, so helium and the cloud can both add. Maps are
built by the cloud now: on helium use `nav map pull`, not `push`. `nav rec pull` skips `raw/` unless `--raw` is given.
