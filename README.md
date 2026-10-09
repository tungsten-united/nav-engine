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
| GET | `/recordings/{id}/files[/{path}]` | list / download any file |

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
