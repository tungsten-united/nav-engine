# Debugging frontend

The team's pages for mapping and debugging: an index (`index.html`, `index.js`), the recordings dashboard
(`dash/`), the map viewer (`viz/`), and the debuggers of the live API: localization (`loc/`) and navigation (`nav/`).
Vanilla JS, no build step. This is not Orient's phone app
([project-description](https://github.com/tungsten-united/project-description) `apps/web`): for guidance that one talks
only to the orchestrator, which calls nav-api.

Recording walks moved to Orient's app, at https://orient.harshdeepsingh.dev/map (`apps/web/src/mapping/`), in Orient's
style. It uploads to map-api exactly as the old recorder did (`nav/schemas/recording.py`). The index here links to it
with this browser's map-api token (`?token=`), so a teammate signed in here isn't asked for it there.

## Talking to the APIs
Two services, two base URLs (both set in `config.js`; OpenAPI at `/docs` on each):

| | base URL | what | used by |
|---|---|---|---|
| **map-api** | `https://map-api-613464313064.europe-southwest1.run.app` | recordings, map builds, review, map files | Orient's `/map` recorder, `dash/`, `viz/` (`NAV`) |
| **nav-api** | `https://nav-api-613464313064.europe-southwest1.run.app` | the live API: published maps, routes, localization | Orient's orchestrator (server-side); `loc/`, `nav/` (`NAV.nav`) |

- Formats: `nav/schemas/recording.py`, `nav/schemas/map.py`, `nav/schemas/navigation.py`.
- Auth: each service has its own token. Send `Authorization: Bearer <token>` on every `/api/*` call except
  `/api/v1/health`. Elements that can't send headers (`<video>`, `<img>`, `<audio>`, download links) put it in the URL
  as `?token=<token>`. Ask the backend owner for the tokens. The orchestrator only needs nav-api's.
- CORS: both allow `http://localhost:*`, `http://127.0.0.1:*` and helium-nav; map-api also allows Orient's app (its
  `/map` recorder), nav-api also allows map-api's pages. Ask for your origin to be added (`NAV_CORS_ORIGINS` in
  `deploy/*.env.yaml`). CORS is for browsers only: the orchestrator calls from its server.
- Large files (videos) are served in ranges of at most 8 MiB. Players handle this; for a download, a plain GET
  streams the whole file.
- Localizing a frame: `POST {nav-api}/api/v1/maps/{map_id}/localize` with multipart `images` (1–4 JPEGs, ~768 px),
  optional `previous` (last confirmed node), `expected` (next checkpoint), `heading_deg` (compass) and `top_k`. See
  `loc/loc-core.js` (`localize`, `Frames.grab`). Optionally one `motion` part per image (the Orient app's `Motion`
  JSON, `phoneMotion()` in `loc-core.js`) and `previous_step_count` (the step count when the user's node last
  changed): nav-api then leaves out frames pointed at the floor and only confirms a node the user can have walked to.
  `/nav/` sends them with the camera as source.
- Confirming checkpoints: `nav/arrival.js` (`StepDetector`, `GraphTracker`, `NodeConfirmer`; a 1:1 port of
  `nav/navigation/pdr.py` and `arrival.py`). Feed it each `/localize` result (`onLocalization`), each step from
  DeviceMotion (`onStep` with the heading in the map frame: compass − `map.frame.heading_zero_abs_deg`) and the
  user's "I'm here" (`confirmManual`); send its `previous` / `expected` with the next `/localize`. It returns
  arrived / passed / uncertain / off_route events. Route hops carry `arrival_deg` for its heading check. Start the
  tracker from `map.frame.live_step_length_m` (metres per step of this detector; `step_length_m` is per step of the
  map pipeline's, which counts more). Send a burst of 2 frames every second: each burst is one vote, 2 of the last
  3 confirm (measured on Itnig: it confirms far more nodes than 3 frames every 2 s, with no false ones). The
  `/nav/` page's "auto-confirm" uses it (dead reckoning only with the camera as source: a recording's video has no
  live steps).

## Running these pages
- Hosted: map-api serves them at `/` (the index), `/dash/`, `/viz/`, `/loc/` and `/nav/`, and so does helium-nav (behind
  Cloudflare Access). Open one with `?token=<map-api token>&navtoken=<nav-api token>` once, or enter each token when
  asked; they're kept in `localStorage` (nav-api falls back to map-api's token when it has none of its own).
- Against local servers: open any page with `?api=http://127.0.0.1:8450&navapi=http://127.0.0.1:8452` (`nav serve`
  and `nav serve --app nav-api`; remembered in this browser; an empty value goes back to the default).
- Caching: the API serves each page with `?v=<content hash>` on its scripts and styles and `Cache-Control: no-cache`,
  so phones never mix a new page with an old script. Do the same wherever these pages are hosted.
