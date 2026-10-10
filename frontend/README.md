# Debugging frontend

The phone recorder (`index.html`, `app.js`), the recordings dashboard (`dash/`), the map viewer (`viz/`), and the
debuggers of the live API: localization (`loc/`) and navigation (`nav/`). Vanilla JS, no build step. It's kept in the
engine repo for reference and for testing the APIs directly. The real frontend can copy any of it.

## Talking to the APIs
Two services, two base URLs (both set in `config.js`; OpenAPI at `/docs` on each):

| | base URL | what | used by |
|---|---|---|---|
| **map-api** | `https://map-api-613464313064.europe-southwest1.run.app` | recordings, map builds, review, map files | recorder, `dash/`, `viz/` (`NAV`) |
| **nav-api** | `https://nav-api-613464313064.europe-southwest1.run.app` | the live API: published maps, routes, localization | `loc/`, `nav/` (`NAV.nav`), the navigation app |

- Formats: `nav/schemas/recording.py`, `nav/schemas/map.py`, `nav/schemas/navigation.py`.
- Auth: each service has its own token. Send `Authorization: Bearer <token>` on every `/api/*` call except
  `/api/v1/health`. Elements that can't send headers (`<video>`, `<img>`, `<audio>`, download links) put it in the URL
  as `?token=<token>`. Ask the backend owner for the tokens. The app only needs nav-api's.
- CORS: both allow `http://localhost:*`, `http://127.0.0.1:*` and helium-nav; nav-api also allows map-api's pages. Ask
  for your origin to be added (`NAV_CORS_ORIGINS` in `deploy/*.env.yaml`).
- Large files (videos) are served in ranges of at most 8 MiB. Players handle this; for a download, a plain GET
  streams the whole file.
- Localizing a frame: `POST {nav-api}/api/v1/maps/{map_id}/localize` with multipart `images` (1–4 JPEGs, ~768 px),
  optional `previous` (last confirmed node), `expected` (next checkpoint), `heading_deg` (compass) and `top_k`. See
  `loc/loc-core.js` (`localize`, `Frames.grab`).

## Running these pages
- Hosted: map-api serves them at `/`, `/dash/`, `/viz/`, `/loc/` and `/nav/`, and so does helium-nav (behind
  Cloudflare Access). Open one with `?token=<map-api token>&navtoken=<nav-api token>` once, or enter each token when
  asked; they're kept in `localStorage` (nav-api falls back to map-api's token when it has none of its own).
- Against local servers: open any page with `?api=http://127.0.0.1:8450&navapi=http://127.0.0.1:8452` (`nav serve`
  and `nav serve --app nav-api`; remembered in this browser; an empty value goes back to the default).
- Caching: the API serves each page with `?v=<content hash>` on its scripts and styles and `Cache-Control: no-cache`,
  so phones never mix a new page with an old script. Do the same wherever these pages are hosted.
