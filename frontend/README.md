# Debugging frontend

The phone recorder (`index.html`, `app.js`), the recordings dashboard (`dash/`) and the map viewer (`viz/`).
Vanilla JS, no build step. It's kept in the engine repo for reference and for testing the API directly. The real
frontend can copy any of it.

## Talking to the API
- Base URL: `https://nav-api-613464313064.europe-southwest1.run.app` (set in `config.js`). OpenAPI: `/docs`.
  The formats are `nav/schemas/recording.py` and `nav/schemas/map.py`.
- Auth: `Authorization: Bearer <token>` on every `/api/*` call except `/api/v1/health`. Elements that can't send
  headers (`<video>`, `<img>`, `<audio>`, download links) put it in the URL as `?token=<token>`. Ask the backend
  owner for the token.
- CORS: allowed from `http://localhost:*`, `http://127.0.0.1:*` and helium-nav. Ask for your origin to be added
  (`NAV_CORS_ORIGINS`).
- Large files (videos) are served in ranges of at most 8 MiB. Players handle this; for a download, a plain GET
  streams the whole file.

## Running these pages
- Hosted: the API serves them at `/`, `/dash/` and `/viz/`. Open one with `?token=<token>` once, or enter the
  token when asked; it's kept in `localStorage`.
- Against a local server: open any page with `?api=http://127.0.0.1:8450` (remembered in this browser; `?api=`
  with no value goes back to the default).
- Caching: the API serves each page with `?v=<content hash>` on its scripts and styles and `Cache-Control: no-cache`,
  so phones never mix a new page with an old script. Do the same wherever these pages are hosted.
