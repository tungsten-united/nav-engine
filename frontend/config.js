// Where the two APIs are and how to authenticate; loaded first by every page.
//   NAV      map-api: recordings, map builds, review, map files (the phone recorder, /dash/, /viz/)
//   NAV.nav  nav-api: the live API (Orient's orchestrator calls it; here /loc/, /nav/): published maps, routes, localization
// Both default to their Cloud Run services. For local servers open any page once with
// ?api=http://127.0.0.1:8450&navapi=http://127.0.0.1:8452 (remembered in this browser; an empty value goes back to
// the default). Each API's token is asked for on its first 401 and kept in localStorage; ?token= (map-api) and
// ?navtoken= (nav-api) in the page URL set them too (then removed from the address bar). Without its own token,
// nav-api is sent map-api's.
(() => {
  const DEFAULT_MAP_API = "https://map-api-613464313064.europe-southwest1.run.app";
  const DEFAULT_NAV_API = "https://nav-api-613464313064.europe-southwest1.run.app";
  const get = (k) => { try { return localStorage.getItem(k); } catch { return null; } };
  const set = (k, v) => { try { v ? localStorage.setItem(k, v) : localStorage.removeItem(k); } catch {} };
  const trim = (u) => (u || "").trim().replace(/\/+$/, "");
  const q = new URLSearchParams(location.search);
  if (q.has("api")) set("nav.apiBase", trim(q.get("api")));
  if (q.has("navapi")) set("nav.navApiBase", trim(q.get("navapi")));
  // Before the split nav-api served everything: a remembered ?api= pointing there now means map-api.
  if (trim(get("nav.apiBase")) === DEFAULT_NAV_API) set("nav.apiBase", null);
  let secret = false;
  for (const [param, key] of [["token", "nav.token"], ["navtoken", "nav.navToken"]]) {
    if (!q.has(param)) continue;
    set(key, q.get(param).trim());
    q.delete(param);
    secret = true;
  }
  if (secret) history.replaceState(null, "", location.pathname + (q.size ? `?${q}` : "") + location.hash);

  function client(name, base, tokenKey, fallbackKey) {
    const token = () => get(tokenKey) || (fallbackKey ? get(fallbackKey) : null);
    let asked = false;
    const c = {
      name,
      base,
      API: base + "/api/v1",
      docs: base + "/docs",
      headers: (h = {}) => (token() ? { ...h, Authorization: `Bearer ${token()}` } : h),
      // For <video>/<img>/<audio>/<a> URLs, which can't send headers.
      url: (u) => (token() ? `${u}${u.includes("?") ? "&" : "?"}token=${encodeURIComponent(token())}` : u),
      async fetch(url, opts = {}) {
        const r = await fetch(url, { ...opts, headers: c.headers(opts.headers) });
        if (r.status === 401 && !asked) {
          asked = true;
          const t = prompt(`API token for ${name} (${base})`);
          if (t) { set(tokenKey, t.trim()); location.reload(); }
        }
        return r;
      },
    };
    return c;
  }

  const NAV = client("map-api", trim(get("nav.apiBase")) || DEFAULT_MAP_API, "nav.token");
  NAV.nav = client("nav-api", trim(get("nav.navApiBase")) || DEFAULT_NAV_API, "nav.navToken", "nav.token");
  window.NAV = NAV;
})();
