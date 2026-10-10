// Where the API is and how to authenticate; loaded first by the phone app, /dash/ and /viz/.
// Defaults to the Cloud Run worker. For a local server: open any page once with ?api=http://127.0.0.1:8450
// (remembered in this browser; ?api= with no value goes back to the default). The API token is asked for on the
// first 401 and kept in localStorage; ?token=... in the page URL sets it too (then removed from the address bar).
(() => {
  const DEFAULT_API_BASE = "https://nav-api-613464313064.europe-southwest1.run.app";
  const get = (k) => { try { return localStorage.getItem(k); } catch { return null; } };
  const set = (k, v) => { try { v ? localStorage.setItem(k, v) : localStorage.removeItem(k); } catch {} };
  const q = new URLSearchParams(location.search);
  if (q.has("api")) set("nav.apiBase", q.get("api").trim());
  if (q.has("token")) {
    set("nav.token", q.get("token").trim());
    q.delete("token");
    history.replaceState(null, "", location.pathname + (q.size ? `?${q}` : "") + location.hash);
  }
  const base = (get("nav.apiBase") || DEFAULT_API_BASE).replace(/\/+$/, "");
  let asked = false;
  const NAV = {
    base,
    API: base + "/api/v1",
    headers: (h = {}) => (get("nav.token") ? { ...h, Authorization: `Bearer ${get("nav.token")}` } : h),
    // For <video>/<img>/<audio>/<a> URLs, which can't send headers.
    url: (u) => (get("nav.token") ? `${u}${u.includes("?") ? "&" : "?"}token=${encodeURIComponent(get("nav.token"))}` : u),
    async fetch(url, opts = {}) {
      const r = await fetch(url, { ...opts, headers: NAV.headers(opts.headers) });
      if (r.status === 401 && !asked) {
        asked = true;
        const t = prompt(`API token for ${base}`);
        if (t) { set("nav.token", t.trim()); location.reload(); }
      }
      return r;
    },
  };
  window.NAV = NAV;
})();
