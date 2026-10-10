// Team index. Recording walks moved to Orient's app (project-description apps/web, /map), which talks to the same
// map-api. The link carries this browser's map-api token (?token=, taken out of the address bar there) so a teammate
// signed in here is not asked again.
"use strict";

const ORIENT_MAP = "https://orient.harshdeepsingh.dev/map";

(() => {
  const link = document.getElementById("record");
  let token = null;
  try { token = localStorage.getItem("nav.token"); } catch {}
  link.href = token ? `${ORIENT_MAP}?token=${encodeURIComponent(token)}` : ORIENT_MAP;
})();
