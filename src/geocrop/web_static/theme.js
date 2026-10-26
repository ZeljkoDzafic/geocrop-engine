"use strict";
/* Applies the stored appearance before first paint (light / dark; absent = follow the system). */
(() => {
  try {
    const saved = localStorage.getItem("geocrop-theme");
    if (saved === "light" || saved === "dark") document.documentElement.dataset.theme = saved;
  } catch { /* storage unavailable: follow the system setting */ }
})();
