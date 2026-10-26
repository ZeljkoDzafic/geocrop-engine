"use strict";
/* Results map: renders /api/jobs/{id}/map (PNG overlays + GeoJSON in WGS 84) with Leaflet. */
const GeoMap = (() => {
  let map = null, basemap = null, overlays = [], loadedJob = null, generation = 0, area = null;
  const BASEMAPS = {
    osm: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png", "&copy; OpenStreetMap contributors", 19],
    "s2-2023": ["https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2023_3857/default/g/{z}/{y}/{x}.jpg",
                "Sentinel-2 cloudless 2023 by EOX IT Services GmbH (contains modified Copernicus Sentinel data 2023)", 15],
    "s2-2024": ["https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2024_3857/default/g/{z}/{y}/{x}.jpg",
                "Sentinel-2 cloudless 2024 by EOX IT Services GmbH (contains modified Copernicus Sentinel data 2024)", 15],
  };
  const TRAFFIC = { green: "#2e9d57", yellow: "#e3b505", red: "#d64545", grey: "#9aa0a6" };
  const VECTOR = "#1f6feb", AREA = "#0a7d5f";
  const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const el = (id) => document.getElementById(id);
  const emit = (detail) => document.dispatchEvent(new CustomEvent("geomap:state", { detail }));

  function setBasemap(value) {
    if (!ensureMap()) return;
    if (basemap) { map.removeLayer(basemap); basemap = null; }
    const choice = BASEMAPS[value];
    // OSM tile policy requires an identifying Referer; send only the page origin (never path or token).
    if (choice) basemap = L.tileLayer(choice[0], { pane: "basemap", maxZoom: 19, maxNativeZoom: choice[2], attribution: choice[1],
                                                  referrerPolicy: "strict-origin-when-cross-origin" }).addTo(map);
    if (el("basemap-select").value !== value) el("basemap-select").value = value;
  }

  function ensureMap() {
    if (map || typeof L === "undefined") return map;
    map = L.map("map", { zoomControl: true, attributionControl: true, preferCanvas: true }).setView([44.8, 17.5], 7);
    map.createPane("basemap"); map.getPane("basemap").style.zIndex = 150;
    el("basemap-select").addEventListener("change", (e) => setBasemap(e.target.value));
    return map;
  }

  function popup(props) {
    const rows = Object.entries(props).filter(([, v]) => v !== null && v !== "")
      .map(([k, v]) => `<tr><th>${escape(k.replaceAll("_", " "))}</th><td>${escape(String(v))}</td></tr>`).join("");
    return `<table class="map-popup">${rows}</table>`;
  }
  function escape(text) { const d = document.createElement("div"); d.textContent = text; return d.innerHTML; }

  function vectorStyle(layer) {
    return (feature) => {
      const p = feature.properties || {};
      const color = layer.kind === "parcels" ? (TRAFFIC[p.traffic_light] || "#555") : VECTOR;
      // parcels: coloured outline + light fill so the class map underneath stays readable
      return { color, weight: layer.kind === "parcels" ? 2.2 : 1.2, fillColor: color,
               fillOpacity: layer.kind === "parcels" ? 0.12 : 0.08 };
    };
  }

  const numeric = (items) => items.length >= 2 && items.every(i => i.label !== "" && Number.isFinite(Number(i.label)));

  function legend() {
    const box = el("map-legend"); box.replaceChildren();
    for (const { layer, leaflet } of overlays) {
      if (!layer.legend || !layer.legend.length || !map.hasLayer(leaflet)) continue;
      const h = document.createElement("h4");
      h.textContent = layer.name.replace(/window \d{4}-\d{2}|\d{4}-\d{2}-\d{2}$/, (t) => formatTime(t)); box.append(h);
      if (numeric(layer.legend)) {
        const ramp = document.createElement("div"), labels = document.createElement("div");
        ramp.className = "ramp";
        ramp.style.background = `linear-gradient(to right, ${layer.legend.map(i => i.color).join(", ")})`;
        labels.className = "ramp-labels";
        for (const item of layer.legend) { const s = document.createElement("span"); s.textContent = item.label; labels.append(s); }
        box.append(ramp, labels);
        continue;
      }
      const list = document.createElement("ul");
      for (const item of layer.legend) {
        const li = document.createElement("li"), sw = document.createElement("span");
        sw.className = "swatch"; sw.style.background = item.color; li.append(sw, document.createTextNode(item.label)); list.append(li);
      }
      box.append(list);
    }
    if (!box.children.length) {
      const p = document.createElement("p"); p.className = "inspector-empty";
      p.textContent = overlays.length ? "No legend for the visible layers." : "Nothing to explain yet.";
      box.append(p);
    }
  }

  /* ----- Imagery: one image at a time, chosen by type (composite / scene), date and product ----- */
  let imagery = [];   // {layer, leaflet}
  const pick = { kind: "composite", index: 0, product: "" };
  const kindOf = (time) => (time.startsWith("window") ? "composite" : "scene");
  const timesOf = (kind) => [...new Set(imagery.filter(i => kindOf(i.layer.time) === kind).map(i => i.layer.time))];
  const productLabel = (p) => p.replace(/\s*\(.*\)$/, "");

  function formatTime(time, short = false) {
    const win = /^window\s+(\d{4})-(\d{2})$/.exec(time), day = /^(\d{4})-(\d{2})-(\d{2})$/.exec(time);
    if (win) return short ? `${MONTHS[win[2] - 1]}${short === "year" ? " " + win[1].slice(2) : ""}` : `${MONTHS[win[2] - 1]} ${win[1]}`;
    if (day) return short ? `${Number(day[3])} ${MONTHS[day[2] - 1]}` : `${Number(day[3])} ${MONTHS[day[2] - 1]} ${day[1]}`;
    return time.replace(/^window\s+/, "");
  }

  function segmented(container, items, current, onPick) {
    container.replaceChildren();
    for (const [value, label] of items) {
      const b = document.createElement("button");
      b.type = "button"; b.setAttribute("role", "radio"); b.dataset.value = value; b.textContent = label;
      b.setAttribute("aria-checked", String(value === current)); b.tabIndex = value === current ? 0 : -1;
      b.addEventListener("click", () => onPick(value));
      b.addEventListener("keydown", (e) => {
        if (!["ArrowLeft", "ArrowRight"].includes(e.key)) return;
        e.preventDefault();
        const buttons = [...container.children], at = buttons.indexOf(b);
        const next = buttons[(at + (e.key === "ArrowRight" ? 1 : -1) + buttons.length) % buttons.length];
        onPick(next.dataset.value); container.querySelector(`[data-value="${CSS.escape(next.dataset.value)}"]`)?.focus();
      });
      container.append(b);
    }
  }

  function renderImageryControls() {
    const times = timesOf(pick.kind);
    pick.index = Math.max(0, Math.min(pick.index, times.length - 1));
    const kinds = ["composite", "scene"].filter(k => timesOf(k).length);
    el("imagery-kind").hidden = kinds.length < 2;
    segmented(el("imagery-kind"), kinds.map(k => [k, `${k === "composite" ? "Monthly composites" : "Scenes"} · ${timesOf(k).length}`]),
              pick.kind, (k) => { pick.kind = k; pick.index = timesOf(k).length - 1; choose(); });
    const products = [...new Set(imagery.map(i => i.layer.product))];
    segmented(el("imagery-product"), products.map(p => [p, productLabel(p)]), pick.product,
              (p) => { pick.product = p; choose(); });
    const useSegments = times.length > 1 && times.length <= 6;
    const sameYear = new Set(times.map(t => (/(\d{4})/.exec(t) || [])[1])).size <= 1;
    el("imagery-dates").hidden = !useSegments;
    el("imagery-control").classList.toggle("slider-mode", !useSegments);
    el("imagery-slider").hidden = useSegments;
    if (useSegments) {
      segmented(el("imagery-dates"), times.map((t, k) => [String(k), formatTime(t, sameYear ? true : "year")]), String(pick.index),
                (k) => { pick.index = Number(k); choose(); });
    } else {
      const range = el("imagery-time");
      range.max = String(Math.max(times.length - 1, 0)); range.value = String(pick.index); range.disabled = times.length < 2;
      range.style.setProperty("--pct", times.length > 1 ? `${(pick.index / (times.length - 1)) * 100}%` : "100%");
      range.setAttribute("aria-valuetext", times[pick.index] ? formatTime(times[pick.index]) : "");
      const ticks = el("imagery-ticks"); ticks.replaceChildren();
      if (times.length > 1) times.forEach((t, k) => {
        const i = document.createElement("i"); i.style.left = `${(k / (times.length - 1)) * 100}%`;
        if (k === 0 || t.slice(0, 7) !== times[k - 1].slice(0, 7)) i.className = "month";
        ticks.append(i);
      });
    }
    const current = times[pick.index];
    const label = el("imagery-label"); label.replaceChildren(document.createTextNode(current ? formatTime(current) : "—"));
    if (current) { const s = document.createElement("small"); s.textContent = pick.kind === "composite" ? "monthly composite" : "single scene"; label.append(s); }
    el("imagery-prev").disabled = pick.index <= 0;
    el("imagery-next").disabled = pick.index >= times.length - 1;
  }

  function choose() { el("imagery-on").checked = true; renderImageryControls(); showImagery(); }

  function showImagery() {
    if (!map) return;
    for (const item of imagery) if (map.hasLayer(item.leaflet)) map.removeLayer(item.leaflet);
    el("imagery-control").classList.toggle("is-off", !el("imagery-on").checked);
    if (!el("imagery-on").checked) { legend(); return; }
    const time = timesOf(pick.kind)[pick.index];
    const chosen = imagery.find(i => i.layer.time === time && i.layer.product === pick.product);
    if (chosen) { chosen.leaflet.setOpacity(el("imagery-opacity").value / 100); chosen.leaflet.addTo(map); chosen.leaflet.bringToBack(); }
    legend();
  }

  function step(delta) {
    if (!imagery.length) return false;
    const times = timesOf(pick.kind), next = pick.index + delta;
    if (next < 0 || next >= times.length) return true;
    pick.index = next; choose();
    return true;
  }

  function clear() {
    if (map) for (const o of overlays) map.removeLayer(o.leaflet);
    overlays = []; imagery = [];
    const list = el("map-layers"); list.replaceChildren();
    const li = document.createElement("li"); li.className = "inspector-empty"; li.textContent = "Layers of the selected job appear here."; list.append(li);
    el("imagery-control").hidden = true; el("imagery-settings").hidden = true;
    el("map-busy").hidden = true; el("map-side").classList.remove("is-loading");
    if (map) legend();
  }

  function render(payload) {
    ensureMap(); if (!map) return;
    for (const o of overlays) map.removeLayer(o.leaflet); overlays = []; imagery = [];
    const list = el("map-layers"); list.replaceChildren();
    for (const layer of payload.layers) {
      const leaflet = layer.type === "image"
        ? L.imageOverlay(layer.url, layer.bounds, { opacity: 0.85, interactive: false })
        : L.geoJSON(layer.data, {
            style: vectorStyle(layer),
            pointToLayer: (f, latlng) => L.circleMarker(latlng, { radius: 5, color: VECTOR, fillOpacity: 0.8 }),
            onEachFeature: (f, l) => {
              l.bindPopup(popup(f.properties || {}));
              // Review view: a click on a parcel opens it in the parcel card instead of a popup
              l.on("click", () => {
                const id = (f.properties || {}).parcel_id ?? (f.properties || {}).segment_id;
                if (id === undefined || id === null || !el("map-shell").classList.contains("in-review")) return;
                map.closePopup();
                document.dispatchEvent(new CustomEvent("geomap:parcel", { detail: { id: String(id) } }));
              });
            },
          });
      overlays.push({ layer, leaflet });
      if (layer.group === "imagery") { imagery.push({ layer, leaflet }); continue; }
      if (layer.visible) leaflet.addTo(map);
      const li = document.createElement("li"), box = document.createElement("input"), label = document.createElement("label");
      box.type = "checkbox"; box.className = "switch"; box.setAttribute("role", "switch");
      box.checked = layer.visible; box.id = `layer-${overlays.length}`;
      label.htmlFor = box.id; label.textContent = layer.name;
      if (layer.truncated) { const s = document.createElement("small"); s.textContent = "First features only"; label.append(s); }
      box.addEventListener("change", () => { if (box.checked) leaflet.addTo(map); else map.removeLayer(leaflet); legend(); });
      li.append(label, box); list.append(li);
    }
    if (!list.children.length) {
      const li = document.createElement("li"); li.className = "inspector-empty";
      li.textContent = imagery.length ? "This job has imagery only." : "No map layers in this job."; list.append(li);
    }
    el("imagery-control").hidden = !imagery.length;
    el("imagery-settings").hidden = !imagery.length;
    if (imagery.length) {
      const def = imagery.find(i => i.layer.visible) || imagery[imagery.length - 1];
      pick.kind = kindOf(def.layer.time); pick.product = def.layer.product;
      pick.index = Math.max(0, timesOf(pick.kind).indexOf(def.layer.time));
      el("imagery-on").checked = imagery.some(i => i.layer.visible) || !payload.layers.some(l => l.group !== "imagery" && l.visible);
      renderImageryControls(); showImagery();
    }
    legend();
    if (payload.bounds) map.fitBounds(payload.bounds, { paddingTopLeft: [24, 64], paddingBottomRight: [24, imagery.length ? 140 : 24] });
    setTimeout(() => map.invalidateSize(), 50);
    const skipped = payload.skipped && payload.skipped.length ? ` Skipped: ${payload.skipped.join("; ")}` : "";
    const other = payload.layers.length - imagery.length;
    const parts = [];
    if (imagery.length) {
      const c = timesOf("composite").length, s = timesOf("scene").length;
      if (c) parts.push(`${c} monthly composite${c === 1 ? "" : "s"}`);
      if (s) parts.push(`${s} scene${s === 1 ? "" : "s"}`);
    }
    if (other) parts.push(`${other} result layer${other === 1 ? "" : "s"}`);
    el("map-status").textContent = payload.layers.length
      ? `${parts.join(" · ")}. ${payload.note}${skipped}`
      : `This job has no map outputs.${skipped}`;
    emit({ state: payload.layers.length ? "ready" : "empty", layers: payload.layers.length, imagery: imagery.length });
  }

  function bindControls() {
    el("imagery-on").addEventListener("change", () => map && showImagery());
    el("imagery-time").addEventListener("input", (e) => { pick.index = Number(e.target.value); choose(); });
    el("imagery-prev").addEventListener("click", () => step(-1));
    el("imagery-next").addEventListener("click", () => step(1));
    const opacity = el("imagery-opacity");
    const syncOpacity = () => { opacity.style.setProperty("--pct", `${opacity.value}%`); el("imagery-opacity-value").textContent = `${opacity.value}%`; };
    opacity.addEventListener("input", () => {
      syncOpacity();
      const shown = imagery.find(i => map && map.hasLayer(i.leaflet)); if (shown) shown.leaflet.setOpacity(opacity.value / 100);
    });
    syncOpacity();
    ensureMap();
  }
  document.addEventListener("DOMContentLoaded", bindControls);

  async function load(jobId, api) {
    if (!jobId) return;
    const request = ++generation;          // only the latest selection may render
    loadedJob = jobId;
    el("map-status").textContent = "Preparing map previews…";
    el("map-busy").hidden = false; el("map-side").classList.add("is-loading");
    emit({ state: "loading" });
    try {
      const payload = await api(`/api/jobs/${jobId}/map`);
      if (request !== generation) return;  // a newer job was selected meanwhile: discard this response
      render(payload);
    } catch (error) {
      if (request === generation) { el("map-status").textContent = `Map unavailable: ${error.message}`; emit({ state: "error", message: error.message }); }
    } finally {
      if (request === generation) { el("map-busy").hidden = true; el("map-side").classList.remove("is-loading"); }
    }
  }
  function forget() { generation++; loadedJob = null; el("map-busy").hidden = true; el("map-side").classList.remove("is-loading"); }

  /* ----- Area of interest: preview rectangle and drawing (drag, or two clicks/taps) ----- */
  function showArea(bbox, fit = false) {
    if (!ensureMap()) return;
    if (area) { map.removeLayer(area); area = null; }
    if (!bbox) return;
    const [w, s, e, n] = bbox;
    area = L.rectangle([[s, w], [n, e]], { color: AREA, weight: 2, dashArray: "6 5", fill: true, fillOpacity: 0.06, interactive: false }).addTo(map);
    if (fit) map.fitBounds(area.getBounds(), { padding: [48, 48], maxZoom: 13 });
  }

  function pickArea() {
    if (!ensureMap()) return Promise.resolve(null);
    return new Promise((resolve) => {
      const container = map.getContainer();
      let start = null, startPoint = null, rect = null, viaMouse = false;
      const draw = (a, b) => {
        const bounds = L.latLngBounds(a, b);
        if (!rect) rect = L.rectangle(bounds, { color: AREA, weight: 2, dashArray: "6 5", fillOpacity: 0.1, interactive: false }).addTo(map);
        else rect.setBounds(bounds);
        return bounds;
      };
      const finish = (bounds) => {
        map.off("mousedown", down); map.off("mousemove", move); map.off("mouseup", up); map.off("click", click);
        document.removeEventListener("keydown", key, true);
        el("pick-cancel").removeEventListener("click", cancel);
        if (rect) map.removeLayer(rect);
        container.classList.remove("is-picking"); map.dragging.enable(); map.boxZoom.enable(); map.doubleClickZoom.enable();
        el("pick-hint").hidden = true;
        if (!bounds) { resolve(null); return; }
        const r = (v) => Math.round(v * 1e5) / 1e5;
        const bbox = [r(bounds.getWest()), r(bounds.getSouth()), r(bounds.getEast()), r(bounds.getNorth())];
        if (bbox[0] === bbox[2] || bbox[1] === bbox[3]) { resolve(null); return; }
        showArea(bbox); resolve(bbox);
      };
      const down = (e) => {
        viaMouse = true;
        if (start) { finish(draw(start, e.latlng)); return; }      // second corner of a two-click selection
        start = e.latlng; startPoint = e.containerPoint; draw(start, start);
      };
      const move = (e) => { if (start) draw(start, e.latlng); };
      const up = (e) => {
        if (start && startPoint && e.containerPoint.distanceTo(startPoint) > 8) finish(draw(start, e.latlng));
        startPoint = null;   // a click without dragging: wait for the second corner
      };
      const click = (e) => {   // touch screens: tap two corners
        if (viaMouse) return;
        if (!start) { start = e.latlng; draw(start, start); } else finish(draw(start, e.latlng));
      };
      const key = (e) => { if (e.key === "Escape") { e.stopPropagation(); finish(null); } };
      const cancel = () => finish(null);
      container.classList.add("is-picking"); map.dragging.disable(); map.boxZoom.disable(); map.doubleClickZoom.disable();
      el("pick-hint").hidden = false;
      map.on("mousedown", down); map.on("mousemove", move); map.on("mouseup", up); map.on("click", click);
      document.addEventListener("keydown", key, true);
      el("pick-cancel").addEventListener("click", cancel);
    });
  }

  /* ----- Review: highlight one parcel (white outline on a dark halo, readable on any imagery) ----- */
  let focus = null;
  function focusParcel(feature, { fly = true } = {}) {
    if (!ensureMap()) return;
    if (focus) { map.removeLayer(focus); focus = null; }
    if (!feature) return;
    focus = L.layerGroup([
      L.geoJSON(feature, { interactive: false, style: { color: "#0b0f0d", weight: 6, opacity: 0.45, fill: false } }),
      L.geoJSON(feature, { interactive: false, style: { color: "#ffffff", weight: 2.5, opacity: 1, fillColor: "#ffffff", fillOpacity: 0.06 } }),
    ]).addTo(map);
    const [[s, w], [n, e]] = feature.bounds;
    const dx = (e - w) * 1.6 + 0.0005, dy = (n - s) * 1.6 + 0.0005;
    const target = L.latLngBounds([[s - dy, w - dx], [n + dy, e + dx]]);
    const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
    map.invalidateSize();
    if (fly && !reduce) map.flyToBounds(target, { maxZoom: 17, duration: 0.45 });
    else map.fitBounds(target, { maxZoom: 17 });
  }

  return {
    focusParcel,
    load, forget, clear, loaded: () => loadedJob, step, showArea, pickArea, setBasemap,
    hasImagery: () => imagery.length > 0,
    invalidate: () => { if (map) setTimeout(() => map.invalidateSize(), 60); },
  };
})();
