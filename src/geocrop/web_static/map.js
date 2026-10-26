"use strict";
/* Results map: renders /api/jobs/{id}/map (PNG overlays + GeoJSON in WGS 84) with Leaflet. */
const GeoMap = (() => {
  let map = null, basemap = null, overlays = [], loadedJob = null;
  const TRAFFIC = { green: "#2e9d57", yellow: "#e3b505", red: "#d64545", grey: "#9aa0a6" };
  const el = (id) => document.getElementById(id);

  function ensureMap() {
    if (map || typeof L === "undefined") return map;
    map = L.map("map", { zoomControl: true, attributionControl: true, preferCanvas: true }).setView([44.8, 17.5], 7);
    basemap = L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png",
      { maxZoom: 19, attribution: "&copy; OpenStreetMap contributors" });
    el("basemap-toggle").addEventListener("change", (e) => { if (e.target.checked) basemap.addTo(map); else map.removeLayer(basemap); });
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
      const color = layer.kind === "parcels" ? (TRAFFIC[p.traffic_light] || "#555") : "#1f6feb";
      // parcels: coloured outline + light fill so the class map underneath stays readable
      return { color, weight: layer.kind === "parcels" ? 2.2 : 1.2, fillColor: color,
               fillOpacity: layer.kind === "parcels" ? 0.12 : 0.08 };
    };
  }

  function legend() {
    const box = el("map-legend"); box.replaceChildren();
    for (const { layer, leaflet } of overlays) {
      if (!layer.legend || !layer.legend.length || !map.hasLayer(leaflet)) continue;
      const h = document.createElement("h4"); h.textContent = layer.name; box.append(h);
      const list = document.createElement("ul");
      for (const item of layer.legend) {
        const li = document.createElement("li"), sw = document.createElement("span");
        sw.className = "swatch"; sw.style.background = item.color; li.append(sw, document.createTextNode(item.label)); list.append(li);
      }
      box.append(list);
    }
  }

  function render(payload) {
    ensureMap(); if (!map) return;
    for (const o of overlays) map.removeLayer(o.leaflet); overlays = [];
    const list = el("map-layers"); list.replaceChildren();
    for (const layer of payload.layers) {
      const leaflet = layer.type === "image"
        ? L.imageOverlay(layer.url, layer.bounds, { opacity: 0.85, interactive: false })
        : L.geoJSON(layer.data, {
            style: vectorStyle(layer),
            pointToLayer: (f, latlng) => L.circleMarker(latlng, { radius: 5, color: "#1f6feb", fillOpacity: 0.8 }),
            onEachFeature: (f, l) => l.bindPopup(popup(f.properties || {})),
          });
      if (layer.visible) leaflet.addTo(map);
      overlays.push({ layer, leaflet });
      const li = document.createElement("li"), box = document.createElement("input"), label = document.createElement("label");
      box.type = "checkbox"; box.checked = layer.visible; box.id = `layer-${overlays.length}`;
      label.htmlFor = box.id; label.textContent = layer.name + (layer.truncated ? " (first features only)" : "");
      box.addEventListener("change", () => { if (box.checked) leaflet.addTo(map); else map.removeLayer(leaflet); legend(); });
      li.append(box, label); list.append(li);
    }
    legend();
    if (payload.bounds) map.fitBounds(payload.bounds, { padding: [12, 12] });
    setTimeout(() => map.invalidateSize(), 50);
    const skipped = payload.skipped && payload.skipped.length ? ` Skipped: ${payload.skipped.join("; ")}` : "";
    el("map-status").textContent = payload.layers.length
      ? `${payload.layers.length} layer(s) shown. ${payload.note}${skipped}`
      : `This job has no map outputs.${skipped}`;
  }

  async function load(jobId, api) {
    if (!jobId) return;
    el("map-status").textContent = "Preparing map previews…";
    try { const payload = await api(`/api/jobs/${jobId}/map`); loadedJob = jobId; render(payload); }
    catch (error) { el("map-status").textContent = `Map unavailable: ${error.message}`; }
  }
  return { load, loaded: () => loadedJob };
})();
