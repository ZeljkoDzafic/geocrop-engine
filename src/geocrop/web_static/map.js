"use strict";
/* Results map: renders /api/jobs/{id}/map (PNG overlays + GeoJSON in WGS 84) with Leaflet. */
const GeoMap = (() => {
  let map = null, basemap = null, overlays = [], loadedJob = null, generation = 0;
  const BASEMAPS = {
    osm: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png", "&copy; OpenStreetMap contributors", 19],
    "s2-2023": ["https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2023_3857/default/g/{z}/{y}/{x}.jpg",
                "Sentinel-2 cloudless 2023 by EOX IT Services GmbH (contains modified Copernicus Sentinel data 2023)", 15],
    "s2-2024": ["https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2024_3857/default/g/{z}/{y}/{x}.jpg",
                "Sentinel-2 cloudless 2024 by EOX IT Services GmbH (contains modified Copernicus Sentinel data 2024)", 15],
  };
  const TRAFFIC = { green: "#2e9d57", yellow: "#e3b505", red: "#d64545", grey: "#9aa0a6" };
  const el = (id) => document.getElementById(id);

  function ensureMap() {
    if (map || typeof L === "undefined") return map;
    map = L.map("map", { zoomControl: true, attributionControl: true, preferCanvas: true }).setView([44.8, 17.5], 7);
    map.createPane("basemap"); map.getPane("basemap").style.zIndex = 150;
    el("basemap-select").addEventListener("change", (e) => {
      if (basemap) { map.removeLayer(basemap); basemap = null; }
      const choice = BASEMAPS[e.target.value];
      // OSM tile policy requires an identifying Referer; send only the page origin (never path or token).
      if (choice) basemap = L.tileLayer(choice[0], { pane: "basemap", maxZoom: 19, maxNativeZoom: choice[2], attribution: choice[1],
                                                    referrerPolicy: "strict-origin-when-cross-origin" }).addTo(map);
    });
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

  let imagery = [];   // {layer, leaflet}

  function showImagery() {
    for (const item of imagery) if (map.hasLayer(item.leaflet)) map.removeLayer(item.leaflet);
    if (!el("imagery-on").checked) { legend(); return; }
    const pick = imagery.find(i => i.layer.time === el("imagery-time").value && i.layer.product === el("imagery-product").value);
    if (pick) { pick.leaflet.setOpacity(el("imagery-opacity").value / 100); pick.leaflet.addTo(map); pick.leaflet.bringToBack(); }
    legend();
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
            pointToLayer: (f, latlng) => L.circleMarker(latlng, { radius: 5, color: "#1f6feb", fillOpacity: 0.8 }),
            onEachFeature: (f, l) => l.bindPopup(popup(f.properties || {})),
          });
      overlays.push({ layer, leaflet });
      if (layer.group === "imagery") { imagery.push({ layer, leaflet }); continue; }
      if (layer.visible) leaflet.addTo(map);
      const li = document.createElement("li"), box = document.createElement("input"), label = document.createElement("label");
      box.type = "checkbox"; box.checked = layer.visible; box.id = `layer-${overlays.length}`;
      label.htmlFor = box.id; label.textContent = layer.name + (layer.truncated ? " (first features only)" : "");
      box.addEventListener("change", () => { if (box.checked) leaflet.addTo(map); else map.removeLayer(leaflet); legend(); });
      li.append(box, label); list.append(li);
    }
    const control = el("imagery-control");
    control.hidden = !imagery.length;
    if (imagery.length) {
      const times = [...new Set(imagery.map(i => i.layer.time))], products = [...new Set(imagery.map(i => i.layer.product))];
      el("imagery-time").replaceChildren(...times.map(t => new Option(t, t)));
      el("imagery-product").replaceChildren(...products.map(p => new Option(p, p)));
      const def = imagery.find(i => i.layer.visible);
      if (def) { el("imagery-time").value = def.layer.time; el("imagery-product").value = def.layer.product; }
      el("imagery-on").checked = !!def || !payload.layers.some(l => l.group !== "imagery" && l.visible);
      showImagery();
    }
    legend();
    if (payload.bounds) map.fitBounds(payload.bounds, { padding: [12, 12] });
    setTimeout(() => map.invalidateSize(), 50);
    const skipped = payload.skipped && payload.skipped.length ? ` Skipped: ${payload.skipped.join("; ")}` : "";
    const shown = payload.layers.length - imagery.length + (imagery.length ? 1 : 0);
    el("map-status").textContent = payload.layers.length
      ? `${shown} layer group(s)${imagery.length ? `, ${new Set(imagery.map(i => i.layer.time)).size} image dates/windows` : ""}. ${payload.note}${skipped}`
      : `This job has no map outputs.${skipped}`;
  }

  function bindImageryControls() {
    for (const id of ["imagery-time", "imagery-product", "imagery-on"]) el(id).addEventListener("change", () => map && showImagery());
    el("imagery-opacity").addEventListener("input", () => { const pick = imagery.find(i => map && map.hasLayer(i.leaflet)); if (pick) pick.leaflet.setOpacity(el("imagery-opacity").value / 100); });
  }
  document.addEventListener("DOMContentLoaded", bindImageryControls);

  async function load(jobId, api) {
    if (!jobId) return;
    const request = ++generation;          // only the latest selection may render
    loadedJob = jobId;
    el("map-status").textContent = "Preparing map previews…";
    try {
      const payload = await api(`/api/jobs/${jobId}/map`);
      if (request !== generation) return;  // a newer job was selected meanwhile: discard this response
      render(payload);
    } catch (error) {
      if (request === generation) el("map-status").textContent = `Map unavailable: ${error.message}`;
    }
  }
  function forget() { generation++; loadedJob = null; }
  return { load, forget, loaded: () => loadedJob };
})();
