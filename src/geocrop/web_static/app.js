"use strict";
/* GeoCrop workspace: connection, views, guided analysis, command forms and palette, jobs, logs and files. */
const $ = (id) => document.getElementById(id);
const state = {
  token: "", commands: [], jobs: [], selected: null, offset: 0, log: "", generation: 0, job: null,
  command: null, view: "map", known: new Map(), mapState: "idle", picking: false, files: "", step: 1,
};
const STATUS = { queued: "Queued", running: "Running", succeeded: "Succeeded", failed: "Failed", cancelled: "Cancelled", interrupted: "Interrupted" };
const terminal = new Set(["succeeded", "failed", "cancelled", "interrupted"]);
const MAP_COMMANDS = new Set(["quicklook", "demo", "run", "predict"]);
const SEMBERIJA = { bbox: [19.10, 44.72, 19.16, 44.76], start: "2023-06-01", end: "2023-08-31" };
const isMac = /Mac|iPhone|iPad/i.test(navigator.platform || navigator.userAgent);
const VIEWS = {
  map: ["Map", "Imagery and results of the selected job"],
  commands: ["Commands", "Every engine command as a form, generated from the CLI parser"],
  jobs: ["Jobs", "History, status and output of every run"],
  terminal: ["Terminal", "Engine commands, typed as in your shell"],
  api: ["Python & API", "Use the same engine from Jupyter or over HTTP"],
};
const GROUPS = [
  ["Get started", "sparkle", ["quicklook", "demo", "indices", "config-check"]],
  ["Imagery & data", "satellite", ["catalog", "stac-search", "download", "grid", "s2-import", "stac-import", "s2-composite", "snap-graph",
    "s1-import", "s1-composite", "cloudmask", "merge-stacks", "inventory"]],
  ["Features & samples", "layers", ["features", "ts-features", "samples", "signatures", "markers", "reduce-features", "variogram", "folds", "check", "audit"]],
  ["Models", "cube", ["run", "train", "predict", "benchmark", "benchmark-models", "ablation", "parcel-model"]],
  ["Parcels & segmentation", "vector", ["parcels", "normalize-parcels", "segment-obia", "segment-patches", "segment-unet-train", "segment-unet-infer",
    "boundaries-to-parcels", "evaluate-segments", "evaluate-boundaries", "benchmark-segmentation"]],
  ["Results & reporting", "doc", ["validate", "area", "fieldwork", "export", "manifest"]],
];
const groupOf = (name) => GROUPS.find(g => g[2].includes(name)) || ["Other", "command", []];

/* ---------- Small helpers ---------- */
function icon(name, cls = "icon") {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg"), use = document.createElementNS("http://www.w3.org/2000/svg", "use");
  svg.setAttribute("class", cls); svg.setAttribute("aria-hidden", "true"); use.setAttribute("href", `#i-${name}`); svg.append(use);
  return svg;
}
function h(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") node.className = v; else if (k === "text") node.textContent = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v); else node.setAttribute(k, v === true ? "" : v);
  }
  node.append(...children.filter(c => c !== null && c !== undefined && c !== false));
  return node;
}
const quote = (arg) => (/^[\w@%+=:,./-]+$/.test(arg) ? arg : `'${arg.replaceAll("'", "'\\''")}'`);
const commandLine = (argv) => ["geocrop", ...argv].map(quote).join(" ");
function showCommand(node, argv) {   // one unbreakable span per argument; flags tinted
  const parts = ["geocrop", ...argv].map(quote);
  node.replaceChildren(...parts.flatMap((part, k) => [k ? " " : "", h("span", { class: part.startsWith("--") ? "argv argv-flag" : "argv", text: part })]));
}
const stamp = () => { const d = new Date(), p = (n) => String(n).padStart(2, "0"); return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`; };
const rtf = new Intl.RelativeTimeFormat("en", { numeric: "auto" });
function relative(iso) {
  if (!iso) return "—";
  const seconds = (Date.parse(iso) - Date.now()) / 1000, abs = Math.abs(seconds);
  if (abs < 45) return "just now";
  if (abs < 3600) return rtf.format(Math.round(seconds / 60), "minute");
  if (abs < 86400) return rtf.format(Math.round(seconds / 3600), "hour");
  if (abs < 7 * 86400) return rtf.format(Math.round(seconds / 86400), "day");
  return new Date(iso).toLocaleDateString("en", { day: "numeric", month: "short", year: "numeric" });
}
function duration(job) {
  if (!job.started) return job.status === "queued" ? "waiting" : "—";
  const end = job.finished ? Date.parse(job.finished) : (terminal.has(job.status) ? null : Date.now());
  if (end === null) return "—";
  const s = Math.max(0, Math.round((end - Date.parse(job.started)) / 1000));
  if (s < 60) return `${s} s`;
  if (s < 3600) return `${Math.floor(s / 60)} min ${s % 60} s`;
  return `${Math.floor(s / 3600)} h ${Math.floor((s % 3600) / 60)} min`;
}
function bytes(n) {
  if (n < 1024) return `${n} B`;
  const units = ["KB", "MB", "GB"]; let v = n / 1024, u = 0;
  while (v >= 1024 && u < units.length - 1) { v /= 1024; u++; }
  return `${v < 10 ? v.toFixed(1) : Math.round(v)} ${units[u]}`;
}
function fileKind(path) {
  const ext = path.toLowerCase().split(".").pop();
  if (["tif", "tiff", "vrt", "nc", "jp2"].includes(ext)) return ["raster", "raster"];
  if (["gpkg", "geojson", "shp", "parquet", "kml"].includes(ext)) return ["vector", "vector"];
  if (["png", "jpg", "jpeg", "svg", "webp"].includes(ext)) return ["image", "image"];
  if (["joblib", "pt", "pth", "onnx", "pkl"].includes(ext)) return ["model", "cube"];
  return ["doc", "doc"];
}

/* ---------- Toasts ---------- */
function toast({ title, text = "", kind = "success", action = null, timeout = 6000 }) {
  const glyph = { success: "check", error: "alert", info: "info" }[kind];
  const close = () => { node.classList.add("leaving"); setTimeout(() => node.remove(), 220); };
  const body = h("div", { class: "toast-body" }, h("strong", { text: title }), text ? h("p", { text }) : null,
    action ? h("button", { class: "btn btn-small btn-tinted", type: "button", onclick: () => { action.run(); close(); } }, action.label) : null);
  const node = h("div", { class: "toast glass", role: kind === "error" ? "alert" : "status", "data-kind": kind }, icon(glyph), body,
    h("button", { class: "icon-button", type: "button", "aria-label": "Dismiss", onclick: close }, icon("close")));
  $("toasts").append(node);
  while ($("toasts").children.length > 4) $("toasts").firstElementChild.remove();
  if (timeout) setTimeout(close, timeout);
}
function message(text) { if (text) toast({ title: "Something went wrong", text, kind: "error", timeout: 9000 }); }
async function copy(text) {
  try { await navigator.clipboard.writeText(text); toast({ title: "Copied to clipboard", kind: "info", timeout: 2000 }); }
  catch { message("The browser did not allow copying. Select the text and copy it manually."); }
}

/* ---------- API ---------- */
async function api(path, options = {}) {
  const response = await fetch(path, { ...options, headers: { "Authorization": `Bearer ${state.token}`, "Content-Type": "application/json", ...options.headers } });
  if (!response.ok) { let error; try { error = await response.json(); } catch { error = {}; } throw new Error(typeof error.detail === "string" ? error.detail : `HTTP ${response.status}`); }
  return response.json();
}
async function download(path, name) {
  const response = await fetch(path, { headers: { Authorization: `Bearer ${state.token}` } });
  if (!response.ok) throw new Error(`Download failed: HTTP ${response.status}`);
  const url = URL.createObjectURL(await response.blob()), link = document.createElement("a");
  link.href = url; link.download = name; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/* ---------- Appearance ---------- */
function effectiveTheme() {
  return document.documentElement.dataset.theme || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
}
function syncThemeLabels() {
  const label = effectiveTheme() === "dark" ? "Switch to light appearance" : "Switch to dark appearance";
  for (const id of ["theme-toggle", "theme-compact"]) { $(id).setAttribute("aria-label", label); $(id).title = label; }
}
function toggleTheme() {
  const next = effectiveTheme() === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem("geocrop-theme", next); } catch { /* not persisted: fine */ }
  syncThemeLabels();
}

/* ---------- Views ---------- */
function setView(name) {
  if (!VIEWS[name]) return;
  if (state.view !== name) window.scrollTo({ top: 0 });
  state.view = name;
  for (const view of Object.keys(VIEWS)) $(`view-${view}`).hidden = view !== name;
  for (const b of document.querySelectorAll("[data-view]")) {
    if (b.dataset.view === name) b.setAttribute("aria-current", "page"); else b.removeAttribute("aria-current");
  }
  [$("view-title").textContent, $("view-subtitle").textContent] = VIEWS[name];
  document.title = `${VIEWS[name][0]} · GeoCrop Workspace`;
  const slot = $(name === "jobs" ? "detail-slot-jobs" : "detail-slot-map");
  if ($("job-detail").parentElement !== slot) slot.append($("job-detail"));
  if (name === "map") GeoMap.invalidate();
  if (name !== "map" && $("map-shell").classList.contains("is-expanded")) toggleExpand(false);
}
function toggleExpand(force) {
  const shell = $("map-shell"), on = force ?? !shell.classList.contains("is-expanded");
  if (on && state.view !== "map") setView("map");
  shell.classList.toggle("is-expanded", on); document.body.classList.toggle("map-expanded", on);
  $("map-expand").setAttribute("aria-pressed", String(on));
  $("map-expand").setAttribute("aria-label", on ? "Restore map" : "Expand map");
  $("map-expand").title = on ? "Restore map (F)" : "Expand map (F)";
  GeoMap.invalidate();
}

/* ---------- Map empty states ---------- */
function updateMapEmpty() {
  const job = state.job && state.job.id === state.selected ? state.job : null;
  let mode = "welcome";
  if (state.picking || state.mapState === "loading" || (state.mapState === "ready")) mode = "hidden";
  else if (job && !terminal.has(job.status)) mode = "progress";
  else if (job && job.status !== "succeeded") mode = "failed";
  else if (job && (state.mapState === "empty" || state.mapState === "error")) mode = "nomap";
  else if (job && state.mapState === "idle") mode = "hidden";
  else if (!job && state.selected) mode = "hidden";
  const card = $("map-empty"); card.hidden = mode === "hidden";
  if (mode === "hidden" || card.dataset.mode === mode + (job ? job.status : "")) { if (mode === "progress") progressText(job); return; }
  card.dataset.mode = mode + (job ? job.status : "");
  const glyph = $("empty-icon");
  glyph.replaceChildren(mode === "progress" ? h("span", { class: "spinner", "aria-hidden": "true" }) : icon(mode === "failed" ? "alert" : mode === "nomap" ? "layers" : "satellite"));
  $("empty-steps").hidden = mode !== "welcome";
  $("empty-foot").hidden = mode !== "welcome";
  $("empty-actions").hidden = mode === "progress";
  $("example-quicklook").hidden = mode === "failed";
  if (mode === "welcome") {
    $("map-status").textContent = state.jobs.length ? "Select a finished job to see its outputs on the map." : "Your first quicklook will appear here.";
    $("empty-title").textContent = "From an area and dates to imagery";
    $("empty-text").textContent = "Draw an area, choose a period, and GeoCrop fetches cloud-screened Sentinel-2 composites — true colour, false colour and NDVI — right here on the map.";
  } else if (mode === "progress") progressText(job);
  else if (mode === "failed") {
    $("empty-title").textContent = `${job.argv[0]} did not finish`;
    $("empty-text").textContent = `The job ended as ${STATUS[job.status].toLowerCase()}${job.returncode !== null ? ` (exit code ${job.returncode})` : ""}. The output below explains why.`;
  } else {
    $("empty-title").textContent = "Nothing to show on the map";
    $("empty-text").textContent = `${job.argv[0]} wrote no rasters or vectors the map can display. Its output and files are listed below.`;
  }
}
function progressText(job) {
  const waiting = job.status === "queued";
  $("empty-title").textContent = waiting ? `${job.argv[0]} is waiting for the worker…` : `Running ${job.argv[0]}…`;
  $("empty-text").textContent = (job.argv[0] === "quicklook"
    ? "Searching Sentinel-2 acquisitions and building monthly composites. Imagery appears here when it is done"
    : "Results appear here when the job finishes") + (waiting ? "." : ` · ${duration(job)} so far.`);
}

/* ---------- Jobs ---------- */
function jobItem(job) {
  const current = job.id === state.selected;
  return h("li", {}, h("button", { class: "job-item", type: "button", "data-job": job.id, "aria-current": current ? "true" : null,
    title: commandLine(job.argv), onclick: () => selectJob(job.id) },
    h("span", { class: "dot", "data-status": job.status, "aria-label": STATUS[job.status] || job.status, role: "img" }),
    h("span", { class: "job-name", text: job.argv[0] }),
    h("span", { class: "job-when", text: relative(job.created) }),
    h("span", { class: "job-args", text: job.argv.slice(1).join(" ") || "no arguments" })));
}
function jobRow(job) {
  const exit = job.returncode === null ? "—" : String(job.returncode);
  const row = h("button", { class: "jobs-row", type: "button", role: "row", "data-job": job.id, "aria-current": job.id === state.selected ? "true" : null, onclick: () => selectJob(job.id) },
    h("span", { class: "cmd", role: "cell" }, h("strong", { text: job.argv[0] }), h("span", { text: job.argv.slice(1).join(" ") || "no arguments", title: commandLine(job.argv) })),
    h("span", { role: "cell" }, h("span", { class: "pill", "data-status": job.status, text: STATUS[job.status] || job.status })),
    h("span", { class: "num", role: "cell", text: relative(job.created), title: new Date(job.created).toLocaleString() }),
    h("span", { class: "num", role: "cell", text: duration(job) }),
    h("span", { class: "num", role: "cell", text: exit }),
    h("span", { class: "meta-compact", text: `${relative(job.created)} · ${duration(job)}${job.returncode !== null ? ` · exit ${exit}` : ""}` }));
  row.addEventListener("click", () => requestAnimationFrame(() => $("job-detail").scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" })));
  return row;
}
function renderJobs() {
  const active = state.jobs.filter(j => !terminal.has(j.status)).length;
  $("total").textContent = state.jobs.length; $("running").textContent = active;
  $("succeeded").textContent = state.jobs.filter(j => j.status === "succeeded").length;
  $("failed").textContent = state.jobs.filter(j => ["failed", "interrupted"].includes(j.status)).length;
  $("jobs-badge").hidden = !active; $("jobs-badge").textContent = active;
  const focused = document.activeElement?.dataset?.job, inList = document.activeElement?.closest?.("#jobs") ? "jobs" : document.activeElement?.closest?.("#jobs-page") ? "jobs-page" : null;
  if (!state.jobs.length) {
    $("jobs").replaceChildren(h("li", { class: "job-empty", text: "No jobs yet. Start with a new analysis." }));
    $("jobs-page").replaceChildren(h("div", { class: "jobs-empty" }, "No jobs yet. ", h("button", { class: "link-button", type: "button", text: "Start a new analysis", onclick: openAnalysis }), "."));
  } else {
    $("jobs").replaceChildren(...state.jobs.map(jobItem));
    $("jobs-page").replaceChildren(...state.jobs.map(jobRow));
  }
  if (focused && inList) $(inList).querySelector(`[data-job="${CSS.escape(focused)}"]`)?.focus();
}
function notify(job) {
  const name = job.argv[0], took = duration(job);
  const show = { label: MAP_COMMANDS.has(name) ? "Show on map" : "Show output", run: () => { selectJob(job.id); setView(MAP_COMMANDS.has(name) ? "map" : "jobs"); } };
  if (job.status === "succeeded") toast({ title: `${name} finished`, text: `Took ${took}.${name === "quicklook" ? " Imagery is ready on the map." : ""}`, action: job.id === state.selected && ["map", "jobs"].includes(state.view) ? null : show });
  else if (job.status === "cancelled") toast({ title: `${name} stopped`, text: "Partial outputs are kept for inspection.", kind: "info" });
  else toast({ title: `${name} ${job.status}`, text: `${job.returncode !== null ? `Exit code ${job.returncode}. ` : ""}The output shows what went wrong.`, kind: "error", action: show, timeout: 12000 });
}
function selectJob(id, { keepArea = false } = {}) {
  if (!id) return;
  const changed = id !== state.selected;
  state.selected = id;
  if (changed) {
    state.offset = 0; state.log = ""; state.generation++; state.files = ""; state.mapState = "idle";
    GeoMap.forget(); GeoMap.clear();                 // drop any map request of the previous job
    if (!keepArea) GeoMap.showArea(null);
    $("logs").textContent = "Loading output…"; $("artifact-list").replaceChildren();
  }
  for (const b of document.querySelectorAll("[data-job]")) {
    if (b.dataset.job === id) b.setAttribute("aria-current", "true"); else b.removeAttribute("aria-current");
  }
  refresh().catch(e => message(e.message));
}
function moveSelection(delta) {
  if (!state.jobs.length) return;
  const at = state.jobs.findIndex(j => j.id === state.selected);
  const next = state.jobs[Math.max(0, Math.min(state.jobs.length - 1, at < 0 ? 0 : at + delta))];
  selectJob(next.id);
}

function renderDetail(job, artifacts) {
  $("job-detail").hidden = false;
  $("selected-job").textContent = job.argv[0];
  const pill = $("selected-status"); pill.dataset.status = job.status; pill.textContent = STATUS[job.status] || job.status;
  const meta = [`Started ${relative(job.started || job.created)}`, duration(job)];
  if (job.returncode !== null) meta.push(`exit code ${job.returncode}`);
  meta.push(`ID ${job.id.slice(0, 8)}`);
  $("selected-meta").textContent = meta.join(" · ");
  showCommand($("selected-command"), job.argv);
  $("cancel").hidden = terminal.has(job.status);
  $("map-load").disabled = job.status !== "succeeded";
  const signature = JSON.stringify(artifacts.files);
  if (signature === state.files) return;
  state.files = signature;
  const count = artifacts.files.length;
  $("file-count").textContent = count ? `${count}${count === artifacts.limit ? "+" : ""}` : "";
  $("artifact-empty").hidden = !!count;
  $("artifact-list").replaceChildren(...artifacts.files.map(file => {
    const [kind, glyph] = fileKind(file.path), parts = file.path.split("/"), name = parts.pop();
    return h("li", {}, h("button", { class: "file-row", type: "button", title: `Download ${file.path}`,
      onclick: () => download(`/api/jobs/${job.id}/artifact?path=${encodeURIComponent(file.path)}`, name).catch(e => message(e.message)) },
      h("span", { class: "file-icon", "data-kind": kind }, icon(glyph)),
      h("span", { class: "file-name" }, h("strong", { text: name }), h("small", { text: parts.join("/") || "workspace" })),
      h("span", { class: "file-size", text: bytes(file.bytes) }), icon("download")));
  }));
}

let refreshing = false, pending = false;
async function refresh() {
  if (!state.token) return;
  if (refreshing) { pending = true; return; }
  refreshing = true;
  try {
    const jobs = (await api("/api/jobs")).jobs;
    const first = !state.known.size;
    for (const job of jobs) {
      const before = state.known.get(job.id);
      if (!first && before && !terminal.has(before) && terminal.has(job.status)) notify(job);
      state.known.set(job.id, job.status);
    }
    state.jobs = jobs; renderJobs();
    const id = state.selected; const generation = state.generation; if (!id) { updateMapEmpty(); return; }
    const job = await api(`/api/jobs/${id}`);
    const result = await api(`/api/jobs/${id}/logs?offset=${state.offset}`);
    const artifacts = await api(`/api/jobs/${id}/artifacts`);
    if (generation !== state.generation) return;
    state.job = job;
    const log = $("logs"), atBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 40;
    state.log = (state.log + result.text).slice(-400000); state.offset = result.next_offset;
    log.textContent = state.log || (job.status === "queued" ? "Waiting for a free worker…" : terminal.has(job.status) ? "This job wrote no output." : "Waiting for command output…");
    if (atBottom || result.text) log.scrollTop = log.scrollHeight;
    renderDetail(job, artifacts);
    if (job.status === "succeeded" && GeoMap.loaded() !== id) GeoMap.load(id, api);
    updateMapEmpty();
  } finally {
    refreshing = false;
    if (pending) { pending = false; refresh().catch(e => message(e.message)); }
  }
}
async function submit(body, { view } = {}) {
  try {
    const job = await api("/api/jobs", { method: "POST", body: JSON.stringify(body) });
    state.known.set(job.id, job.status);
    const name = job.argv[0];
    selectJob(job.id, { keepArea: name === "quicklook" });
    setView(view || (MAP_COMMANDS.has(name) ? "map" : "jobs"));
    return job;
  } catch (error) { message(error.message); return null; }
}
function runExample() {
  GeoMap.showArea(SEMBERIJA.bbox, true);
  return submit({ argv: ["quicklook", "--bbox", ...SEMBERIJA.bbox.map(v => v.toFixed(2)), "--start", SEMBERIJA.start, "--end", SEMBERIJA.end,
    "--out", `outputs/semberija-2023-${stamp()}`] }, { view: "map" });
}
function runDemo() { return submit({ argv: ["demo", "--root", `outputs/web-demo-${stamp()}`, "--quick"] }, { view: "map" }); }

/* ---------- Command forms ---------- */
const WORDS = { bbox: "bounding box", out: "output", crs: "CRS", id: "ID", aoi: "AOI", url: "URL", s1: "S1", s2: "S2", gpkg: "GPKG", ndvi: "NDVI", stac: "STAC", dl: "DL", ml: "ML", obia: "OBIA" };
function humanise(flag) {
  const words = flag.slice(2).split("-").map(w => WORDS[w] || w);
  const text = words.join(" ");
  return text.charAt(0).toUpperCase() + text.slice(1);
}
function field(spec, container) {
  const id = "field-" + spec.flag.slice(2);
  let input;
  if (spec.boolean) {
    input = h("input", { type: "checkbox", class: "switch", role: "switch", id });
    input.checked = !!spec.default;
  } else if (spec.choices) {
    input = h("select", { id });
    if (!spec.required) input.append(new Option(spec.default !== null && spec.default !== undefined ? `Default (${spec.default})` : "Default", ""));
    for (const value of spec.choices) input.append(new Option(String(value), String(value)));
  } else {
    input = h(spec.multiple ? "textarea" : "input", { id, autocomplete: "off", spellcheck: "false", rows: spec.multiple ? "2" : null });
    if (spec.multiple) input.placeholder = "Values separated by spaces or new lines";
    if (typeof spec.default === "number") input.inputMode = "decimal";
  }
  input.dataset.flag = spec.flag; input.required = spec.required && !spec.boolean;
  if (!spec.boolean && spec.default !== null && spec.default !== undefined && !spec.choices) input.value = Array.isArray(spec.default) ? spec.default.join("\n") : String(spec.default);
  const helpText = spec.help || "";
  const label = h("label", { for: id }, humanise(spec.flag), spec.required && !spec.boolean ? h("span", { class: "req", text: "Required" }) : null);
  if (spec.boolean) {
    container.append(h("div", { class: "field switch-field" }, h("div", { class: "field-label" }, label, h("span", { class: "flag", text: spec.flag })), input));
    if (spec.help) input.title = spec.help;
    return;
  }
  const wrap = h("div", { class: `field${spec.multiple ? " wide" : ""}` }, h("div", { class: "field-label" }, label, h("span", { class: "flag", text: spec.flag })), input,
    helpText ? h("p", { class: "help", id: `${id}-help`, text: helpText }) : null);
  if (helpText) input.setAttribute("aria-describedby", `${id}-help`);
  container.append(wrap);
}
function currentCommand() { return state.commands.find(c => c.name === state.command); }
function buildForm() {
  const command = currentCommand(); if (!command) return;
  const [group] = groupOf(command.name);
  $("command-group").textContent = group;
  $("panel-title").textContent = command.name;
  $("command-help").textContent = command.description || "No description.";
  $("required-fields").replaceChildren(); $("optional-fields").replaceChildren(); $("form-error").textContent = "";
  for (const spec of command.fields) field(spec, $(spec.required || command.name === "demo" ? "required-fields" : "optional-fields"));
  if (!$("required-fields").children.length) $("required-fields").append(h("p", { class: "no-required", text: "No required inputs — run it as is, or adjust the advanced options." }));
  const optional = $("optional-fields").children.length;
  $("advanced").hidden = !optional; $("advanced").open = false; $("advanced-count").textContent = optional ? String(optional) : "";
  if (command.name === "demo") { $("field-root").value = `outputs/web-demo-${stamp()}`; $("field-quick").checked = true; }
  if (command.name === "quicklook") {
    $("field-out").placeholder = `outputs/quicklook-${stamp().slice(0, 8)}`;
    $("field-bbox").placeholder = "19.10 44.72 19.16 44.76";
    for (const id of ["field-start", "field-end"]) $(id).type = "date";
    $("required-fields").prepend(h("div", { class: "callout" }, icon("sparkle"),
      h("p", {}, "Prefer a guided start? ", h("button", { class: "link-button", type: "button", text: "New analysis", onclick: openAnalysis }),
        " lets you draw the area on the map and pick the period in three steps.")));
  }
  $("command-preview-wrap").hidden = false; $("run-command").disabled = false;
  updatePreview();
}
function splitValues(value) {
  const lines = value.split(/\r?\n/).map(v => v.trim()).filter(Boolean);
  // a single line of numbers ("19.10 44.72 19.16 44.76") is split on spaces/commas; paths with spaces stay intact
  if (lines.length === 1 && /^[-+\d.eE\s,]+$/.test(lines[0])) return lines[0].split(/[\s,]+/).filter(Boolean);
  return lines;
}
function argumentsFromForm() {
  const command = currentCommand(), args = [command.name];
  for (const spec of command.fields) {
    const input = $("field-" + spec.flag.slice(2));
    if (spec.boolean) { if (input.checked) args.push(spec.flag); }
    else if (input.value.trim()) {
      const values = spec.multiple ? splitValues(input.value) : [input.value];
      const fallback = spec.default === null || spec.default === undefined ? null : [].concat(spec.default).map(String);
      // an unchanged default is implied: keep the command line minimal, exactly as one would type it
      if (!spec.required && fallback && values.join("\u0000") === fallback.join("\u0000")) continue;
      args.push(spec.flag, ...values);
    }
  }
  return args;
}
function updatePreview() { if (currentCommand()) showCommand($("command-preview"), argumentsFromForm()); }
function renderCommandList() {
  const q = $("command-filter").value.trim().toLowerCase(), list = $("command-list");
  list.replaceChildren();
  const groups = new Map();
  for (const c of state.commands) {
    if (q && !`${c.name} ${c.description || ""}`.toLowerCase().includes(q)) continue;
    const [title] = groupOf(c.name);
    if (!groups.has(title)) groups.set(title, []);
    groups.get(title).push(c);
  }
  const order = [...GROUPS.map(g => g[0]), "Other"];
  for (const title of order) {
    const items = groups.get(title); if (!items) continue;
    list.append(h("div", { class: "command-group-title", text: title, role: "presentation" }));
    for (const c of items) list.append(h("button", { class: "command-item", type: "button", role: "option", "aria-selected": String(c.name === state.command), "data-command": c.name,
      onclick: () => selectCommand(c.name) }, h("strong", { text: c.name }), h("span", { text: c.description || "" })));
  }
  if (!list.children.length) list.append(h("p", { class: "palette-empty", text: "No command matches." }));
}
function selectCommand(name, { show = true } = {}) {
  state.command = name;
  for (const b of $("command-list").querySelectorAll("[data-command]")) b.setAttribute("aria-selected", String(b.dataset.command === name));
  buildForm();
  if (show) { setView("commands"); $("command-list").querySelector(`[data-command="${CSS.escape(name)}"]`)?.scrollIntoView({ block: "nearest" }); }
}

/* ---------- Command palette ---------- */
const palette = { items: [], active: 0 };
function paletteActions() {
  const dark = effectiveTheme() === "dark";
  return [
    { title: "New analysis", desc: "Area, period and a Sentinel-2 quicklook", icon: "plus", kbd: "N", run: openAnalysis },
    { title: "Try the Semberija example", desc: "Quicklook of 19.10–19.16 °E, 44.72–44.76 °N, June–August 2023", icon: "sparkle", run: runExample },
    { title: "Run the synthetic demo", desc: "Software check with crop classes, uncertainty and parcels", icon: "cube", run: runDemo },
    { title: "Go to Map", icon: "map", kbd: "1", run: () => setView("map") },
    { title: "Go to Commands", icon: "command", kbd: "2", run: () => setView("commands") },
    { title: "Go to Jobs", icon: "jobs", kbd: "3", run: () => setView("jobs") },
    { title: "Go to Terminal", desc: "Type engine commands as in your shell", icon: "terminal", kbd: "4", run: () => setView("terminal") },
    { title: "Go to Python & API", icon: "code", kbd: "5", run: () => setView("api") },
    { title: dark ? "Use light appearance" : "Use dark appearance", icon: dark ? "sun" : "moon", run: toggleTheme },
    { title: "Expand map", icon: "expand", kbd: "F", run: () => toggleExpand(true) },
    { title: "Keyboard shortcuts", icon: "keyboard", kbd: "?", run: () => $("shortcuts").showModal() },
  ];
}
function highlight(text, q) {
  const at = q ? text.toLowerCase().indexOf(q) : -1;
  if (at < 0) return [document.createTextNode(text)];
  return [document.createTextNode(text.slice(0, at)), h("mark", { text: text.slice(at, at + q.length) }), document.createTextNode(text.slice(at + q.length))];
}
function renderPalette() {
  const q = $("palette-input").value.trim().toLowerCase(), list = $("palette-list");
  const score = (item) => {
    const title = item.title.toLowerCase(), hay = `${title} ${(item.desc || "").toLowerCase()}`;
    if (!q) return 3;
    if (!q.split(/\s+/).every(t => hay.includes(t))) return -1;
    return title.startsWith(q) ? 0 : title.includes(q) ? 1 : 2;
  };
  const sections = [["Actions", paletteActions()]];
  const byGroup = new Map();
  for (const c of state.commands) {
    const [title, glyph] = groupOf(c.name);
    if (!byGroup.has(title)) byGroup.set(title, []);
    byGroup.get(title).push({ title: c.name, desc: c.description, icon: glyph, run: () => selectCommand(c.name) });
  }
  for (const title of [...GROUPS.map(g => g[0]), "Other"]) if (byGroup.has(title)) sections.push([`Commands · ${title}`, byGroup.get(title)]);
  palette.items = []; list.replaceChildren();
  const ranked = [];
  for (const [title, items] of sections) {
    const matches = items.map(i => [score(i), i]).filter(([s]) => s >= 0);
    if (q) ranked.push(...matches.map(([s, i]) => [s, title, i]));
    else if (matches.length) { list.append(h("li", { class: "palette-group", role: "presentation", text: title })); for (const [, i] of matches) addPaletteItem(i, q); }
  }
  if (q) {
    ranked.sort((a, b) => a[0] - b[0]);
    for (const [, , i] of ranked) addPaletteItem(i, q);
  }
  if (!palette.items.length) list.append(h("li", { class: "palette-empty", role: "presentation", text: `No results for “${$("palette-input").value.trim()}”` }));
  palette.active = 0; markPalette();
}
function addPaletteItem(item, q) {
  const index = palette.items.length;
  palette.items.push(item);
  $("palette-list").append(h("li", { class: "palette-item", role: "option", id: `palette-${index}`, "aria-selected": "false",
    onmousemove: () => { if (palette.active !== index) { palette.active = index; markPalette(false); } },
    onclick: () => runPalette(index) },
    h("span", { class: "p-icon" }, icon(item.icon)),
    h("span", { class: "p-text" }, h("strong", {}, ...highlight(item.title, q)), item.desc ? h("span", { class: "p-desc", text: item.desc }) : null),
    item.kbd ? h("kbd", { text: item.kbd }) : null));
}
function markPalette(scroll = true) {
  palette.items.forEach((_, k) => $(`palette-${k}`).setAttribute("aria-selected", String(k === palette.active)));
  const active = $(`palette-${palette.active}`);
  if (active) { $("palette-input").setAttribute("aria-activedescendant", active.id); if (scroll) active.scrollIntoView({ block: "nearest" }); }
  else $("palette-input").removeAttribute("aria-activedescendant");
}
function runPalette(index) { const item = palette.items[index]; if (!item) return; $("palette").close(); item.run(); }
function openPalette() {
  if (!state.token) return;
  if ($("palette").open) { $("palette").close(); return; }
  for (const d of document.querySelectorAll("dialog[open]")) d.close();
  $("palette-input").value = ""; renderPalette(); $("palette").showModal(); $("palette-input").focus();
}

/* ---------- New analysis sheet ---------- */
const bboxIds = ["bbox-west", "bbox-south", "bbox-east", "bbox-north"];
function readBbox() {
  const values = bboxIds.map(id => Number($(id).value.trim().replace(",", ".")));
  const blank = bboxIds.some(id => !$(id).value.trim());
  for (const id of bboxIds) $(id).removeAttribute("aria-invalid");
  if (blank) return { error: "Enter all four coordinates, or draw the area on the map." };
  const bad = values.map((v, k) => !Number.isFinite(v) || Math.abs(v) > (k % 2 ? 90 : 180));
  if (bad.some(Boolean)) { bad.forEach((b, k) => b && $(bboxIds[k]).setAttribute("aria-invalid", "true")); return { error: "Longitudes are −180…180 and latitudes −90…90 degrees." }; }
  const [w, s, e, n] = values;
  if (w >= e) { $("bbox-east").setAttribute("aria-invalid", "true"); return { error: "East must be greater than west." }; }
  if (s >= n) { $("bbox-north").setAttribute("aria-invalid", "true"); return { error: "North must be greater than south." }; }
  return { bbox: values };
}
function areaSize([w, s, e, n]) {
  const kx = (e - w) * 111.32 * Math.cos(((s + n) / 2) * Math.PI / 180), ky = (n - s) * 110.57;
  const f = (v) => (v < 10 ? v.toFixed(1) : Math.round(v));
  return { text: `≈ ${f(kx)} × ${f(ky)} km`, km2: kx * ky };
}
function updateBboxInfo() {
  const { bbox } = readBbox();
  if (!bbox) { $("bbox-info").textContent = ""; return; }
  const size = areaSize(bbox);
  $("bbox-info").textContent = size.text + (size.km2 > 2500 ? " · large areas take noticeably longer" : "");
  GeoMap.showArea(bbox);
}
const iso = (d) => `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
function periodPresets() {
  const now = new Date(), last = new Date(now.getFullYear(), now.getMonth(), 0), first = new Date(last.getFullYear(), last.getMonth() - 2, 1);
  const y = now.getMonth() >= 9 ? now.getFullYear() : now.getFullYear() - 1;
  return [
    ["Jun–Aug 2023", "2023-06-01", "2023-08-31"],
    [`Growing season ${y}`, `${y}-04-01`, `${y}-09-30`],
    [`Summer ${y}`, `${y}-06-01`, `${y}-08-31`],
    ["Last 3 full months", iso(first), iso(last)],
  ];
}
function readPeriod() {
  const start = $("period-start").value, end = $("period-end").value;
  if (!start || !end) return { error: "Choose a start and an end date." };
  if (start >= end) return { error: "The end date must be after the start date." };
  return { start, end };
}
function updatePeriodInfo() {
  for (const b of $("period-presets").children) b.setAttribute("aria-pressed", String(b.dataset.start === $("period-start").value && b.dataset.end === $("period-end").value));
  const { start, end } = readPeriod();
  if (!start) { $("period-info").textContent = ""; return; }
  const a = new Date(start), b = new Date(end);
  const months = (b.getFullYear() - a.getFullYear()) * 12 + b.getMonth() - a.getMonth() + 1;
  const days = Math.round((b - a) / 86400000) + 1;
  let text = `${days} days · up to ${months} monthly composite${months === 1 ? "" : "s"}`;
  if (start < "2017-04-01") text += " · Sentinel-2 L2A coverage is sparse before 2017";
  if (months > 12) text += " · long periods take longer";
  $("period-info").textContent = text;
}
function analysisArgv() {
  const { bbox } = readBbox(), { start, end } = readPeriod();
  const raw = bboxIds.map(id => $(id).value.trim().replace(",", "."));
  return ["quicklook", "--bbox", ...(bbox ? raw : []), "--start", start, "--end", end, "--out", $("analysis-out").value.trim()];
}
function showStep(step) {
  state.step = step;
  for (const s of document.querySelectorAll("#analysis-form [data-step]")) s.hidden = Number(s.dataset.step) !== step;
  for (const dot of document.querySelectorAll("[data-step-dot]")) {
    const n = Number(dot.dataset.stepDot);
    dot.classList.toggle("done", n < step);
    if (n === step) dot.setAttribute("aria-current", "step"); else dot.removeAttribute("aria-current");
  }
  $("analysis-back").hidden = step === 1;
  $("analysis-next").replaceChildren(...(step === 3 ? [icon("play"), document.createTextNode("Run quicklook")] : [document.createTextNode("Continue")]));
  $("analysis-error").textContent = "";
  if (step === 3) {
    const { bbox } = readBbox(), { start, end } = readPeriod();
    const raw = bboxIds.map(id => $(id).value.trim());
    $("summary-area").textContent = `${raw[0]}–${raw[2]} °E · ${raw[1]}–${raw[3]} °N (${areaSize(bbox).text.slice(2)})`;
    const fmt = (d) => new Date(d).toLocaleDateString("en", { day: "numeric", month: "short", year: "numeric" });
    $("summary-period").textContent = `${fmt(start)} – ${fmt(end)}`;
    if (!$("analysis-out").value.trim()) $("analysis-out").value = `outputs/quicklook-${stamp()}`;
    showCommand($("analysis-preview"), analysisArgv());
  }
  const focus = { 1: "bbox-west", 2: "period-start", 3: "analysis-out" }[step];
  if ($("analysis-sheet").open) setTimeout(() => $(focus).focus(), 30);
}
function openAnalysis() {
  if (!state.token) return;
  for (const d of document.querySelectorAll("dialog[open]")) d.close();
  if (!$("period-start").value) { const [, s, e] = periodPresets()[1]; $("period-start").value = s; $("period-end").value = e; }
  $("analysis-out").value = "";
  $("analysis-sheet").showModal(); showStep(1); updateBboxInfo(); updatePeriodInfo();
}
async function pickOnMap() {
  $("analysis-sheet").close();
  setView("map"); state.picking = true; updateMapEmpty();
  $("pick-basemap").hidden = $("basemap-select").value !== "none";
  const bbox = await GeoMap.pickArea();
  state.picking = false; updateMapEmpty();
  if (bbox) bboxIds.forEach((id, k) => { $(id).value = String(bbox[k]); });
  $("analysis-sheet").showModal(); showStep(1); updateBboxInfo();
}

/* ---------- Connection ---------- */
async function connect(event) {
  event.preventDefault();
  const button = event.submitter || $("connect-form").querySelector("button");
  state.token = $("token").value.trim(); $("auth-error").textContent = ""; button.disabled = true; button.textContent = "Connecting…";
  try {
    const health = await api("/api/health");
    state.commands = (await api("/api/commands")).commands;
    $("version").textContent = `Engine ${health.version}`; $("workspace-path").textContent = health.workspace; $("workspace-path").title = health.workspace;
    $("connection-dot").classList.add("online"); $("connection-label").textContent = "Engine connected";
    $("auth-panel").hidden = true; $("token").value = ""; $("disconnect").hidden = false;
    for (const id of ["new-analysis", "open-palette", "new-compact", "refresh", "api-schema", "empty-new", "example-quicklook", "example-demo", "run-command"]) $(id).disabled = false;
    $("jobs").replaceChildren(...[1, 2, 3].map(() => h("li", { class: "skeleton-row" })));
    renderCommandList(); selectCommand(state.commands.some(c => c.name === "quicklook") ? "quicklook" : state.commands[0]?.name, { show: false });
    await refresh();
    // --- terminal bridge (terminal.js) ---
    document.dispatchEvent(new CustomEvent("geocrop:connected", { detail: { health, commands: state.commands } }));
    if (!state.selected) {
      const recent = state.jobs.find(j => j.status === "succeeded" && MAP_COMMANDS.has(j.argv[0]));
      if (recent) selectJob(recent.id);
    }
  } catch (error) {
    state.token = ""; $("auth-error").textContent = error.message.includes("Bearer") ? "That token was not accepted. Check GEOCROP_WEB_TOKEN of the running server." : error.message;
    $("token").focus(); $("token").select();
  } finally { button.disabled = false; button.textContent = "Connect"; }
}

/* ---------- Terminal bridge (terminal.js): authenticated API helper and job selection ---------- */
window.GeoCropApp = { api, selectJob: (id) => selectJob(id) };

/* ---------- Wiring ---------- */
function bind() {
  for (const k of document.querySelectorAll("[data-mod-key]")) k.textContent = isMac ? "⌘K" : "Ctrl K";
  syncThemeLabels();
  matchMedia("(prefers-color-scheme: dark)").addEventListener?.("change", syncThemeLabels);
  $("connect-form").addEventListener("submit", connect);
  $("disconnect").addEventListener("click", () => { state.token = ""; location.reload(); });
  for (const b of document.querySelectorAll("[data-view]")) b.addEventListener("click", () => setView(b.dataset.view));
  for (const id of ["theme-toggle", "theme-compact"]) $(id).addEventListener("click", toggleTheme);
  for (const id of ["open-palette", "palette-compact"]) $(id).addEventListener("click", openPalette);
  for (const id of ["new-analysis", "new-compact", "empty-new"]) $(id).addEventListener("click", openAnalysis);
  $("example-quicklook").addEventListener("click", runExample);
  $("example-demo").addEventListener("click", runDemo);
  $("shortcuts-open").addEventListener("click", () => $("shortcuts").showModal());
  $("shortcuts-close").addEventListener("click", () => $("shortcuts").close());
  $("refresh").addEventListener("click", () => refresh().catch(e => message(e.message)));
  $("cancel").addEventListener("click", async () => { try { await api(`/api/jobs/${state.selected}/cancel`, { method: "POST" }); await refresh(); } catch (e) { message(e.message); } });
  $("map-load").addEventListener("click", () => { if (state.selected) { setView("map"); GeoMap.load(state.selected, api); } });
  $("map-expand").addEventListener("click", () => toggleExpand());
  $("copy-command").addEventListener("click", () => copy($("selected-command").textContent));
  $("copy-preview").addEventListener("click", () => copy($("command-preview").textContent));
  $("api-schema").addEventListener("click", () => download("/api/openapi.json", "geocrop-openapi.json").catch(e => message(e.message)));
  $("pick-basemap").addEventListener("click", () => { GeoMap.setBasemap("s2-2024"); $("pick-basemap").hidden = true; });
  document.addEventListener("geomap:state", (e) => { state.mapState = e.detail.state; updateMapEmpty(); });

  // command forms
  $("command-filter").addEventListener("input", renderCommandList);
  $("command-filter").addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); $("command-list").querySelector("[data-command]")?.click(); }
  });
  $("command-form").addEventListener("input", updatePreview);
  $("command-form").addEventListener("change", updatePreview);
  $("command-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const missing = [...$("command-form").querySelectorAll("[required]")].filter(i => !i.value.trim());
    for (const i of $("command-form").querySelectorAll("[aria-invalid]")) i.removeAttribute("aria-invalid");
    if (missing.length) {
      missing.forEach(i => i.setAttribute("aria-invalid", "true"));
      $("form-error").textContent = `Fill in ${missing.map(i => humanise(i.dataset.flag).toLowerCase()).join(", ")}.`;
      missing[0].focus(); return;
    }
    $("form-error").textContent = "";
    submit({ argv: argumentsFromForm() }).then(job => {
      if (job && state.command === "demo" && /^outputs\/web-demo-/.test($("field-root").value)) $("field-root").value = `outputs/web-demo-${stamp()}`;
    });
  });

  // palette
  $("palette-input").addEventListener("input", renderPalette);
  $("palette-input").addEventListener("keydown", (e) => {
    const n = palette.items.length;
    if (e.key === "ArrowDown" || e.key === "ArrowUp") { e.preventDefault(); if (n) { palette.active = (palette.active + (e.key === "ArrowDown" ? 1 : -1) + n) % n; markPalette(); } }
    else if (e.key === "Home" || e.key === "End") { if (n) { e.preventDefault(); palette.active = e.key === "Home" ? 0 : n - 1; markPalette(); } }
    else if (e.key === "Enter") { e.preventDefault(); runPalette(palette.active); }
  });
  $("palette").addEventListener("click", (e) => { if (e.target === $("palette")) $("palette").close(); });

  // analysis sheet
  for (const id of bboxIds) $(id).addEventListener("input", () => { $("analysis-error").textContent = ""; updateBboxInfo(); });
  for (const id of ["period-start", "period-end"]) $(id).addEventListener("input", () => { $("analysis-error").textContent = ""; updatePeriodInfo(); });
  $("period-presets").replaceChildren(...periodPresets().map(([label, s, e]) => h("button", { class: "chip chip-button", type: "button", "data-start": s, "data-end": e, "aria-pressed": "false", text: label,
    onclick: () => { $("period-start").value = s; $("period-end").value = e; updatePeriodInfo(); } })));
  $("preset-semberija").addEventListener("click", () => {
    bboxIds.forEach((id, k) => { $(id).value = SEMBERIJA.bbox[k].toFixed(2); });
    updateBboxInfo(); GeoMap.showArea(SEMBERIJA.bbox, true);
  });
  $("pick-area").addEventListener("click", pickOnMap);
  $("analysis-close").addEventListener("click", () => $("analysis-sheet").close());
  $("analysis-back").addEventListener("click", () => showStep(state.step - 1));
  $("analysis-out").addEventListener("input", () => showCommand($("analysis-preview"), analysisArgv()));
  $("analysis-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const check = state.step === 1 ? readBbox() : state.step === 2 ? readPeriod() : null;
    if (check && check.error) { $("analysis-error").textContent = check.error; return; }
    if (state.step < 3) { if (state.step === 1) GeoMap.showArea(check.bbox, true); showStep(state.step + 1); return; }
    const out = $("analysis-out").value.trim();
    if (!out) { $("analysis-error").textContent = "Choose an output folder."; return; }
    $("analysis-next").disabled = true;
    const job = await submit({ argv: analysisArgv() }, { view: "map" });
    $("analysis-next").disabled = false;
    if (job) $("analysis-sheet").close();
  });

  // keyboard
  document.addEventListener("keydown", (e) => {
    const inTerminal = e.target.closest?.("#terminal-root");
    if ((e.metaKey || e.ctrlKey) && !e.altKey && e.key.toLowerCase() === "k" && !inTerminal) { e.preventDefault(); openPalette(); return; }
    if (!state.token || document.querySelector("dialog[open]") || state.picking) return;
    if (e.metaKey || e.ctrlKey || e.altKey || inTerminal) return;
    const t = e.target;
    if (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName)) return;
    const keys = {
      n: openAnalysis, N: openAnalysis, "?": () => $("shortcuts").showModal(),
      1: () => setView("map"), 2: () => setView("commands"), 3: () => setView("jobs"), 4: () => setView("terminal"), 5: () => setView("api"),
      f: () => toggleExpand(), "[": () => GeoMap.step(-1), "]": () => GeoMap.step(1),
      j: () => moveSelection(1), k: () => moveSelection(-1),
      Escape: () => { if ($("map-shell").classList.contains("is-expanded")) toggleExpand(false); },
    };
    if (keys[e.key]) { e.preventDefault(); keys[e.key](); }
  });

  setView("map"); updateMapEmpty();
  $("token").focus();
}
bind();
let lastPollError = "";
async function poll() {
  try { await refresh(); lastPollError = ""; }
  catch (e) { if (e.message !== lastPollError) message(e.message); lastPollError = e.message; }
  setTimeout(poll, 2000);
}
poll();
