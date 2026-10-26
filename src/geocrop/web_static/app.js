"use strict";
const $ = (id) => document.getElementById(id);
const state = { token: "", commands: [], jobs: [], selected: null, offset: 0, log: "", generation: 0 };
const statuses = { queued: "QUEUED", running: "RUNNING", succeeded: "SUCCEEDED", failed: "FAILED", cancelled: "CANCELLED", interrupted: "INTERRUPTED" };
const terminal = new Set(["succeeded", "failed", "cancelled", "interrupted"]);
function message(text) { $("message").textContent = text; $("message").hidden = !text; }
async function api(path, options = {}) {
  const response = await fetch(path, { ...options, headers: { "Authorization": `Bearer ${state.token}`, "Content-Type": "application/json", ...options.headers } });
  if (!response.ok) { let error; try { error = await response.json(); } catch { error = {}; } throw new Error(typeof error.detail === "string" ? error.detail : `HTTP ${response.status}`); }
  return response.json();
}
function field(spec, container) {
  const wrapper = document.createElement("div"); wrapper.className = "field";
  const id = "field-" + spec.flag.slice(2); const label = document.createElement("label"); label.htmlFor = id;
  label.textContent = spec.flag.slice(2).replaceAll("-", " ") + (spec.required ? " *" : "");
  let input;
  if (spec.boolean) { input = document.createElement("input"); input.type = "checkbox"; input.checked = !!spec.default; wrapper.classList.add("checkbox"); }
  else if (spec.choices) { input = document.createElement("select"); if (!spec.required) input.append(new Option("Default", "")); for (const value of spec.choices) input.append(new Option(String(value), String(value))); }
  else input = document.createElement(spec.multiple ? "textarea" : "input");
  input.id = id; input.dataset.flag = spec.flag; input.required = spec.required && !spec.boolean;
  if (!spec.boolean && spec.default !== null && spec.default !== undefined) input.value = Array.isArray(spec.default) ? spec.default.join("\n") : String(spec.default);
  if (spec.multiple) input.placeholder = "One value per line";
  if (spec.boolean) wrapper.append(input, label); else wrapper.append(label, input);
  if (spec.help && !spec.boolean) { const help = document.createElement("p"); help.className = "help"; help.textContent = spec.help; wrapper.append(help); }
  container.append(wrapper);
}
function buildForm() {
  const command = state.commands.find(c => c.name === $("command-select").value); if (!command) return;
  $("command-help").textContent = command.description; $("required-fields").replaceChildren(); $("optional-fields").replaceChildren();
  for (const spec of command.fields) field(spec, $(spec.required || command.name === "demo" ? "required-fields" : "optional-fields"));
  $("advanced").hidden = !$("optional-fields").children.length;
  if (command.name === "demo") { $("field-root").value = `outputs/web-demo-${Date.now()}`; $("field-quick").checked = true; }
}
function argumentsFromForm() {
  const command = state.commands.find(c => c.name === $("command-select").value), args = [command.name];
  for (const spec of command.fields) {
    const input = $("field-" + spec.flag.slice(2));
    if (spec.boolean) { if (input.checked) args.push(spec.flag); }
    else if (input.value.trim()) { args.push(spec.flag); args.push(...(spec.multiple ? input.value.split(/\r?\n/).map(v => v.trim()).filter(Boolean) : [input.value])); }
  }
  return args;
}
async function submit(body) {
  message(""); try { const job = await api("/api/jobs", { method: "POST", body: JSON.stringify(body) }); state.selected = job.id; state.offset = 0; state.log = ""; state.generation++; $("cancel").hidden = true;
    if (body.argv?.[0] === "demo" && $("field-root") && /^outputs\/web-demo-\d+$/.test($("field-root").value)) $("field-root").value = `outputs/web-demo-${Date.now()}`;
    await refresh(); }
  catch (error) { message(error.message); }
}
function badge(status) { const span = document.createElement("span"); span.className = "badge " + status; span.textContent = statuses[status] || status; return span; }
function renderJobs() {
  $("total").textContent = state.jobs.length;
  $("running").textContent = state.jobs.filter(j => !terminal.has(j.status)).length;
  $("succeeded").textContent = state.jobs.filter(j => j.status === "succeeded").length;
  $("failed").textContent = state.jobs.filter(j => ["failed", "interrupted"].includes(j.status)).length;
  $("jobs").replaceChildren();
  if (!state.jobs.length) { const row = document.createElement("tr"), cell = document.createElement("td"); cell.colSpan = 4; cell.className = "empty"; cell.textContent = "No jobs yet. Run your first command above."; row.append(cell); $("jobs").append(row); }
  for (const job of state.jobs) {
    const row = document.createElement("tr"); row.dataset.job = job.id; row.tabIndex = 0;
    const command = document.createElement("td"); command.className = "job-command"; command.textContent = "geocrop " + job.argv.join(" "); command.title = command.textContent;
    const status = document.createElement("td"); status.append(badge(job.status));
    const created = document.createElement("td"); created.textContent = new Date(job.created).toLocaleString();
    const code = document.createElement("td"); code.textContent = job.returncode === null ? "—" : job.returncode;
    const select = () => { state.selected = job.id; state.offset = 0; state.log = ""; state.generation++; refresh().catch(e => message(e.message)); };
    row.addEventListener("click", select); row.addEventListener("keydown", e => { if (e.key === "Enter") select(); });
    row.append(command, status, created, code); $("jobs").append(row);
  }
}
async function download(path, name) {
  const response = await fetch(path, { headers: { Authorization: `Bearer ${state.token}` } });
  if (!response.ok) throw new Error(`Download failed: HTTP ${response.status}`);
  const url = URL.createObjectURL(await response.blob()), link = document.createElement("a");
  link.href = url; link.download = name; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
}
let refreshing = false;
async function refresh() {
  if (!state.token || refreshing) return; refreshing = true;
  try {
    state.jobs = (await api("/api/jobs")).jobs; renderJobs();
    const id = state.selected; const generation = state.generation; if (!id) return;
    const job = await api(`/api/jobs/${id}`);
    const result = await api(`/api/jobs/${id}/logs?offset=${state.offset}`);
    const artifacts = await api(`/api/jobs/${id}/artifacts`);
    if (generation !== state.generation) return;
    state.log = (state.log + result.text).slice(-400000); state.offset = result.next_offset;
    $("logs").textContent = state.log || (job.status === "queued" ? "Waiting for a free worker…" : "Waiting for command output…");
    $("logs").scrollTop = $("logs").scrollHeight;
    $("selected-job").textContent = `${job.argv[0]} · ${id.slice(0, 8)}`;
    $("selected-status").className = "badge " + job.status; $("selected-status").textContent = statuses[job.status];
    $("cancel").hidden = terminal.has(job.status); $("artifact-list").replaceChildren();
    if (job.status === "succeeded" && typeof GeoMap !== "undefined" && GeoMap.loaded() !== id) GeoMap.load(id, api);
    $("file-count").textContent = artifacts.files.length ? `(${artifacts.files.length}${artifacts.files.length === artifacts.limit ? "+" : ""})` : "";
    $("artifact-empty").hidden = !!artifacts.files.length;
    for (const file of artifacts.files) { const li = document.createElement("li"), button = document.createElement("button"), size = document.createElement("span");
      button.className = "file-button"; button.textContent = file.path; size.className = "file-size"; size.textContent = `${(file.bytes / 1024).toFixed(1)} KB ↓`;
      button.addEventListener("click", () => download(`/api/jobs/${id}/artifact?path=${encodeURIComponent(file.path)}`, file.path.split("/").pop()).catch(e => message(e.message)));
      li.append(button, size); $("artifact-list").append(li); }
  } finally { refreshing = false; }
}
$("connect-form").addEventListener("submit", async event => {
  event.preventDefault(); state.token = $("token").value;
  try { const health = await api("/api/health"); state.commands = (await api("/api/commands")).commands;
    $("command-select").replaceChildren(...state.commands.map(c => new Option(c.name, c.name)));
    $("command-select").value = "demo"; $("command-select").disabled = false; buildForm();
    $("version").textContent = `ENGINE ${health.version}`; $("workspace-path").textContent = health.workspace;
    $("connection-dot").classList.add("online"); $("connection-label").textContent = "Engine connected";
    $("auth-panel").hidden = true; $("token").value = ""; $("disconnect").hidden = false;
    for (const id of ["run-command", "run-console", "refresh", "api-schema", "map-load"]) $(id).disabled = false;
    message(""); await refresh();
  } catch (error) { state.token = ""; message(error.message); }
});
$("disconnect").addEventListener("click", () => { state.token = ""; location.reload(); });
$("command-select").addEventListener("change", buildForm);
$("command-form").addEventListener("submit", e => { e.preventDefault(); submit({ argv: argumentsFromForm() }); });
$("console-form").addEventListener("submit", e => { e.preventDefault(); submit({ command: $("console-input").value }); });
$("refresh").addEventListener("click", () => refresh().catch(e => message(e.message)));
$("cancel").addEventListener("click", async () => { try { await api(`/api/jobs/${state.selected}/cancel`, { method: "POST" }); await refresh(); } catch (e) { message(e.message); } });
$("map-load").addEventListener("click", () => { if (state.selected) GeoMap.load(state.selected, api); else message("Select a job first."); });
$("api-schema").addEventListener("click", () => download("/api/openapi.json", "geocrop-openapi.json").catch(e => message(e.message)));
for (const button of document.querySelectorAll("[data-tab]")) button.addEventListener("click", () => {
  for (const item of document.querySelectorAll("[data-tab]")) item.classList.toggle("active", item === button);
  for (const name of ["commands", "console", "notebook"]) $(name + "-tab").hidden = name !== button.dataset.tab;
  $("panel-title").textContent = { commands: "Engine commands", console: "CLI console", notebook: "Jupyter & API" }[button.dataset.tab];
});
async function poll() { try { await refresh(); } catch (e) { message(e.message); } setTimeout(poll, 2000); }
poll();
