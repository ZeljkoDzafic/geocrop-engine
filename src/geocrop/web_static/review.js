"use strict";
/* Review: walk the parcels of a finished job, see the evidence and record analyst decisions (gap G7). */
const GeoCropReview = (() => {
  const $ = (id) => document.getElementById(id);
  const App = () => window.GeoCropApp;
  const R = {
    job: null, summary: null, items: [], total: 0, offset: 0, current: null, detail: null, generation: 0,
    chipsKind: "true", busy: false, evidence: "", chart: null,
    filters: { light: "", state: "pending", cls: "", reason: "", unc: "", decision: "" }, sort: "priority", q: "",
  };
  const PAGE = 100;
  const LIGHT_LABEL = { red: "Red", yellow: "Yellow", green: "Green", grey: "Grey" };
  const DECISION = {
    agree: { label: "Agreed", icon: "thumbup" }, disagree: { label: "Disagreed", icon: "swap" }, unsure: { label: "Can't tell", icon: "question" },
  };
  const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const store = {
    get(key) { try { return localStorage.getItem(key) || ""; } catch { return ""; } },
    set(key, value) { try { localStorage.setItem(key, value); } catch { /* per-browser convenience only */ } },
  };
  const human = (name) => String(name ?? "—").replaceAll("_", " ");
  const pct = (v) => (v === null || v === undefined ? "—" : `${Math.round(v * 100)}%`);
  const svgEl = (tag, attrs = {}) => {
    const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (const [k, v] of Object.entries(attrs)) if (v !== undefined && v !== null) node.setAttribute(k, String(v));
    return node;
  };
  function fmtDate(text, short = false) {
    const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(text || "");
    if (!m) return String(text || "").replace(/^window\s+/, "");
    return short ? `${Number(m[3])} ${MONTHS[m[2] - 1]}` : `${Number(m[3])} ${MONTHS[m[2] - 1]} ${m[1]}`;
  }
  const base = () => `/api/jobs/${R.job.id}/review`;
  const layerParam = () => (R.summary && R.summary.layer ? `layer=${encodeURIComponent(R.summary.layer)}` : "");

  /* ---------- empty states ---------- */
  function empty(title, text, { demo = true } = {}) {
    $("review-main").hidden = true; $("review-empty").hidden = false;
    $("review-empty-title").textContent = title; $("review-empty-text").textContent = text;
    $("review-demo").hidden = !demo;
    if (window.GeoMap) GeoMap.focusParcel(null);
  }

  /* ---------- job / summary ---------- */
  async function sync(force = false) {
    if (!App() || !App().token() || App().view() !== "review") return;
    const job = App().current();
    if (!job) {
      R.job = null;
      empty("Review parcels of a finished job", "Select a finished parcel job in the sidebar — the synthetic demo, a run with parcels, parcels or traffic-light — to walk its parcels one by one, see the evidence and record agree, disagree or can't tell.");
      return;
    }
    if (job.status !== "succeeded") {
      R.job = null;
      empty(job.status === "running" || job.status === "queued" ? `${job.argv[0]} is still running` : `${job.argv[0]} did not finish`,
            "Review opens when the job has finished successfully.", { demo: false });
      return;
    }
    if (!force && R.job && R.job.id === job.id && R.summary) return;
    R.job = job; R.summary = null; R.current = null; R.items = []; R.detail = null;
    const generation = ++R.generation;
    try {
      const summary = await App().api(`${base()}`);
      if (generation !== R.generation) return;
      if (!summary.available) {
        empty("Nothing to review in this job", `${summary.reason}${summary.imagery ? ` This job has ${summary.imagery} image date${summary.imagery === 1 ? "" : "s"} for the Map view.` : ""}`);
        return;
      }
      R.summary = summary;
      $("review-empty").hidden = true; $("review-main").hidden = false;
      renderBar(); await loadQueue(true);
    } catch (error) {
      if (generation === R.generation) empty("Review unavailable", error.message, { demo: false });
    }
  }
  async function refreshSummary() {
    try { R.summary = { ...R.summary, ...(await App().api(`${base()}?${layerParam()}`)) }; renderProgress(); } catch { /* keep the old counts */ }
  }

  function renderProgress() {
    const p = R.summary.progress;
    $("review-count").textContent = String(p.reviewed);
    $("review-total").textContent = `of ${p.total} reviewed`;
    for (const seg of $("review-meter").children) seg.style.width = `${p.total ? (p[seg.dataset.k] / p.total) * 100 : 0}%`;
    $("review-meter").setAttribute("aria-label", `${p.reviewed} of ${p.total} parcels reviewed: ${p.agree} agreed, ${p.disagree} disagreed, ${p.unsure} can't tell`);
    const { h, icon } = App();
    $("review-tally").replaceChildren(...["agree", "disagree", "unsure"].map(k => h("span", { class: `tally tally-${k}`, title: DECISION[k].label }, icon(DECISION[k].icon), String(p[k]))));
  }

  function fillSelect(select, entries, first) {
    const keep = select.value;
    select.replaceChildren(new Option(first, ""), ...entries.map(([value, label]) => new Option(label, value)));
    select.value = entries.some(([v]) => v === keep) ? keep : "";
  }
  function renderBar() {
    const s = R.summary, f = s.facets;
    renderProgress();
    const lights = Object.keys(LIGHT_LABEL).filter(k => f.traffic_light[k]);
    const seg = $("filter-light"); seg.replaceChildren();
    const { h } = App();
    for (const [value, label] of [["", `All ${s.parcels}`], ...lights.map(k => [k, `${LIGHT_LABEL[k]} ${f.traffic_light[k]}`])]) {
      seg.append(h("button", { type: "button", role: "radio", "data-value": value, "aria-checked": String(R.filters.light === value),
        tabindex: R.filters.light === value ? "0" : "-1", onclick: () => { R.filters.light = value; renderBar(); loadQueue(true); } },
        value ? h("span", { class: `light-dot tl-${value}`, "aria-hidden": "true" }) : null, label));
    }
    fillSelect($("filter-class"), Object.entries(f.declared_class).map(([k, n]) => [k, `Declared ${human(k)} · ${n}`]), "Any declared class");
    fillSelect($("filter-reason"), Object.entries(f.reason).map(([k, n]) => [k, `${human(k)} · ${n}`]), "Any reason");
    $("filter-reason").hidden = !Object.keys(f.reason).length;
    const decisions = Object.entries(f.decision);
    fillSelect($("filter-decision"), decisions.map(([k, n]) => [k, `Model: ${human(k)} · ${n}`]), "Any model decision");
    $("filter-decision").hidden = decisions.length < 2;
    $("review-path").textContent = s.review_dir ? `${s.review_dir}/decisions.csv` : "review/decisions.csv";
  }

  /* ---------- queue ---------- */
  function queueQuery(offset) {
    const f = R.filters, params = new URLSearchParams({ offset: String(offset), limit: String(PAGE), review: f.state });
    if (R.summary.layer) params.set("layer", R.summary.layer);
    const [sort, order] = R.sort.split(":"); params.set("sort", sort); params.set("order", order || "asc");
    if (f.light) params.set("traffic_light", f.light);
    if (f.cls) params.set("declared_class", f.cls);
    if (f.reason) params.set("reason", f.reason);
    if (f.decision) params.set("decision", f.decision);
    if (f.unc === "set") params.set("min_set_size", "2"); else if (f.unc) params.set("max_confidence", f.unc);
    if (R.q) params.set("q", R.q);
    return `${base()}/parcels?${params}`;
  }
  async function loadQueue(reset, { keep = false } = {}) {
    const generation = R.generation;
    const offset = reset ? 0 : R.offset;
    try {
      const page = await App().api(queueQuery(offset));
      if (generation !== R.generation) return;
      R.items = reset ? page.items : [...R.items, ...page.items];
      R.total = page.total; R.offset = offset + page.items.length;
      renderQueue();
      if (reset && !keep) {
        if (R.items.length) select(R.items[0].parcel_id, { scroll: true });
        else showNone();
      }
    } catch (error) { App().message(error.message); }
  }
  function showNone() {
    R.current = null; R.detail = null;
    $("parcel-card").classList.add("is-none");
    $("parcel-title").textContent = "No parcel matches";
    $("parcel-eyebrow").textContent = R.filters.state === "pending" && R.summary.progress.reviewed ? "All done for these filters" : "Change the filters";
    for (const id of ["parcel-light", "parcel-reviewed"]) $(id).hidden = true;
    GeoMap.focusParcel(null);
  }
  function queueItem(item, k) {
    const { h, icon } = App();
    const differs = item.declared_class && item.predicted_class && item.declared_class !== item.predicted_class;
    const reviewed = item.review && DECISION[item.review.decision];
    return h("li", { role: "option", id: `q-${k}`, "aria-selected": String(item.parcel_id === R.current), "data-id": item.parcel_id,
      class: "queue-item", onclick: () => select(item.parcel_id) },
      h("span", { class: `light-dot tl-${item.traffic_light || "none"}`, role: "img", "aria-label": `${LIGHT_LABEL[item.traffic_light] || "No"} traffic light` }),
      h("span", { class: "q-main" },
        h("strong", { text: item.parcel_id }),
        h("span", { class: "q-classes" }, human(item.declared_class), differs ? h("span", { class: "q-arrow", text: " → " }) : null,
          differs ? h("b", { text: human(item.predicted_class) }) : null)),
      h("span", { class: "q-side" },
        reviewed ? h("span", { class: `q-state q-${item.review.decision}`, title: reviewed.label }, icon(reviewed.icon)) : null,
        h("span", { class: "q-conf", text: pct(item.confidence) })));
  }
  function renderQueue() {
    const list = $("queue");
    list.replaceChildren(...R.items.map(queueItem));
    if (!R.items.length) list.append(App().h("li", { class: "queue-empty", text: "No parcels match these filters." }));
    $("queue-more").hidden = R.items.length >= R.total;
    const shown = R.items.length < R.total ? `${R.items.length} of ${R.total}` : String(R.total);
    $("queue-summary").textContent = `${shown} parcel${R.total === 1 ? "" : "s"}${R.filters.state === "pending" ? " to review" : R.filters.state === "done" ? " reviewed" : ""}`;
    markQueue(false);
  }
  function markQueue(scroll = true) {
    for (const li of $("queue").querySelectorAll(".queue-item")) {
      const on = li.dataset.id === R.current;
      li.setAttribute("aria-selected", String(on));
      if (on) { $("queue").setAttribute("aria-activedescendant", li.id); if (scroll) li.scrollIntoView({ block: "nearest" }); }
    }
    const at = R.items.findIndex(i => i.parcel_id === R.current);
    $("parcel-position").textContent = at >= 0 ? `${at + 1} / ${R.total}` : "";
    $("parcel-prev").disabled = at <= 0;
    $("parcel-next").disabled = at < 0 || (at >= R.items.length - 1 && R.items.length >= R.total);
  }
  async function move(delta) {
    if (!R.items.length) return;
    let at = R.items.findIndex(i => i.parcel_id === R.current) + delta;
    if (at >= R.items.length && R.items.length < R.total) await loadQueue(false);
    at = Math.max(0, Math.min(R.items.length - 1, at));
    select(R.items[at].parcel_id, { scroll: true });
  }

  /* ---------- one parcel ---------- */
  async function select(id, { scroll = false } = {}) {
    R.current = id; markQueue(scroll);
    $("parcel-card").classList.remove("is-none");
    $("parcel-card").classList.add("is-loading");
    hideCorrection();
    $("decide-error").textContent = "";
    const generation = ++R.generation;
    try {
      const params = new URLSearchParams({ id });
      if (R.summary.layer) params.set("layer", R.summary.layer);
      if (R.evidence) params.set("evidence", R.evidence);
      const detail = await App().api(`${base()}/parcel?${params}`);
      if (generation !== R.generation) return;
      R.detail = detail; renderDetail(detail);
      loadChips(generation);
    } catch (error) {
      if (generation === R.generation) App().message(error.message);
    } finally {
      if (generation === R.generation) $("parcel-card").classList.remove("is-loading");
    }
  }

  function renderDetail(d) {
    const { h } = App();
    const item = R.items.find(i => i.parcel_id === d.parcel_id) || {};
    $("parcel-eyebrow").textContent = d.decision ? `Parcel · model ${human(d.decision)}` : "Parcel";
    $("parcel-title").textContent = d.parcel_id;
    const light = $("parcel-light");
    light.hidden = !d.traffic_light; light.className = `light-pill tl-${d.traffic_light}`;
    light.replaceChildren(h("span", { class: `light-dot tl-${d.traffic_light}`, "aria-hidden": "true" }), LIGHT_LABEL[d.traffic_light] || human(d.traffic_light));
    const last = d.history.length ? d.history[d.history.length - 1] : null;
    const badge = $("parcel-reviewed");
    badge.hidden = !last;
    if (last) {
      badge.dataset.status = { agree: "succeeded", disagree: "interrupted", unsure: "cancelled" }[last.decision];
      badge.textContent = `${DECISION[last.decision].label}${last.label && last.decision === "disagree" ? `: ${human(last.label)}` : ""}`;
    }
    $("parcel-declared").textContent = human(d.declared_class);
    $("parcel-predicted").textContent = human(d.predicted_class);
    const differs = d.declared_class && d.predicted_class && d.declared_class !== d.predicted_class;
    $("parcel-predicted").classList.toggle("is-different", !!differs);
    $("verdict-arrow").classList.toggle("is-different", !!differs);
    $("parcel-confidence").textContent = pct(d.probabilities[d.predicted_class] ?? item.confidence);
    $("parcel-set").textContent = d.conformal_set.length ? d.conformal_set.map(human).join(", ") : "—";
    $("parcel-set").title = d.conformal_set.length > 1 ? "Several classes are plausible at the chosen error rate" : "";

    const reasons = $("parcel-reasons"); reasons.replaceChildren();
    for (const r of d.reasons) reasons.append(h("li", { class: `reason reason-${r.source}`, title: `${human(r.source)} reason code` }, human(r.code)));
    // rules triggered in the parcel's evidence package (geocrop evidence) that the layer itself does not list
    const shown = new Set(d.reasons.map(r => r.code));
    for (const r of (d.evidence ? d.evidence.triggered_rules : [])) {
      if (!r.reason_code || shown.has(r.reason_code)) continue;
      shown.add(r.reason_code);
      reasons.append(h("li", { class: "reason reason-rule", title: `Evidence package: ${human(r.stage || r.rule)} → ${human(r.outcome)}` }, human(r.reason_code)));
    }
    if (!reasons.children.length) reasons.append(h("li", { class: "reason reason-none", text: "No reason codes" }));
    const ev = $("parcel-evidence");
    ev.hidden = !(d.evidence && d.evidence.html);
    ev.onclick = () => {
      const params = new URLSearchParams({ id: d.parcel_id, format: "html" });
      if (R.summary.layer) params.set("layer", R.summary.layer);
      if (R.evidence) params.set("evidence", R.evidence);
      App().download(`${base()}/evidence?${params}`, `evidence_${d.parcel_id}.html`).catch(err => App().message(err.message));
    };

    // probabilities: top classes as bars; predicted, declared and conformal-set membership marked
    const probs = Object.entries(d.probabilities).slice(0, 6);
    $("parcel-probs").replaceChildren(...probs.map(([cls, p]) => {
      const tags = [];
      if (cls === d.predicted_class) tags.push(h("span", { class: "tag tag-predicted", text: "predicted" }));
      if (cls === d.declared_class) tags.push(h("span", { class: "tag tag-declared", text: "declared" }));
      if (d.conformal_set.includes(cls)) tags.push(h("span", { class: "tag tag-set", text: "in set" }));
      const bar = h("span", { class: `prob-bar${cls === d.predicted_class ? " is-predicted" : ""}` }, h("i"));
      bar.firstChild.style.width = `${Math.max(0.5, p * 100)}%`;
      return h("li", {}, h("span", { class: "prob-name" }, h("span", { text: human(cls) }), ...tags), bar, h("span", { class: "prob-value", text: pct(p) }));
    }));
    $("probs-note").textContent = probs.length ? "Mean class probability over the parcel's interior pixels." : "This layer has no class probabilities.";

    kv($("parcel-markers"), d.markers); $("markers-count").textContent = String(Object.keys(d.markers).length);
    $("markers-block").hidden = !Object.keys(d.markers).length;
    kv($("parcel-attrs"), d.attributes); $("attrs-count").textContent = String(Object.keys(d.attributes).length);
    $("history-block").hidden = !d.history.length; $("history-count").textContent = String(d.history.length);
    $("parcel-history").replaceChildren(...d.history.slice().reverse().map(r => h("li", {},
      h("strong", { text: `${DECISION[r.decision]?.label || r.decision}${r.label && r.decision === "disagree" ? ` → ${human(r.label)}` : ""}` }),
      h("span", { class: "muted", text: ` · ${r.purpose} · ${r.analyst || "anonymous"} · ${new Date(r.timestamp_utc).toLocaleString()}` }),
      r.comment ? h("p", { text: r.comment }) : null)));

    $("ts-source").textContent = d.series.source === "signatures"
      ? `Signatures · ${d.series.files.join(", ")}`
      : d.series.source === "evidence" ? `Evidence package · observed points and interpolated curve`
      : d.series.panels.length ? "Parcel means from this job's imagery (no signature table found)" : "";
    R.chart = d.series; drawChart();
    GeoMap.focusParcel(d.geometry);
  }

  function kv(node, values) {
    const { h } = App();
    node.replaceChildren(...Object.entries(values).flatMap(([k, v]) => {
      let text = v === null || v === undefined || v === "" ? "—" : typeof v === "boolean" ? (v ? "yes" : "no") : String(v);
      if (typeof v === "number" && !Number.isInteger(v)) text = Math.abs(v) >= 100 ? v.toFixed(0) : v.toPrecision(3);
      return [h("dt", { text: human(k) }), h("dd", { text })];
    }));
  }

  /* ---------- time-series chart: inline SVG, one panel per sensor (never two y-axes) ---------- */
  function drawChart() {
    const box = $("ts-chart"), series = R.chart;
    box.replaceChildren();
    if (!series || !series.panels.length) {
      box.append(App().h("p", { class: "ts-empty", text: "No time series: add a signature table (geocrop signatures / s1-signatures) or imagery to the job's outputs, or pick a job with signatures as evidence." }));
      return;
    }
    const all = series.panels.flatMap(p => p.series.flatMap(s => s.points));
    const timeAxis = all.length > 0 && all.every(pt => /^\d{4}-\d{2}-\d{2}$/.test(pt.date || ""));
    const labels = [...new Set(all.map(pt => pt.label))];
    const xs = timeAxis ? all.map(pt => Date.parse(pt.date)) : [];
    const x0 = timeAxis ? Math.min(...xs) : 0, x1 = timeAxis ? Math.max(...xs) : Math.max(labels.length - 1, 1);
    const width = Math.max(280, box.clientWidth || 600), padL = 44, padR = 14;
    const xOf = (pt) => {
      const v = timeAxis ? Date.parse(pt.date) : labels.indexOf(pt.label);
      const span = x1 - x0 || 1;
      return padL + ((v - x0) / span) * (width - padL - padR) * (timeAxis || labels.length > 1 ? 1 : 0) + (timeAxis || labels.length > 1 ? 0 : (width - padL - padR) / 2);
    };
    const panels = [];
    for (const panel of series.panels) {
      const height = 136, top = 8, bottom = 24;
      const values = panel.series.flatMap(s => s.points.filter(p => p.observed && p.mean !== null).flatMap(p => (p.std ? [p.mean - p.std, p.mean + p.std] : [p.mean])));
      let lo = Math.min(...values), hi = Math.max(...values);
      if (panel.unit === "index") { lo = Math.min(lo, 0); hi = Math.max(hi, 1); }
      if (!Number.isFinite(lo) || !Number.isFinite(hi) || lo === hi) { lo = 0; hi = 1; }
      // "nice" ticks: a step of 1, 2 or 5 x 10^k and about four intervals
      const raw = (hi - lo) / 4, mag = 10 ** Math.floor(Math.log10(raw));
      const stepY = [1, 2, 2.5, 5, 10].map(f => f * mag).find(v => v >= raw) || raw;
      lo = Math.floor(lo / stepY + 1e-9) * stepY; hi = Math.ceil(hi / stepY - 1e-9) * stepY;
      const ticks = Math.round((hi - lo) / stepY), digits = Math.max(0, -Math.floor(Math.log10(stepY) + 1e-9));
      const yOf = (v) => top + (1 - (v - lo) / (hi - lo)) * (height - top - bottom);
      const svg = svgEl("svg", { viewBox: `0 0 ${width} ${height}`, width, height, class: "ts-svg", role: "img",
        "aria-label": `${panel.name}: ${panel.series.map(s => s.name).join(", ")} over ${labels.length} dates` });
      // title + legend (legend only for two or more series), as HTML so it wraps on narrow screens
      const { h } = App();
      box.append(h("div", { class: "ts-panel-head" },
        h("span", { class: "ts-panel-title", text: `${panel.name} · ${panel.series.length === 1 ? panel.series[0].name : panel.unit}` }),
        panel.series.length > 1 ? h("span", { class: "ts-legend-row" }, ...panel.series.map((s, k) => h("span", { class: "ts-legend-item" }, h("span", { class: `ts-key s${k + 1}` }), s.name))) : null));
      // grid and y labels (recessive)
      for (let i = 0; i <= ticks; i++) {
        const v = lo + ((hi - lo) * i) / ticks, y = yOf(v);
        svg.append(svgEl("line", { x1: padL, x2: width - padR, y1: y, y2: y, class: i === 0 ? "ts-axis" : "ts-grid" }));
        const t = svgEl("text", { x: padL - 6, y: y + 3.5, "text-anchor": "end", class: "ts-tick" });
        t.textContent = v.toFixed(Math.min(digits + (stepY / mag === 2.5 ? 1 : 0), 3)); svg.append(t);
      }
      // x labels: months on a time axis, folder labels otherwise
      const xTicks = [];
      if (timeAxis) {
        const d = new Date(x0); d.setUTCDate(1); d.setUTCHours(0, 0, 0, 0);
        if (d.getTime() < x0) d.setUTCMonth(d.getUTCMonth() + 1);
        while (d.getTime() <= x1) { xTicks.push([d.getTime(), MONTHS[d.getUTCMonth()]]); d.setUTCMonth(d.getUTCMonth() + 1); }
        const every = Math.ceil(xTicks.length / Math.max(2, Math.floor((width - padL) / 56)));
        xTicks.forEach(([v, text], i) => {
          if (i % every) return;
          const x = padL + ((v - x0) / (x1 - x0 || 1)) * (width - padL - padR);
          svg.append(svgEl("line", { x1: x, x2: x, y1: height - bottom, y2: height - bottom + 4, class: "ts-axis" }));
          const t = svgEl("text", { x, y: height - 6, "text-anchor": "middle", class: "ts-tick" }); t.textContent = text; svg.append(t);
        });
      } else {
        labels.forEach((label) => {
          const t = svgEl("text", { x: xOf({ label }), y: height - 6, "text-anchor": "middle", class: "ts-tick" });
          t.textContent = fmtDate(label, true); svg.append(t);
        });
      }
      // series: ±1 std band (single series), 2px line broken at gaps, observed points with a surface ring
      panel.series.forEach((s, k) => {
        const pts = s.points.filter(p => p.observed && p.mean !== null);
        if (panel.series.length === 1 && pts.some(p => p.std)) {
          const up = pts.map(p => `${xOf(p)},${yOf(p.mean + (p.std || 0))}`), down = pts.slice().reverse().map(p => `${xOf(p)},${yOf(p.mean - (p.std || 0))}`);
          if (pts.length > 1) svg.append(svgEl("polygon", { points: [...up, ...down].join(" "), class: `ts-band s${k + 1}` }));
        }
        let d = "", pen = false;
        for (const p of s.points) {
          if (!p.observed || p.mean === null) { pen = false; continue; }
          d += `${pen ? "L" : "M"}${xOf(p).toFixed(1)},${yOf(p.mean).toFixed(1)}`; pen = true;
        }
        if (d) svg.append(svgEl("path", { d, class: `ts-line s${k + 1}` }));
        // dense series stay a clean line; sparse ones show every observation as a ringed marker
        if (pts.length <= 24 && (width - padL - padR) / Math.max(pts.length, 1) >= 16) for (const p of pts) svg.append(svgEl("circle", { cx: xOf(p), cy: yOf(p.mean), r: 4, class: `ts-dot s${k + 1}` }));
        // missing acquisitions (clouds / no data) as small ticks on the axis
        for (const p of s.points.filter(p => !p.observed)) svg.append(svgEl("line", { x1: xOf(p), x2: xOf(p), y1: height - bottom - 4, y2: height - bottom, class: "ts-missing" }));
      });
      const cross = svgEl("line", { y1: top - 4, y2: height - bottom, class: "ts-cross", visibility: "hidden" });
      svg.append(cross);
      const marks = panel.series.map((s, k) => { const c = svgEl("circle", { r: 4.5, class: `ts-dot s${k + 1}`, visibility: "hidden" }); svg.append(c); return c; });
      panels.push({ svg, panel, cross, marks, xOf, yOf });
      box.append(svg);
    }
    // hover layer: crosshair across all panels and one tooltip with every value at the nearest date
    const keys = timeAxis ? [...new Set(all.map(p => p.date))].sort() : labels;
    const tip = App().h("div", { class: "ts-tip", role: "status" }); tip.hidden = true; box.append(tip);
    const keyOf = (p) => (timeAxis ? p.date : p.label);
    const show = (clientX) => {
      const rect = panels[0].svg.getBoundingClientRect(), x = ((clientX - rect.left) / rect.width) * width;
      let best = null, dist = Infinity;
      for (const k of keys) { const px = panels[0].xOf(timeAxis ? { date: k } : { label: k }); if (Math.abs(px - x) < dist) { dist = Math.abs(px - x); best = k; } }
      if (best === null) return;
      const px = panels[0].xOf(timeAxis ? { date: best } : { label: best });
      const rows = [];
      for (const { panel, cross, marks, yOf } of panels) {
        cross.setAttribute("x1", px); cross.setAttribute("x2", px); cross.setAttribute("visibility", "visible");
        if (panels.length > 1) rows.push(App().h("div", { class: "ts-tip-panel", text: panel.name }));
        panel.series.forEach((s, k) => {
          const p = s.points.find(q => keyOf(q) === best);
          const seen = p && p.observed && p.mean !== null;
          marks[k].setAttribute("visibility", seen ? "visible" : "hidden");
          if (seen) { marks[k].setAttribute("cx", px); marks[k].setAttribute("cy", yOf(p.mean)); }
          if (!p) return;
          const value = p.observed && p.mean !== null ? `${p.mean.toFixed(panel.unit === "dB" ? 1 : 3)}${panel.unit === "dB" ? " dB" : ""}` : "not observed";
          rows.push(App().h("div", { class: "ts-tip-row" }, App().h("span", { class: `ts-key s${k + 1}` }), App().h("span", { text: s.name }),
            App().h("strong", { text: value }), p.n_valid !== null && p.n_valid !== undefined ? App().h("small", { text: `n=${p.n_valid}` }) : null));
        });
      }
      tip.replaceChildren(App().h("div", { class: "ts-tip-title", text: fmtDate(best) }), ...rows);
      tip.hidden = false;
      const left = (px / width) * rect.width;
      tip.style.left = `${Math.min(Math.max(left + 12, 0), rect.width - tip.offsetWidth - 4)}px`;
      if (left + 12 + tip.offsetWidth > rect.width) tip.style.left = `${Math.max(0, left - tip.offsetWidth - 12)}px`;
    };
    const hide = () => { tip.hidden = true; for (const { cross, marks } of panels) { cross.setAttribute("visibility", "hidden"); for (const m of marks) m.setAttribute("visibility", "hidden"); } };
    box.onpointermove = (e) => show(e.clientX);
    box.onpointerleave = hide;
    box.tabIndex = 0;
    let at = -1;
    box.onkeydown = (e) => {
      if (!["ArrowLeft", "ArrowRight"].includes(e.key)) return;
      e.preventDefault(); e.stopPropagation();
      at = Math.max(0, Math.min(keys.length - 1, at + (e.key === "ArrowRight" ? 1 : -1)));
      const rect = panels[0].svg.getBoundingClientRect();
      show(rect.left + (panels[0].xOf(timeAxis ? { date: keys[at] } : { label: keys[at] }) / width) * rect.width);
    };
    box.onblur = hide;
    if (series.panels.some(p => p.more)) box.append(App().h("p", { class: "muted small", text: "More series exist; the first four per sensor are drawn." }));
  }

  /* ---------- chips ---------- */
  async function loadChips(generation) {
    const strip = $("chips"), { h } = App();
    if (!R.summary.sources.imagery) { strip.replaceChildren(h("p", { class: "ts-empty", text: "No imagery in this job's outputs (composites or scene folders with B02, B03, B04, B08)." })); return; }
    strip.replaceChildren(...Array.from({ length: Math.min(6, R.summary.sources.imagery) }, () => h("div", { class: "chip-tile is-skeleton" })));
    try {
      const params = new URLSearchParams({ id: R.current, kind: R.chipsKind, size: "96" });
      if (R.summary.layer) params.set("layer", R.summary.layer);
      const result = await App().api(`${base()}/chips?${params}`);
      if (generation !== R.generation) return;
      if (!result.chips.length) { strip.replaceChildren(h("p", { class: "ts-empty", text: "No imagery covers this parcel." })); return; }
      strip.replaceChildren(...result.chips.map(c => {
        const img = h("img", { src: c.url, alt: `${R.chipsKind === "true" ? "True colour" : "NDVI"} ${fmtDate(c.label)}`, width: "112", height: "112", decoding: "async" });
        const outline = svgEl("svg", { viewBox: "0 0 1 1", preserveAspectRatio: "none", class: "chip-outline", "aria-hidden": "true" });
        for (const ring of c.outline) {
          const pts = ring.map(([x, y]) => `${x},${y}`).join(" ");
          outline.append(svgEl("polygon", { points: pts, class: "chip-halo" }), svgEl("polygon", { points: pts, class: "chip-line" }));
        }
        const cloudy = c.valid_fraction !== null && c.valid_fraction < 0.7;
        return h("figure", { class: `chip-tile${cloudy ? " is-cloudy" : ""}` }, h("div", { class: "chip-frame" }, img, outline,
          cloudy ? h("span", { class: "chip-badge", text: c.valid_fraction === 0 ? "No data" : `${Math.round(c.valid_fraction * 100)}% clear` }) : null),
          h("figcaption", { text: fmtDate(c.label, true) }));
      }));
    } catch (error) {
      if (generation === R.generation) strip.replaceChildren(h("p", { class: "ts-empty", text: `Chips unavailable: ${error.message}` }));
    }
  }

  /* ---------- decisions ---------- */
  function purpose() { return $("decide-purpose").querySelector('[aria-checked="true"]')?.dataset.value || ""; }
  function setSegment(container, value) {
    for (const b of container.querySelectorAll("[data-value]")) {
      b.setAttribute("aria-checked", String(b.dataset.value === value)); b.tabIndex = b.dataset.value === value ? 0 : -1;
    }
  }
  function syncMetaSummary() {
    const p = purpose(), who = $("decide-analyst").value.trim();
    $("decide-summary").textContent = `${p ? p[0].toUpperCase() + p.slice(1) : "Choose training or validation"}${who ? ` · ${who}` : ""} · comment`;
  }
  function hideCorrection() { $("decide-correct").hidden = true; $("decide-disagree").setAttribute("aria-pressed", "false"); }
  function showCorrection() {
    if (!R.detail) return;
    const d = R.detail, select = $("decide-class");
    const options = R.summary.classes.filter(c => c !== d.predicted_class);
    select.replaceChildren(...options.map(c => new Option(human(c), c)), new Option("Other class…", "__other"));
    select.value = d.declared_class && d.declared_class !== d.predicted_class && options.includes(d.declared_class) ? d.declared_class : options[0] || "__other";
    $("decide-new-class").hidden = select.value !== "__other";
    $("decide-correct").hidden = false; $("decide-disagree").setAttribute("aria-pressed", "true");
    select.focus();
  }
  async function decide(decision) {
    if (!R.detail || R.busy) return;
    const err = $("decide-error"); err.textContent = "";
    if (!purpose()) {
      err.textContent = "Choose whether this label is for Training or Validation — it decides how active-learning ingest uses it.";
      $("decide").classList.add("show-meta");
      $("decide-purpose").classList.add("needs-choice"); $("decide-purpose").querySelector("button").focus();
      return;
    }
    if (decision === "disagree" && $("decide-correct").hidden) { showCorrection(); return; }
    let label = null;
    if (decision === "disagree") {
      label = $("decide-class").value === "__other" ? $("decide-new-class").value.trim() : $("decide-class").value;
      if (!label) { err.textContent = "Enter the correct class."; $("decide-new-class").focus(); return; }
    }
    const body = { parcel_id: R.detail.parcel_id, decision, purpose: purpose(), analyst: $("decide-analyst").value.trim(), comment: $("decide-comment").value.trim() };
    if (label) body.label = label;
    if (R.summary.layer) body.layer = R.summary.layer;
    R.busy = true;
    for (const b of document.querySelectorAll(".decide-btn")) b.disabled = true;
    try {
      const result = await App().api(`${base()}/decisions`, { method: "POST", body: JSON.stringify(body) });
      const pid = R.detail.parcel_id, at = R.items.findIndex(i => i.parcel_id === pid);
      $("decide-comment").value = ""; hideCorrection();
      const rec = result.decision;
      App().toast({ title: `${DECISION[decision].label}${decision === "disagree" ? `: ${human(rec.label)}` : ""}`, text: `${pid} · ${rec.purpose}${result.new_class ? " · new class name" : ""}`, kind: "info", timeout: 2200 });
      if (at >= 0) R.items[at] = { ...R.items[at], review: { decision, label: rec.label, timestamp_utc: rec.timestamp_utc } };
      refreshSummary();
      if (R.filters.state === "pending" && at >= 0) {           // reviewed parcels leave the "to review" queue
        R.items.splice(at, 1); R.total = Math.max(0, R.total - 1); R.offset = Math.max(0, R.offset - 1);
        renderQueue();
        if (R.items.length) select(R.items[Math.min(at, R.items.length - 1)].parcel_id, { scroll: true });
        else if (R.total) loadQueue(true); else showNone();
      } else {
        renderQueue();
        if (at >= 0 && at < R.items.length - 1) select(R.items[at + 1].parcel_id, { scroll: true });
        else select(pid);
      }
    } catch (error) { err.textContent = error.message; }
    finally { R.busy = false; for (const b of document.querySelectorAll(".decide-btn")) b.disabled = false; }
  }

  /* ---------- export ---------- */
  function openExport() {
    if (!R.summary) return;
    $("export-result").hidden = true; $("export-download").hidden = true; $("export-error").textContent = "";
    setSegment($("export-purpose"), purpose() || "training");
    $("export-sheet").showModal();
  }
  async function runExport(e) {
    e.preventDefault();
    const p = $("export-purpose").querySelector('[aria-checked="true"]').dataset.value;
    const raw = $("export-round").value.trim();
    const body = { purpose: p };
    if (raw) {
      if (!/^\d+$/.test(raw)) { $("export-error").textContent = "The round is a whole number."; return; }
      body.round = Number(raw);
    }
    if (R.summary.layer) body.layer = R.summary.layer;
    $("export-error").textContent = "";
    try {
      const result = await App().api(`${base()}/export`, { method: "POST", body: JSON.stringify(body) });
      $("export-result").hidden = false;
      $("export-path").textContent = result.path;
      $("export-rows").textContent = `${result.rows} (${result.labelled} with a class)`;
      $("export-command-wrap").hidden = !result.ingest_argv;
      if (result.ingest_argv) App().showCommand($("export-command"), result.ingest_argv);
      else $("export-command").textContent = "";
      $("export-download").hidden = false;
      $("export-download").onclick = () => App().download(`${base()}/file?name=${encodeURIComponent(result.path.split("/").pop())}${layerParam() ? `&${layerParam()}` : ""}`, result.path.split("/").pop()).catch(err => App().message(err.message));
      if (!result.ingest_argv) $("export-error").textContent = "Validation labels are kept apart: pass this file to your accuracy assessment, never to training.";
    } catch (error) { $("export-error").textContent = error.message; }
  }

  /* ---------- keyboard (called by app.js while the Review view is active) ---------- */
  function key(e) {
    if (!R.summary || $("review-main").hidden) return false;
    const k = e.key.toLowerCase();
    if (k === "j" || e.key === "ArrowDown" && e.target.closest?.("#queue")) { move(1); return true; }
    if (k === "k" || e.key === "ArrowUp" && e.target.closest?.("#queue")) { move(-1); return true; }
    if (k === "a") { decide("agree"); return true; }
    if (k === "d") { decide("disagree"); return true; }
    if (k === "u") { decide("unsure"); return true; }
    return false;
  }

  /* ---------- wiring ---------- */
  function segmentedKeys(container, onPick) {
    container.addEventListener("click", (e) => { const b = e.target.closest("[data-value]"); if (b) onPick(b.dataset.value); });
    container.addEventListener("keydown", (e) => {
      if (!["ArrowLeft", "ArrowRight"].includes(e.key)) return;
      e.preventDefault(); e.stopPropagation();
      const buttons = [...container.querySelectorAll("[data-value]")], at = buttons.findIndex(b => b.getAttribute("aria-checked") === "true");
      const next = buttons[(at + (e.key === "ArrowRight" ? 1 : -1) + buttons.length) % buttons.length];
      onPick(next.dataset.value); next.focus();
    });
  }
  function bind() {
    $("decide-analyst").value = store.get("geocrop-analyst");
    const saved = store.get("geocrop-review-purpose");
    if (saved === "training" || saved === "validation") setSegment($("decide-purpose"), saved);
    $("decide-analyst").addEventListener("change", () => { store.set("geocrop-analyst", $("decide-analyst").value.trim()); syncMetaSummary(); });
    $("decide-summary").addEventListener("click", () => { $("decide").classList.toggle("show-meta"); if ($("decide").classList.contains("show-meta")) $("decide-comment").focus(); });
    syncMetaSummary();
    segmentedKeys($("decide-purpose"), (v) => { setSegment($("decide-purpose"), v); store.set("geocrop-review-purpose", v); $("decide-purpose").classList.remove("needs-choice"); $("decide-error").textContent = ""; syncMetaSummary(); });
    segmentedKeys($("export-purpose"), (v) => setSegment($("export-purpose"), v));
    segmentedKeys($("chips-kind"), (v) => { R.chipsKind = v; setSegment($("chips-kind"), v); if (R.current) loadChips(R.generation); });
    for (const b of document.querySelectorAll(".decide-btn")) b.addEventListener("click", () => decide(b.dataset.decision));
    $("decide").addEventListener("submit", (e) => { e.preventDefault(); decide("disagree"); });
    $("decide-class").addEventListener("change", () => { $("decide-new-class").hidden = $("decide-class").value !== "__other"; if (!$("decide-new-class").hidden) $("decide-new-class").focus(); });
    $("decide-correct").addEventListener("keydown", (e) => { if (e.key === "Escape") { e.stopPropagation(); hideCorrection(); $("decide-disagree").focus(); } });
    $("decide-class").addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); decide("disagree"); } });
    for (const [id, field] of [["filter-state", "state"], ["filter-class", "cls"], ["filter-reason", "reason"], ["filter-uncertainty", "unc"], ["filter-decision", "decision"]]) {
      $(id).addEventListener("change", () => { R.filters[field] = $(id).value; loadQueue(true); });
    }
    $("filter-state").value = R.filters.state;
    $("queue-sort").addEventListener("change", () => { R.sort = $("queue-sort").value; loadQueue(true); });
    let timer = null;
    $("queue-search").addEventListener("input", () => { clearTimeout(timer); timer = setTimeout(() => { R.q = $("queue-search").value.trim(); loadQueue(true); }, 200); });
    $("queue-more").addEventListener("click", () => loadQueue(false, { keep: true }));
    $("parcel-prev").addEventListener("click", () => move(-1));
    $("parcel-next").addEventListener("click", () => move(1));
    $("review-demo").addEventListener("click", () => App().runDemo());
    $("review-export").addEventListener("click", openExport);
    $("export-form").addEventListener("submit", runExport);
    $("export-close").addEventListener("click", () => $("export-sheet").close());
    $("export-copy").addEventListener("click", () => App().copy($("export-command").textContent));
    document.addEventListener("geocrop:view", (e) => { if (e.detail.view === "review") sync(); });
    document.addEventListener("geocrop:job", () => sync());
    document.addEventListener("geocrop:connected", () => sync());
    document.addEventListener("geomap:parcel", (e) => { if (R.summary) select(e.detail.id, { scroll: true }); });
    document.addEventListener("geomap:state", (e) => { if (e.detail.state === "ready" && R.detail && App().view() === "review") GeoMap.focusParcel(R.detail.geometry, { fly: false }); });
    if (typeof ResizeObserver !== "undefined") {
      let last = 0;
      new ResizeObserver((entries) => { const w = Math.round(entries[0].contentRect.width); if (w && Math.abs(w - last) > 8) { last = w; if (R.chart) drawChart(); } }).observe($("ts-chart"));
    }
  }
  document.addEventListener("DOMContentLoaded", bind);
  return { key, sync };
})();
window.GeoCropReview = GeoCropReview;
