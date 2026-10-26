"use strict";
/* GeoCrop web terminal: a macOS-style console with live syntax colouring, IntelliSense-like
   completion, history and streamed job output. Self-contained; mounts into #terminal-root and
   talks to the engine through window.GeoCropApp.api (authenticated helper exposed by app.js). */
(() => {
  const HISTORY_KEY = "geocrop.terminal.history";
  const LOGIN_KEY = "geocrop.terminal.lastLogin";
  const HISTORY_MAX = 100;
  const TERMINAL_STATES = new Set(["succeeded", "failed", "cancelled", "interrupted"]);
  const BUILTINS = [
    { name: "help", description: "List commands; help <command> shows its flags" },
    { name: "?", description: "Alias of help" },
    { name: "clear", description: "Clear the screen (⌃L)" },
    { name: "history", description: "Show the command history" },
    { name: "jobs", description: "Recent engine jobs and their status" },
  ];
  const PATH_WORDS = new Set(["path", "out", "root", "config", "raster", "features", "samples", "polygons", "parcels",
    "observations", "model", "reference", "scenes", "safe", "manifest", "input", "output", "aoi", "training", "stack",
    "stacks", "dir", "file", "inputs", "composites", "prediction", "probabilities", "classes", "validation", "segments",
    "signatures", "images", "masks", "mask", "geotiff", "dem", "params", "groups", "aoa", "calendar", "markers"]);
  const NON_PATH_TAIL = new Set(["days", "m", "px", "km2", "fraction", "field", "size", "distance", "mode", "step",
    "spacing", "seed", "id", "n", "split"]);
  const NUMBER = /^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$/;
  const OPERATOR = /[;&|<>`]|\$\(/;

  const root = document.getElementById("terminal-root");
  if (!root) return;

  const S = {
    commands: [], byName: new Map(), connected: false, health: null,
    history: loadHistory(), histIndex: 0, draft: "",
    popup: { open: false, items: [], active: 0, ctx: null, touched: false, seq: 0 },
    ghost: null, running: null, search: null, fileCache: new Map(), focused: false,
  };
  S.histIndex = S.history.length;

  /* ---------------------------------------------------------------- DOM ---- */
  const el = (tag, cls, text) => {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined && text !== null) node.textContent = text;
    return node;
  };
  const term = el("div", "gc-terminal");
  const win = el("div", "gc-term-window");
  win.setAttribute("role", "region");
  win.setAttribute("aria-label", "GeoCrop terminal");
  const titlebar = el("div", "gc-term-titlebar");
  const lights = el("div", "gc-term-lights");
  lights.setAttribute("aria-hidden", "true");
  for (const c of ["close", "min", "max"]) lights.append(el("span", "gc-light gc-light-" + c));
  const title = el("div", "gc-term-title", "geocrop — zsh-like session");
  const badge = el("div", "gc-term-badge");
  titlebar.append(lights, title, badge);

  const screen = el("div", "gc-term-screen");
  screen.tabIndex = -1;
  const output = el("div", "gc-term-output");
  output.setAttribute("role", "log");
  output.setAttribute("aria-label", "Terminal output");
  const searchBar = el("div", "gc-term-search");
  searchBar.hidden = true;
  const searchLabel = el("label", "gc-term-search-label", "bck-i-search:");
  const searchInput = el("input", "gc-term-search-input");
  searchInput.id = "gc-term-search-input";
  searchLabel.htmlFor = searchInput.id;
  searchInput.type = "text";
  searchInput.spellcheck = false;
  searchInput.autocomplete = "off";
  const searchInfo = el("span", "gc-term-search-info");
  searchBar.append(searchLabel, searchInput, searchInfo);

  const inputLine = el("div", "gc-term-inputline");
  const prompt = makePrompt();
  const editor = el("div", "gc-term-editor");
  const mirror = el("pre", "gc-term-mirror");
  mirror.setAttribute("aria-hidden", "true");
  const input = el("textarea", "gc-term-input");
  Object.assign(input, { rows: 1, spellcheck: false, autocomplete: "off", wrap: "soft" });
  input.setAttribute("autocapitalize", "off");
  input.setAttribute("autocorrect", "off");
  input.setAttribute("aria-label", "GeoCrop command line");
  input.setAttribute("role", "combobox");
  input.setAttribute("aria-autocomplete", "both");
  input.setAttribute("aria-haspopup", "listbox");
  input.setAttribute("aria-expanded", "false");
  input.setAttribute("aria-controls", "gc-term-listbox");
  input.setAttribute("aria-describedby", "gc-term-signature");
  editor.append(mirror, input);
  inputLine.append(prompt, editor);
  const signature = el("div", "gc-term-signature");
  signature.id = "gc-term-signature";
  signature.setAttribute("aria-live", "polite");
  screen.append(output, searchBar, inputLine, signature);

  const statusbar = el("div", "gc-term-statusbar");
  for (const [k, label] of [["⇥", "complete"], ["→", "accept"], ["↑↓", "history"], ["⌃R", "search"], ["⌃C", "cancel"], ["⌃L", "clear"]]) {
    const item = el("span", "gc-term-hint");
    item.append(el("kbd", null, k), document.createTextNode(" " + label));
    statusbar.append(item);
  }
  const workspaceLabel = el("span", "gc-term-workspace", "not connected");
  statusbar.append(workspaceLabel);
  win.append(titlebar, screen, statusbar);

  const popup = el("div", "gc-term-popup");
  popup.hidden = true;
  const listbox = el("ul", "gc-term-list");
  listbox.id = "gc-term-listbox";
  listbox.setAttribute("role", "listbox");
  listbox.setAttribute("aria-label", "Suggestions");
  const docPane = el("div", "gc-term-doc");
  popup.append(listbox, docPane);
  const live = el("div", "gc-term-sr");
  live.setAttribute("role", "status");
  live.setAttribute("aria-live", "polite");
  term.append(win, popup, live);
  root.replaceChildren(term);

  function makePrompt() {
    const p = el("span", "gc-prompt");
    p.setAttribute("aria-hidden", "true");
    p.append(el("span", "gc-prompt-user", "user"), el("span", "gc-prompt-at", "@"), el("span", "gc-prompt-host", "geocrop"),
      el("span", "gc-prompt-dir", " ~ "), el("span", "gc-prompt-sym", "%"), document.createTextNode(" "));
    return p;
  }

  /* ------------------------------------------------------------- helpers ---- */
  function loadHistory() {
    try {
      const value = JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
      return Array.isArray(value) ? value.filter(v => typeof v === "string").slice(-HISTORY_MAX) : [];
    } catch { return []; }
  }
  function saveHistory() {
    try { localStorage.setItem(HISTORY_KEY, JSON.stringify(S.history.slice(-HISTORY_MAX))); } catch { /* private mode */ }
  }
  function api(path, options) {
    const app = window.GeoCropApp;
    if (!app || typeof app.api !== "function") return Promise.reject(new Error("Not connected"));
    return app.api(path, options);
  }
  const builtin = name => BUILTINS.find(b => b.name === name) || null;
  function isPathFlag(field) {
    if (!field || field.boolean || field.choices) return false;
    if (typeof field.default === "number") return false;
    const words = field.flag.replace(/^-+/, "").split("-");
    if (NON_PATH_TAIL.has(words[words.length - 1])) return false;
    return words.some(w => PATH_WORDS.has(w));
  }
  function quoteIfNeeded(value) {
    if (value && !/[\s"'\\;&|<>`$]/.test(value)) return value;
    return '"' + value.replace(/(["\\$`])/g, "\\$1") + '"';
  }
  function nearBottom() { return screen.scrollHeight - screen.scrollTop - screen.clientHeight < 48; }
  function scrollToEnd() { screen.scrollTop = screen.scrollHeight; }
  function append(node) {
    const stick = nearBottom();
    output.append(node);
    while (output.childElementCount > 3000) output.firstElementChild.remove();
    if (stick || !S.running) scrollToEnd();
    return node;
  }
  function print(text, cls) { return append(el("div", "gc-line" + (cls ? " " + cls : ""), text)); }
  function announce(text) { live.textContent = text; }
  function fmtDuration(ms) {
    if (!Number.isFinite(ms) || ms < 0) return "";
    if (ms < 1000) return `${Math.round(ms)} ms`;
    if (ms < 60000) return `${(ms / 1000).toFixed(ms < 10000 ? 2 : 1)} s`;
    const m = Math.floor(ms / 60000), s = Math.round((ms % 60000) / 1000);
    return `${m} min ${s} s`;
  }

  /* ----------------------------------------------------------- tokenizer ---- */
  function tokenize(line) {
    const tokens = [];
    let i = 0;
    while (i < line.length) {
      if (/\s/.test(line[i])) { i++; continue; }
      const start = i;
      let value = "", quote = null, quoted = false;
      while (i < line.length) {
        const ch = line[i];
        if (quote) {
          if (ch === quote) { quote = null; i++; continue; }
          if (ch === "\\" && quote === '"' && /["\\$`]/.test(line[i + 1] || "")) { value += line[i + 1]; i += 2; continue; }
          value += ch; i++; continue;
        }
        if (/\s/.test(ch)) break;
        if (ch === '"' || ch === "'") { quote = ch; quoted = true; i++; continue; }
        if (ch === "\\" && i + 1 < line.length) { value += line[i + 1]; i += 2; continue; }
        value += ch; i++;
      }
      tokens.push({ start, end: i, raw: line.slice(start, i), value, quoted, open: !!quote });
    }
    return tokens;
  }

  /* Classify tokens: program, command, flags, values. */
  function analyse(line) {
    const tokens = tokenize(line);
    const a = { tokens, k: 0, command: null, builtin: null, cmdToken: null, used: new Map(), expecting: null, unknown: false };
    if (tokens[0] && !tokens[0].quoted && tokens[0].value === "geocrop") { tokens[0].kind = "prog"; a.k = 1; }
    const ct = tokens[a.k];
    if (!ct) return a;
    a.cmdToken = ct;
    a.command = S.byName.get(ct.value) || null;
    a.builtin = a.command ? null : builtin(ct.value);
    ct.kind = a.command ? "cmd" : a.builtin ? "builtin" : "cmd-err";
    a.unknown = !a.command && !a.builtin;
    let expecting = null;
    for (let i = a.k + 1; i < tokens.length; i++) {
      const t = tokens[i];
      if (!t.quoted && OPERATOR.test(t.raw)) { t.kind = "op"; expecting = null; continue; }
      if (a.builtin) { t.kind = a.builtin.name === "help" || a.builtin.name === "?" ? (S.byName.has(t.value) ? "cmd" : "value") : "value"; continue; }
      const looksFlag = !t.quoted && t.value.startsWith("-") && t.value.length > 1 && !(expecting && NUMBER.test(t.value));
      if (looksFlag) {
        const eq = t.value.indexOf("=");
        const flag = eq > 0 ? t.value.slice(0, eq) : t.value;
        const field = a.command ? a.command.fields.find(f => f.flag === flag) || null : null;
        t.flag = flag; t.field = field; t.eq = eq > 0 ? t.raw.indexOf("=") : -1;
        t.kind = !a.command ? "flag" : field ? "flag" : "flag-err";
        if (field) a.used.set(flag, (a.used.get(flag) || 0) + 1);
        if (eq > 0 && field) t.valueKind = valueKind(t.value.slice(eq + 1), field, false, false);
        expecting = field && !field.boolean && eq < 0 ? { field, count: 0 } : null;
        continue;
      }
      const field = expecting ? expecting.field : null;
      t.owner = field;
      t.kind = valueKind(t.value, field, t.quoted, t.open);
      if (expecting) { expecting.count++; if (!field.multiple) expecting = null; }
    }
    a.expecting = expecting;
    return a;
  }
  function valueKind(value, field, quoted, open) {
    if (open) return "str";
    if (field && field.choices) return field.choices.map(String).includes(value) ? "choice" : "value-err";
    if (!quoted && NUMBER.test(value)) return "num";
    if (quoted) return "str";
    if (/[/\\]/.test(value) || /\.[A-Za-z][A-Za-z0-9]{0,7}$/.test(value) || isPathFlag(field)) return "path";
    return "value";
  }

  /* Coloured segments for the mirror / echo. `caret` suppresses errors on the word being typed. */
  function segments(line, a, caret) {
    const segs = [];
    let pos = 0;
    const editing = t => caret >= 0 && caret === t.end;
    for (const t of a.tokens) {
      if (t.start > pos) segs.push({ text: line.slice(pos, t.start), cls: "" });
      let kind = t.kind || "value";
      if (kind === "cmd-err" && editing(t) && [...S.byName.keys(), ...BUILTINS.map(b => b.name), "geocrop"].some(n => n.startsWith(t.value))) kind = "cmd-partial";
      if (kind === "flag-err" && editing(t) && a.command && a.command.fields.some(f => f.flag.startsWith(t.flag))) kind = "flag-partial";
      if (kind === "value-err" && editing(t) && t.owner && t.owner.choices && t.owner.choices.some(c => String(c).startsWith(t.value))) kind = "value";
      if (t.eq > 0) {
        segs.push({ text: t.raw.slice(0, t.eq), cls: "gc-tk-" + kind });
        segs.push({ text: "=", cls: "gc-tk-punct" });
        segs.push({ text: t.raw.slice(t.eq + 1), cls: "gc-tk-" + (t.valueKind || "value") });
      } else segs.push({ text: t.raw, cls: "gc-tk-" + kind });
      pos = t.end;
    }
    if (pos < line.length) segs.push({ text: line.slice(pos), cls: "" });
    return segs;
  }
  function segmentNodes(segs, caret, ghost) {
    const nodes = [];
    let pos = 0, placed = caret < 0;
    const span = (cls, text) => { const s = el("span", cls || null, text); return s; };
    for (const seg of segs) {
      const end = pos + seg.text.length;
      if (!placed && caret >= pos && caret < end) {
        const at = caret - pos;
        if (at > 0) nodes.push(span(seg.cls, seg.text.slice(0, at)));
        const c = span("gc-caret " + seg.cls, seg.text[at]);
        nodes.push(c);
        if (at + 1 < seg.text.length) nodes.push(span(seg.cls, seg.text.slice(at + 1)));
        placed = true;
      } else nodes.push(span(seg.cls, seg.text));
      pos = end;
    }
    if (!placed) {
      if (ghost) {
        nodes.push(span("gc-caret gc-ghost", ghost[0]));
        if (ghost.length > 1) nodes.push(span("gc-ghost", ghost.slice(1)));
      } else nodes.push(span("gc-caret gc-caret-end", " "));
    } else if (ghost) nodes.push(span("gc-ghost", ghost));
    return nodes;
  }
  function highlighted(line) {
    const frag = el("span", "gc-echo-cmd");
    frag.append(...segmentNodes(segments(line, analyse(line), -1), -1, null));
    return frag;
  }

  /* ------------------------------------------------------------- render ---- */
  function render() {
    const line = input.value;
    const collapsed = input.selectionStart === input.selectionEnd;
    const caret = collapsed ? input.selectionStart : -1;
    const a = analyse(line);
    S.ghost = computeGhost(line, caret);
    const atEnd = caret === line.length;
    mirror.replaceChildren(...segmentNodes(segments(line, a, collapsed ? input.selectionStart : -1),
      S.focused || !collapsed ? caret : line.length, atEnd && S.ghost ? S.ghost.text : null));
    mirror.classList.toggle("gc-unfocused", !S.focused);
    mirror.scrollTop = input.scrollTop;
    renderSignature(a, line, collapsed ? input.selectionStart : line.length);
  }
  function computeGhost(line, caret) {
    if (caret !== line.length || !line.trim()) return null;
    if (S.popup.open) {
      const item = S.popup.items[S.popup.active], ctx = S.popup.ctx;
      if (!item || !ctx || ctx.wordEnd !== caret) return null;
      const insert = item.insert.replace(/\s$/, "");
      const typed = line.slice(ctx.wordStart, caret);
      if (insert.length > typed.length && insert.startsWith(typed)) return { text: insert.slice(typed.length), item };
      return null;
    }
    for (let i = S.history.length - 1; i >= 0; i--) {
      const h = S.history[i];
      if (h.length > line.length && h.startsWith(line)) return { text: h.slice(line.length), history: h };
    }
    return null;
  }
  function renderSignature(a, line, caret) {
    signature.replaceChildren();
    if (!line.trim() || S.running) { signature.hidden = true; return; }
    signature.hidden = false;
    if (a.unknown && a.cmdToken && a.cmdToken.end !== caret) {
      signature.append(el("span", "gc-sig-err", `Unknown command “${a.cmdToken.value}”`), el("span", "gc-sig-dim", " · type help to list commands"));
      return;
    }
    if (a.builtin) { signature.append(el("span", "gc-sig-name", a.builtin.name), el("span", "gc-sig-dim", " — " + a.builtin.description)); return; }
    if (!a.command) { signature.append(el("span", "gc-sig-dim", "Press ⌃Space for suggestions")); return; }
    const cmd = a.command;
    signature.append(el("span", "gc-sig-name", cmd.name));
    const required = cmd.fields.filter(f => f.required);
    if (required.length) {
      for (const f of required) {
        const ok = a.used.has(f.flag);
        const chip = el("span", "gc-sig-flag " + (ok ? "gc-sig-ok" : "gc-sig-missing"), (ok ? "✓ " : "● ") + f.flag);
        chip.title = ok ? "provided" : "required — still missing";
        signature.append(chip);
      }
      const missing = required.filter(f => !a.used.has(f.flag)).length;
      signature.append(el("span", missing ? "gc-sig-count gc-sig-missing" : "gc-sig-count gc-sig-ok",
        missing ? `${missing} required missing` : "ready"));
    } else signature.append(el("span", "gc-sig-dim", " " + (cmd.description || "")));
    const current = a.tokens.filter(t => t.start <= caret).pop();
    let field = current && (current.field || current.owner);
    if (!field && a.expecting) field = a.expecting.field;
    if (field) {
      const info = el("div", "gc-sig-param");
      info.append(el("span", "gc-sig-param-flag", field.flag));
      if (field.help) info.append(el("span", null, " " + field.help));
      if (field.choices) info.append(el("span", "gc-sig-dim", " · " + field.choices.join(" | ")));
      if (!field.boolean && field.default !== null && field.default !== undefined && field.default !== "")
        info.append(el("span", "gc-sig-dim", " · default " + (Array.isArray(field.default) ? field.default.join(" ") : field.default)));
      signature.append(info);
    }
  }

  /* --------------------------------------------------------- completion ---- */
  function fuzzy(query, text) {
    if (!query) return { score: 1, idx: [] };
    const q = query.toLowerCase(), s = text.toLowerCase();
    const range = (a, n) => Array.from({ length: n }, (_, i) => a + i);
    if (s.startsWith(q)) return { score: 100 - text.length / 100, idx: range(0, q.length) };
    for (let i = 1; i < s.length; i++) {
      if ("-_ ./".includes(s[i - 1]) && s.startsWith(q, i)) return { score: 70 - i / 100, idx: range(i, q.length) };
    }
    const sub = s.indexOf(q);
    if (sub >= 0) return { score: 50 - sub / 100, idx: range(sub, q.length) };
    const idx = [];
    for (let i = 0, j = 0; i < s.length && j < q.length; i++) if (s[i] === q[j]) { idx.push(i); j++; }
    if (idx.length === q.length && q.length >= 3 && idx[idx.length - 1] - idx[0] < q.length * 3) return { score: 20 - (idx[idx.length - 1] - idx[0]) / 10, idx };
    return null;
  }

  function context() {
    const line = input.value, caret = input.selectionStart;
    if (caret !== input.selectionEnd) return null;
    const a = analyse(line);
    let ti = a.tokens.findIndex(t => caret >= t.start && caret <= t.end);
    let wordStart = caret, wordEnd = caret, word = "";
    if (ti >= 0) {
      const t = a.tokens[ti];
      wordStart = t.start; wordEnd = t.end;
      const partial = tokenize(line.slice(t.start, caret))[0];
      word = partial ? partial.value : "";
    } else ti = a.tokens.filter(t => t.end < caret).length;
    const before = a.tokens.slice(0, ti);
    const ctx = { line, caret, wordStart, wordEnd, word, a, before };
    const k = before[0] && !before[0].quoted && before[0].value === "geocrop" ? 1 : 0;
    if (before.length === k) return { ...ctx, kind: "command" };
    const name = before[k].value;
    const b = builtin(name);
    if (b) return b.name === "help" || b.name === "?" ? (before.length === k + 1 ? { ...ctx, kind: "help" } : null) : null;
    const command = S.byName.get(name);
    if (!command) return null;
    ctx.command = command;
    // Flag/value state just before the word.
    let expecting = null;
    for (const t of before.slice(k + 1)) {
      if (!t.quoted && t.value.startsWith("-") && t.value.length > 1 && !(expecting && NUMBER.test(t.value))) {
        const field = command.fields.find(f => f.flag === t.value) || null;
        expecting = field && !field.boolean ? { field, count: 0 } : null;
      } else if (expecting) { expecting.count++; if (!expecting.field.multiple) expecting = null; }
    }
    const raw = line.slice(wordStart, caret);
    const eq = raw.indexOf("=");
    if (raw.startsWith("-") && eq > 0) {
      const field = command.fields.find(f => f.flag === raw.slice(0, eq));
      if (!field || field.boolean) return null;
      return { ...ctx, kind: "value", field, wordStart: wordStart + eq + 1, word: tokenize(raw.slice(eq + 1))[0]?.value || "", alsoFlags: false };
    }
    if (raw.startsWith("-") && !(expecting && NUMBER.test(word))) return { ...ctx, kind: "flags" };
    if (expecting) return { ...ctx, kind: "value", field: expecting.field, alsoFlags: expecting.field.multiple && expecting.count > 0 };
    return { ...ctx, kind: "flags" };
  }

  function flagItems(ctx) {
    const query = ctx.word.replace(/^-+/, "");
    const items = [];
    for (const f of ctx.command.fields) {
      const usedHere = ctx.a.tokens.some(t => t.flag === f.flag && !(t.start === ctx.wordStart));
      if (usedHere && !f.multiple) continue;
      const m = fuzzy(query, f.flag.replace(/^-+/, ""));
      if (!m) continue;
      const lead = f.flag.length - f.flag.replace(/^-+/, "").length;
      let detail = f.boolean ? "switch" : f.choices ? f.choices.slice(0, 4).join("|") + (f.choices.length > 4 ? "|…" : "") : "";
      if (!f.boolean && f.default !== null && f.default !== undefined && f.default !== "" && !f.choices)
        detail = "= " + (Array.isArray(f.default) ? f.default.join(" ") : f.default);
      items.push({ kind: "flag", label: f.flag, insert: f.flag + " ", idx: m.idx.map(i => i + lead), score: m.score + (f.required ? 5 : 0),
        detail, doc: f.help || "", required: f.required, field: f, chain: !f.boolean });
    }
    return items.sort((x, y) => y.score - x.score);
  }
  function commandItems(word, helpArg) {
    const items = [];
    const pool = helpArg ? S.commands : [...S.commands, ...BUILTINS.map(b => ({ ...b, builtin: true }))];
    for (const c of pool) {
      const m = fuzzy(word, c.name);
      if (!m) continue;
      items.push({ kind: c.builtin ? "builtin" : "command", label: c.name, insert: c.name + " ", idx: m.idx, score: m.score,
        detail: c.builtin ? "built-in" : "", doc: c.description || "", chain: !c.builtin && !helpArg });
    }
    return items.sort((x, y) => y.score - x.score || x.label.localeCompare(y.label));
  }
  function choiceItems(ctx) {
    const items = [];
    for (const value of ctx.field.choices) {
      const m = fuzzy(ctx.word, String(value));
      if (m) items.push({ kind: "choice", label: String(value), insert: quoteIfNeeded(String(value)) + " ", idx: m.idx, score: m.score,
        detail: String(value) === String(ctx.field.default) ? "default" : "", doc: ctx.field.help || "" });
    }
    return items.sort((x, y) => y.score - x.score);
  }
  function fileItems(ctx, entries) {
    const slash = ctx.word.lastIndexOf("/");
    const base = ctx.word.slice(slash + 1);
    return entries.map(e => {
      const m = fuzzy(base, e.name) || { idx: [] };
      const dirLen = e.path.length - e.name.length;
      const insert = e.is_dir ? quoteIfNeeded(e.path + "/").replace(/"$/, "") : quoteIfNeeded(e.path) + " ";
      return { kind: e.is_dir ? "dir" : "file", label: e.path + (e.is_dir ? "/" : ""), insert,
        idx: m.idx.map(i => i + dirLen), score: 1, detail: e.is_dir ? "folder" : extension(e.name), doc: ctx.field.help || "", chain: e.is_dir };
    });
  }
  function extension(name) { const i = name.lastIndexOf("."); return i > 0 ? name.slice(i + 1).toLowerCase() : "file"; }

  function fetchFiles(prefix) {
    const hit = S.fileCache.get(prefix);
    if (hit && Date.now() - hit.time < 4000) return Promise.resolve(hit.entries);
    return api("/api/files?prefix=" + encodeURIComponent(prefix)).then(r => {
      S.fileCache.set(prefix, { time: Date.now(), entries: r.entries || [] });
      if (S.fileCache.size > 100) S.fileCache.delete(S.fileCache.keys().next().value);
      return r.entries || [];
    }).catch(() => []);
  }

  function suggest(explicit) {
    if (!S.connected || S.running || S.search) return closePopup();
    const ctx = context();
    if (!ctx) return closePopup();
    if (!explicit && ctx.kind === "command" && !ctx.word) return closePopup();
    const seq = ++S.popup.seq;
    let items = [];
    if (ctx.kind === "command") items = commandItems(ctx.word, false);
    else if (ctx.kind === "help") items = commandItems(ctx.word, true);
    else if (ctx.kind === "flags") items = flagItems(ctx);
    else if (ctx.kind === "value") {
      const f = ctx.field;
      if (f.choices) items = choiceItems(ctx);
      else if (isPathFlag(f) || /[/]/.test(ctx.word) || ctx.word.startsWith(".")) {
        const prefix = ctx.word.replace(/^\.\//, "");
        if (!/^(\/|~|[A-Za-z]:)/.test(prefix) && !prefix.split("/").some(p => p === ".." || (p.startsWith(".") && p !== "."))) {
          fetchFiles(prefix).then(entries => {
            if (seq !== S.popup.seq) return;
            let list = fileItems(ctx, entries);
            if (ctx.alsoFlags) list = list.concat(flagItems({ ...ctx, word: "" }));
            show(list, ctx, explicit);
          });
          if (S.popup.open && S.popup.ctx && S.popup.ctx.kind === "value") return;  // keep old list until files arrive
          return closePopup(true);
        }
      }
      if (ctx.alsoFlags && !ctx.word) items = items.concat(flagItems({ ...ctx, word: "" }));
    }
    show(items, ctx, explicit);
  }
  function show(items, ctx, explicit) {
    items = items.slice(0, 200);
    if (!items.length) { closePopup(); if (explicit) announce("No suggestions"); return; }
    const sameList = S.popup.open && S.popup.ctx && S.popup.ctx.kind === ctx.kind && S.popup.ctx.wordStart === ctx.wordStart;
    const prevLabel = sameList ? S.popup.items[S.popup.active]?.label : null;
    S.popup.items = items;
    S.popup.ctx = ctx;
    const keep = prevLabel ? items.findIndex(i => i.label === prevLabel) : -1;
    S.popup.active = S.popup.touched && keep >= 0 ? keep : 0;
    if (!sameList) S.popup.touched = false;
    S.popup.open = true;
    renderPopup();
    announce(`${items.length} suggestion${items.length === 1 ? "" : "s"}`);
    render();
  }
  function closePopup(silent) {
    if (S.popup.open || !popup.hidden) {
      S.popup.open = false; S.popup.items = []; S.popup.touched = false;
      popup.hidden = true;
      input.setAttribute("aria-expanded", "false");
      input.removeAttribute("aria-activedescendant");
      if (!silent) render();
    }
  }
  const KIND_ICON = { command: "⌘", builtin: "◆", flag: "⚑", choice: "≡", dir: "▸", file: "▫" };
  function renderPopup() {
    listbox.replaceChildren();
    S.popup.items.forEach((item, i) => {
      const li = el("li", "gc-opt gc-opt-" + item.kind);
      li.id = "gc-term-opt-" + i;
      li.setAttribute("role", "option");
      li.setAttribute("aria-selected", String(i === S.popup.active));
      const icon = el("span", "gc-opt-icon", KIND_ICON[item.kind] || "•");
      icon.setAttribute("aria-hidden", "true");
      const label = el("span", "gc-opt-label");
      const marks = new Set(item.idx);
      let run = "", runMark = false;
      const flush = () => { if (run) label.append(runMark ? el("mark", null, run) : document.createTextNode(run)); run = ""; };
      for (let c = 0; c < item.label.length; c++) {
        const m = marks.has(c);
        if (m !== runMark) { flush(); runMark = m; }
        run += item.label[c];
      }
      flush();
      li.append(icon, label);
      if (item.required) li.append(el("span", "gc-opt-req", "required"));
      const detail = el("span", "gc-opt-detail", item.detail || (item.kind === "command" ? shortDoc(item.doc) : ""));
      li.append(detail);
      li.addEventListener("mousedown", e => { e.preventDefault(); S.popup.active = i; accept(); });
      li.addEventListener("mousemove", () => { if (S.popup.active !== i) { S.popup.active = i; S.popup.touched = true; updateActive(); } });
      listbox.append(li);
    });
    popup.hidden = false;
    input.setAttribute("aria-expanded", "true");
    updateActive();
    positionPopup();
  }
  function shortDoc(doc) { return doc && doc.length > 38 ? doc.slice(0, 37) + "…" : doc || ""; }
  function updateActive() {
    const items = listbox.children;
    for (let i = 0; i < items.length; i++) {
      const on = i === S.popup.active;
      items[i].classList.toggle("gc-active", on);
      items[i].setAttribute("aria-selected", String(on));
    }
    const active = items[S.popup.active];
    if (active) {
      input.setAttribute("aria-activedescendant", active.id);
      const top = active.offsetTop, bottom = top + active.offsetHeight;
      if (top < listbox.scrollTop) listbox.scrollTop = top;
      else if (bottom > listbox.scrollTop + listbox.clientHeight) listbox.scrollTop = bottom - listbox.clientHeight;
    }
    const item = S.popup.items[S.popup.active];
    docPane.replaceChildren();
    if (item && item.doc) {
      docPane.append(el("strong", null, item.label), document.createTextNode(" " + item.doc));
      docPane.hidden = false;
    } else docPane.hidden = true;
    render();
  }
  function positionPopup() {
    if (popup.hidden) return;
    const caretNode = mirror.querySelector(".gc-caret") || mirror;
    const rootRect = term.getBoundingClientRect();
    const c = caretNode.getBoundingClientRect();
    const charW = c.width || 8;
    const ctx = S.popup.ctx;
    const back = ctx ? Math.max(0, ctx.caret - ctx.wordStart) : 0;
    const width = Math.min(460, rootRect.width - 16);
    popup.style.width = width + "px";
    let left = c.left - rootRect.left - back * charW - 30;
    left = Math.max(8, Math.min(left, rootRect.width - width - 8));
    popup.style.left = left + "px";
    const below = window.innerHeight - c.bottom, above = c.top;
    const h = popup.offsetHeight;
    if (below < h + 12 && above > below) {
      popup.style.top = (c.top - rootRect.top - h - 6) + "px";
      popup.classList.add("gc-above");
    } else {
      popup.style.top = (c.bottom - rootRect.top + 6) + "px";
      popup.classList.remove("gc-above");
    }
  }
  function accept(index) {
    const item = S.popup.items[index ?? S.popup.active], ctx = S.popup.ctx;
    if (!item || !ctx) return false;
    const line = input.value;
    let insert = item.insert, skip = 0;
    if (insert.endsWith(" ") && line[ctx.wordEnd] === " ") { insert = insert.slice(0, -1); skip = 1; }
    setLine(line.slice(0, ctx.wordStart) + insert + line.slice(ctx.wordEnd), ctx.wordStart + insert.length + skip);
    closePopup(true);
    if (item.chain) suggest(false); else render();
    return true;
  }
  function setLine(value, caret) {
    input.value = value;
    const pos = caret ?? value.length;
    input.setSelectionRange(pos, pos);
    render();
  }

  /* ----------------------------------------------------------- execute ---- */
  function echo(line) {
    const row = el("div", "gc-line gc-echo");
    row.append(makePrompt(), highlighted(line));
    append(row);
  }
  async function execute(raw) {
    const line = raw.replace(/\s+$/, "");
    closePopup(true);
    echo(line);
    setLine("");
    S.histIndex = S.history.length;
    S.draft = "";
    if (!line.trim()) return;
    if (S.history[S.history.length - 1] !== line) { S.history.push(line); if (S.history.length > HISTORY_MAX) S.history.shift(); saveHistory(); }
    S.histIndex = S.history.length;
    const tokens = tokenize(line).map(t => t.value);
    if (tokens[0] === "geocrop") tokens.shift();
    const name = tokens[0];
    if (!name) return;
    if (name === "help" || name === "?") return helpCommand(tokens[1]);
    if (name === "clear") return clearScreen();
    if (name === "history") return historyCommand();
    if (name === "jobs") return jobsCommand();
    if (!S.connected) { print("geocrop: not connected — enter the access token first", "gc-err"); return; }
    if (!S.byName.has(name)) {
      print(`geocrop: command not found: ${name}`, "gc-err");
      const close = S.commands.map(c => ({ name: c.name, d: distance(name, c.name) })).filter(c => c.d <= Math.max(2, name.length / 3))
        .sort((x, y) => x.d - y.d).slice(0, 3).map(c => c.name);
      if (close.length) print(`Did you mean: ${close.join(", ")}?`, "gc-dim");
      return;
    }
    await runJob(line);
  }
  function distance(a, b) {
    const row = Array.from({ length: b.length + 1 }, (_, i) => i);
    for (let i = 1; i <= a.length; i++) {
      let prev = row[0];
      row[0] = i;
      for (let j = 1; j <= b.length; j++) {
        const tmp = row[j];
        row[j] = Math.min(row[j] + 1, row[j - 1] + 1, prev + (a[i - 1] === b[j - 1] ? 0 : 1));
        prev = tmp;
      }
    }
    return row[b.length];
  }
  function clearScreen() { output.replaceChildren(); scrollToEnd(); }
  function helpCommand(topic) {
    if (topic && S.byName.has(topic)) {
      const c = S.byName.get(topic);
      const head = el("div", "gc-line gc-help-head");
      head.append(el("span", "gc-tk-cmd", c.name), el("span", "gc-dim", "  " + (c.description || "")));
      append(head);
      const table = el("div", "gc-help-flags");
      for (const f of c.fields) {
        table.append(el("span", "gc-tk-flag", f.flag),
          el("span", f.required ? "gc-help-req" : "gc-dim", f.required ? "required" : f.boolean ? "switch" : f.choices ? f.choices.join("|") : (f.default ?? "") === "" ? "" : "= " + f.default),
          el("span", "gc-help-text", f.help || ""));
      }
      append(table);
      return;
    }
    if (topic) { print(`help: no such command: ${topic}`, "gc-err"); return; }
    print("Built-ins", "gc-help-section");
    const builtins = el("div", "gc-help-grid");
    for (const b of BUILTINS) { const cell = el("span", "gc-tk-builtin", b.name); cell.title = b.description; builtins.append(cell); }
    append(builtins);
    print(S.commands.length ? `Engine commands (${S.commands.length})` : "Engine commands appear after connecting.", "gc-help-section");
    const grid = el("div", "gc-help-grid");
    for (const c of S.commands) { const cell = el("span", "gc-tk-cmd", c.name); cell.title = c.description || ""; grid.append(cell); }
    append(grid);
    print("help <command> lists its flags · Tab completes commands, flags, choices and workspace paths", "gc-dim");
  }
  function historyCommand() {
    if (!S.history.length) { print("history is empty", "gc-dim"); return; }
    const start = Math.max(0, S.history.length - HISTORY_MAX);
    S.history.slice(start).forEach((h, i) => {
      const row = el("div", "gc-line");
      row.append(el("span", "gc-hist-n", String(start + i + 1).padStart(4) + "  "), highlighted(h));
      append(row);
    });
  }
  async function jobsCommand() {
    if (!S.connected) { print("jobs: not connected", "gc-err"); return; }
    try {
      const { jobs } = await api("/api/jobs?limit=15");
      if (!jobs.length) { print("no jobs yet", "gc-dim"); return; }
      const table = el("div", "gc-jobs");
      for (const j of jobs) {
        const ok = j.status === "succeeded", bad = ["failed", "interrupted"].includes(j.status);
        table.append(el("span", "gc-dim", j.id.slice(0, 8)),
          el("span", "gc-job-status gc-job-" + j.status, (ok ? "✓ " : bad ? "✗ " : j.status === "cancelled" ? "⊘ " : "◌ ") + j.status),
          el("span", "gc-dim", new Date(j.created).toLocaleTimeString()));
        const cmd = el("span", "gc-job-cmd");
        cmd.append(highlighted(j.argv.map(quoteIfNeeded).join(" ")));
        table.append(cmd);
      }
      append(table);
    } catch (e) { print("jobs: " + e.message, "gc-err"); }
  }

  /* ----------------------------------------------------- job streaming ---- */
  async function runJob(line) {
    let job;
    try { job = await api("/api/jobs", { method: "POST", body: JSON.stringify({ command: line }) }); }
    catch (e) { print("geocrop: " + e.message, "gc-err"); status(false, "rejected by the server", null); return; }
    const block = el("pre", "gc-job-output");
    const tail = document.createTextNode("");
    block.append(tail);
    const waiting = print("◌ queued — waiting for the worker…", "gc-dim gc-waiting");
    append(block);
    S.running = { id: job.id, name: job.argv[0], offset: 0, tail: "", tailNode: tail, block, cr: false, trace: false,
      t0: performance.now(), waiting, errors: 0, interrupted: false, lines: 0 };
    try { window.GeoCropApp?.selectJob?.(job.id); } catch { /* app optional */ }
    setRunning(true);
    pump();
  }
  function setRunning(on) {
    term.classList.toggle("gc-busy", on);
    inputLine.hidden = on;
    badge.textContent = on && S.running ? "● " + S.running.name : "";
    if (on) { closePopup(true); signature.hidden = true; screen.focus({ preventScroll: true }); }
    else { input.focus({ preventScroll: true }); render(); }
    scrollToEnd();
  }
  async function pump() {
    const r = S.running;
    if (!r) return;
    let delay = 350;
    try {
      const logs = await api(`/api/jobs/${r.id}/logs?offset=${r.offset}`);
      if (S.running !== r) return;
      if (logs.text) {
        if (r.waiting) { r.waiting.remove(); r.waiting = null; }
        appendJobText(r, logs.text);
        r.offset = logs.next_offset;
        delay = 60;
      } else {
        const job = await api(`/api/jobs/${r.id}`);
        if (S.running !== r) return;
        if (job.status === "running" && r.waiting) { r.waiting.remove(); r.waiting = null; }
        if (TERMINAL_STATES.has(job.status)) {
          const more = await api(`/api/jobs/${r.id}/logs?offset=${r.offset}`);
          if (more.text) { appendJobText(r, more.text); r.offset = more.next_offset; delay = 0; }
          else return finish(r, job);
        }
      }
      r.errors = 0;
    } catch (e) {
      if (++r.errors > 5) { print("geocrop: lost contact with the engine (" + e.message + ")", "gc-err"); return finish(r, null); }
      delay = 1000;
    }
    r.timer = setTimeout(pump, delay);
  }
  function appendJobText(r, chunk) {
    const stick = nearBottom();
    // eslint-disable-next-line no-control-regex
    chunk = chunk.replace(/\x1b\[[0-?]*[ -/]*[@-~]/g, "").replace(/\x1b[@-_]/g, "");
    for (const part of chunk.split(/(\r\n|\n|\r)/)) {
      if (part === "\n" || part === "\r\n") commitLine(r);
      else if (part === "\r") r.cr = true;
      else if (part) { if (r.cr) { r.tail = ""; r.cr = false; } r.tail += part; }
    }
    r.tailNode.textContent = r.tail;
    if (stick) scrollToEnd();
  }
  function commitLine(r) {
    r.cr = false;
    const text = r.tail;
    let cls = "";
    if (/^Traceback \(most recent call last\)/.test(text)) { r.trace = true; cls = "gc-out-err"; }
    else if (r.trace) { cls = "gc-out-err"; if (!/^\s/.test(text)) r.trace = false; }
    else if (/^(\w+(Error|Exception)\b|error:|usage:|ERROR\b|FAILED\b)/.test(text)) cls = "gc-out-err";
    else if (/\b(warning|warn)\b/i.test(text)) cls = "gc-out-warn";
    r.block.insertBefore(el("span", cls || null, text + "\n"), r.tailNode);
    r.tail = "";
    if (++r.lines > 5000) { r.block.firstChild.remove(); r.lines--; }
  }
  function finish(r, job) {
    clearTimeout(r.timer);
    if (r.tail) commitLine(r);
    if (r.waiting) r.waiting.remove();
    if (!r.block.textContent.trim()) r.block.remove();
    let ms = performance.now() - r.t0;
    if (job && job.started && job.finished) ms = Date.parse(job.finished) - Date.parse(job.started);
    if (!job) status(false, "unknown", ms, r.id);
    else if (job.status === "succeeded") status(true, `exit ${job.returncode ?? 0}`, ms, r.id);
    else if (job.status === "cancelled") status(false, "cancelled", ms, r.id, "gc-status-warn");
    else status(false, `${job.status} · exit ${job.returncode ?? "?"}`, ms, r.id);
    S.running = null;
    setRunning(false);
  }
  function status(ok, text, ms, id, extra) {
    const row = el("div", "gc-line gc-status " + (ok ? "gc-status-ok" : "gc-status-err") + (extra ? " " + extra : ""));
    row.append(el("span", "gc-status-mark", ok ? "✓" : "✗"), el("span", "gc-status-text", text));
    const meta = [ms !== null && ms !== undefined ? fmtDuration(ms) : "", id ? "job " + id.slice(0, 8) : ""].filter(Boolean);
    if (meta.length) row.append(el("span", "gc-status-meta", meta.join(" · ")));
    append(row);
  }
  async function interrupt() {
    const r = S.running;
    if (r) {
      if (r.interrupted) return;
      r.interrupted = true;
      print("^C", "gc-ctrlc");
      try { await api(`/api/jobs/${r.id}/cancel`, { method: "POST" }); } catch (e) { print("cancel: " + e.message, "gc-err"); }
      return;
    }
    const row = el("div", "gc-line gc-echo");
    const typed = highlighted(input.value);
    typed.append(el("span", "gc-ctrlc", "^C"));
    row.append(makePrompt(), typed);
    append(row);
    closePopup(true);
    S.histIndex = S.history.length;
    setLine("");
  }

  /* ------------------------------------------------- reverse search ---- */
  function startSearch() {
    closePopup(true);
    S.search = { draft: input.value, from: S.history.length, match: -1 };
    searchBar.hidden = false;
    searchInput.value = "";
    searchInfo.textContent = "";
    term.classList.add("gc-searching");
    searchInput.focus();
  }
  function searchFind(older) {
    const q = searchInput.value, s = S.search;
    let start = older && s.match >= 0 ? s.match - 1 : S.history.length - 1;
    if (!q) { s.match = -1; searchInfo.textContent = ""; setLine(s.draft); return; }
    for (let i = start; i >= 0; i--) {
      if (S.history[i].includes(q)) { s.match = i; setLine(S.history[i], S.history[i].indexOf(q)); searchInfo.textContent = ""; return; }
    }
    searchInfo.textContent = "no match";
  }
  function endSearch(keep, run) {
    if (!S.search) return;
    const draft = S.search.draft;
    S.search = null;
    searchBar.hidden = true;
    term.classList.remove("gc-searching");
    if (!keep) setLine(draft);
    input.focus();
    input.setSelectionRange(input.value.length, input.value.length);
    render();
    if (run) execute(input.value);
  }
  searchInput.addEventListener("input", () => searchFind(false));
  searchInput.addEventListener("keydown", e => {
    if (e.ctrlKey && e.key.toLowerCase() === "r") { e.preventDefault(); searchFind(true); }
    else if (e.key === "Enter") { e.preventDefault(); endSearch(true, true); }
    else if (e.key === "Escape" || (e.ctrlKey && ["g", "c"].includes(e.key.toLowerCase()))) { e.preventDefault(); endSearch(false); }
    else if (["ArrowRight", "ArrowLeft", "Tab", "End", "Home"].includes(e.key)) { e.preventDefault(); endSearch(true); }
  });
  searchInput.addEventListener("blur", () => { if (S.search) setTimeout(() => { if (S.search && document.activeElement !== searchInput) endSearch(true); }, 0); });

  /* ------------------------------------------------------------- keys ---- */
  input.addEventListener("keydown", e => {
    if (e.isComposing) return;
    const key = e.key, ctrl = e.ctrlKey && !e.metaKey && !e.altKey;
    const pop = S.popup.open;
    if (ctrl && (e.code === "Space" || key === " ")) { e.preventDefault(); suggest(true); return; }
    if (ctrl && key.toLowerCase() === "l") { e.preventDefault(); clearScreen(); return; }
    if (ctrl && key.toLowerCase() === "c" && input.selectionStart === input.selectionEnd) { e.preventDefault(); interrupt(); return; }
    if (ctrl && key.toLowerCase() === "r") { e.preventDefault(); startSearch(); return; }
    if (ctrl && key.toLowerCase() === "u") { e.preventDefault(); setLine(input.value.slice(input.selectionStart), 0); return; }
    if (pop && (key === "ArrowDown" || key === "ArrowUp" || key === "PageDown" || key === "PageUp")) {
      e.preventDefault();
      const n = S.popup.items.length, step = key.startsWith("Page") ? 8 : 1;
      const dir = key === "ArrowDown" || key === "PageDown" ? step : -step;
      S.popup.active = key.startsWith("Page") ? Math.max(0, Math.min(n - 1, S.popup.active + dir)) : (S.popup.active + dir + n) % n;
      S.popup.touched = true;
      updateActive();
      return;
    }
    if (key === "Tab" && !e.shiftKey) {
      e.preventDefault();
      if (pop) { accept(); return; }
      const ctx = context();
      if (!ctx) return;
      suggest(true);
      if (S.popup.open && S.popup.items.length === 1) accept();
      return;
    }
    if (key === "Escape") { if (pop) { e.preventDefault(); closePopup(); } return; }
    if ((key === "ArrowRight" || key === "End") && S.ghost && input.selectionStart === input.value.length && !e.shiftKey) {
      e.preventDefault();
      if (S.ghost.item) accept(S.popup.items.indexOf(S.ghost.item));
      else { setLine(S.ghost.history); closePopup(true); }
      return;
    }
    if (key === "Enter") {
      e.preventDefault();
      if (pop && !e.shiftKey) {
        const item = S.popup.items[S.popup.active], ctx = S.popup.ctx;
        const typed = ctx ? input.value.slice(ctx.wordStart, ctx.caret) : "";
        const complete = item && item.insert.replace(/\s$/, "") === typed;
        if (item && !complete && (S.popup.touched || ctx.word)) { accept(); return; }
      }
      if (S.running) return;
      execute(input.value);
      return;
    }
    if (!pop && (key === "ArrowUp" || key === "ArrowDown") && !e.shiftKey) {
      if (!S.history.length) return;
      e.preventDefault();
      if (key === "ArrowUp") {
        if (S.histIndex === 0) return;
        if (S.histIndex === S.history.length) S.draft = input.value;
        S.histIndex--;
        setLine(S.history[S.histIndex]);
      } else {
        if (S.histIndex >= S.history.length) return;
        S.histIndex++;
        setLine(S.histIndex === S.history.length ? S.draft : S.history[S.histIndex]);
      }
    }
  });
  input.addEventListener("input", () => {
    if (/[\r\n]/.test(input.value)) {
      const caret = input.selectionStart;
      input.value = input.value.replace(/\r?\n|\r/g, " ");
      input.setSelectionRange(caret, caret);
    }
    restartBlink();
    suggest(false);
    render();
  });
  const caretMoved = () => { if (document.activeElement !== input) return; restartBlink(); if (S.popup.open) suggest(false); render(); };
  input.addEventListener("click", caretMoved);
  input.addEventListener("keyup", e => { if (["ArrowLeft", "ArrowRight", "Home", "End"].includes(e.key)) caretMoved(); });
  document.addEventListener("selectionchange", () => { if (document.activeElement === input) render(); });
  input.addEventListener("scroll", () => { mirror.scrollTop = input.scrollTop; });
  input.addEventListener("focus", () => { S.focused = true; render(); });
  input.addEventListener("blur", () => {
    S.focused = false;
    setTimeout(() => { if (document.activeElement !== input) closePopup(); }, 0);
    render();
  });
  screen.addEventListener("keydown", e => {
    if (e.target !== screen) return;
    if (e.ctrlKey && e.key.toLowerCase() === "c") { e.preventDefault(); interrupt(); }
    else if (e.ctrlKey && e.key.toLowerCase() === "l") { e.preventDefault(); clearScreen(); }
  });
  term.addEventListener("focusin", () => term.classList.add("gc-focused"));
  term.addEventListener("focusout", e => { if (!term.contains(e.relatedTarget)) term.classList.remove("gc-focused"); });
  screen.addEventListener("mouseup", () => {
    const sel = window.getSelection();
    if (sel && !sel.isCollapsed) return;
    if (S.search) searchInput.focus();
    else if (!S.running && !input.disabled) input.focus({ preventScroll: true });
    else screen.focus({ preventScroll: true });
  });
  let blinkTimer = null;
  function restartBlink() {
    term.classList.add("gc-typing");
    clearTimeout(blinkTimer);
    blinkTimer = setTimeout(() => term.classList.remove("gc-typing"), 600);
  }
  window.addEventListener("resize", () => positionPopup());
  screen.addEventListener("scroll", () => positionPopup(), { passive: true });

  /* Focus the terminal when its tab becomes visible. */
  const panel = root.closest("[hidden], .tab-panel") || root.parentElement;
  if (panel && "MutationObserver" in window) {
    new MutationObserver(() => {
      if (!panel.hidden && panel.offsetParent !== null) {
        scrollToEnd();
        if (S.running) screen.focus({ preventScroll: true }); else if (!input.disabled) input.focus({ preventScroll: true });
      } else closePopup(true);
    }).observe(panel, { attributes: true, attributeFilter: ["hidden", "class", "style"] });
  }

  /* --------------------------------------------------------- lifecycle ---- */
  function disconnectedBanner() {
    input.disabled = true;
    print("Not connected. Enter the access token to start a session.", "gc-dim");
    render();
  }
  function connected(detail) {
    const commands = Array.isArray(detail?.commands) ? detail.commands : [];
    S.commands = commands.slice().sort((a, b) => a.name.localeCompare(b.name));
    S.byName = new Map(S.commands.map(c => [c.name, c]));
    S.health = detail?.health || null;
    S.connected = true;
    input.disabled = false;
    output.replaceChildren();
    let last = null;
    try { last = localStorage.getItem(LOGIN_KEY); localStorage.setItem(LOGIN_KEY, new Date().toISOString()); } catch { /* ignore */ }
    const when = last ? new Date(last) : new Date();
    const stamp = when.toLocaleString("en-US", { weekday: "short", month: "short", day: "2-digit", hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false }).replace(/,/g, "");
    print(`Last login: ${stamp} on ttys000`, "gc-dim");
    const banner = el("div", "gc-line gc-banner");
    banner.append(el("span", "gc-banner-name", "GeoCrop Engine"), el("span", "gc-banner-ver", S.health?.version ? " " + S.health.version : ""),
      el("span", "gc-dim", ` · ${S.commands.length} commands`));
    append(banner);
    const tip = el("div", "gc-line gc-dim");
    tip.append(document.createTextNode("Type "), el("span", "gc-tk-builtin", "help"), document.createTextNode(" to list commands, "),
      el("kbd", null, "Tab"), document.createTextNode(" to complete, "), el("kbd", null, "↑"), document.createTextNode(" for history."));
    append(tip);
    if (S.health?.workspace) {
      workspaceLabel.textContent = "\u200e" + S.health.workspace + "\u200e";
      workspaceLabel.title = "Workspace: " + S.health.workspace;
    }
    render();
    if (!root.closest("[hidden]")) input.focus({ preventScroll: true });
  }
  document.addEventListener("geocrop:connected", e => connected(e.detail));
  disconnectedBanner();
})();
