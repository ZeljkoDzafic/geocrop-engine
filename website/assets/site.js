"use strict";
const $ = id => document.getElementById(id);

/* Appearance: system by default; the toggle stores an explicit choice (same key as the workspace). */
const root = document.documentElement;
const darkQuery = window.matchMedia("(prefers-color-scheme: dark)");
const isDark = () => root.dataset.theme ? root.dataset.theme === "dark" : darkQuery.matches;
function syncThemeButton() {
  const dark = isDark();
  $("theme-toggle").setAttribute("aria-pressed", String(dark));
  $("theme-toggle").setAttribute("aria-label", dark ? "Switch to light appearance" : "Switch to dark appearance");
}
$("theme-toggle").addEventListener("click", () => {
  const next = isDark() ? "light" : "dark";
  root.dataset.theme = next;
  try { localStorage.setItem("geocrop-theme", next); } catch { /* not persisted: fine */ }
  syncThemeButton();
});
darkQuery.addEventListener("change", syncThemeButton);
syncThemeButton();

/* Menu on narrow screens */
const menu = $("menu-toggle");
const nav = $("main-nav");
function closeMenu() { menu.setAttribute("aria-expanded", "false"); nav.classList.remove("open"); }
menu.addEventListener("click", () => {
  const open = menu.getAttribute("aria-expanded") !== "true";
  menu.setAttribute("aria-expanded", String(open)); nav.classList.toggle("open", open);
});
for (const link of nav.querySelectorAll("a")) link.addEventListener("click", closeMenu);
document.addEventListener("keydown", event => { if (event.key === "Escape") closeMenu(); });

/* Tab lists (interfaces, install platform) with arrow-key navigation */
for (const list of document.querySelectorAll('[role="tablist"]')) {
  const tabs = [...list.querySelectorAll('[role="tab"]')];
  const select = (tab, focus) => {
    for (const item of tabs) {
      const active = item === tab;
      item.setAttribute("aria-selected", String(active));
      item.tabIndex = active ? 0 : -1;
      $(item.getAttribute("aria-controls")).hidden = !active;
    }
    if (focus) tab.focus();
  };
  tabs.forEach((tab, index) => {
    tab.addEventListener("click", () => select(tab, false));
    tab.addEventListener("keydown", event => {
      const keys = { ArrowRight: index + 1, ArrowLeft: index - 1, Home: 0, End: tabs.length - 1 };
      if (!(event.key in keys)) return;
      event.preventDefault();
      select(tabs[(keys[event.key] + tabs.length) % tabs.length], true);
    });
  });
}

/* Screenshot viewer */
const dialog = $("image-dialog");
let opener;
for (const button of document.querySelectorAll("[data-image]")) button.addEventListener("click", () => {
  opener = button;
  let src = button.dataset.image;
  if (src.endsWith("web.webp") && isDark()) src = src.replace("web.webp", "web-dark.webp");
  const image = $("dialog-image");
  image.src = src;
  image.alt = [...button.querySelectorAll("img")].find(img => img.offsetWidth > 0)?.alt || button.dataset.caption;
  $("dialog-caption").textContent = button.dataset.caption;
  dialog.showModal(); document.body.classList.add("modal-open");
});
$("close-dialog").addEventListener("click", () => dialog.close());
dialog.addEventListener("click", event => { if (event.target === dialog) dialog.close(); });
dialog.addEventListener("close", () => { document.body.classList.remove("modal-open"); if (opener) opener.focus(); });

/* Copy install commands */
for (const button of document.querySelectorAll("[data-copy]")) button.addEventListener("click", async () => {
  const code = $(button.dataset.copy);
  const label = button.querySelector("span");
  try {
    await navigator.clipboard.writeText(code.textContent.trim());
    label.textContent = "Copied"; $("copy-status").textContent = "Commands copied to the clipboard.";
    setTimeout(() => { label.textContent = "Copy"; $("copy-status").textContent = ""; }, 2000);
  } catch {
    const range = document.createRange(); range.selectNodeContents(code);
    const selection = window.getSelection(); selection.removeAllRanges(); selection.addRange(range);
    $("copy-status").textContent = "Clipboard unavailable; the commands are selected for copying.";
  }
});
