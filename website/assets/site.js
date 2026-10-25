"use strict";
const tabs = [...document.querySelectorAll("[data-interface]")];
const platforms = [...document.querySelectorAll("[data-platform]")];
for (const [index, tab] of platforms.entries()) {
  const select = () => {
    for (const item of platforms) {
      const active = item === tab;
      item.setAttribute("aria-selected", String(active)); item.tabIndex = active ? 0 : -1;
      document.getElementById(item.getAttribute("aria-controls")).hidden = !active;
    }
  };
  tab.addEventListener("click", select);
  tab.addEventListener("keydown", event => {
    if (["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) {
      event.preventDefault();
      const next = event.key === "Home" ? 0 : event.key === "End" ? platforms.length - 1 : (index + 1) % platforms.length;
      platforms[next].click(); platforms[next].focus();
    }
  });
}
function selectTab(tab, focus = false) {
  for (const item of tabs) {
    const selected = item === tab;
    item.setAttribute("aria-selected", String(selected));
    item.tabIndex = selected ? 0 : -1;
    document.getElementById(item.getAttribute("aria-controls")).hidden = !selected;
  }
  if (focus) tab.focus();
}
for (const [index, tab] of tabs.entries()) {
  tab.addEventListener("click", () => selectTab(tab));
  tab.addEventListener("keydown", event => {
    let next;
    if (event.key === "ArrowRight") next = (index + 1) % tabs.length;
    if (event.key === "ArrowLeft") next = (index - 1 + tabs.length) % tabs.length;
    if (event.key === "Home") next = 0;
    if (event.key === "End") next = tabs.length - 1;
    if (next !== undefined) { event.preventDefault(); selectTab(tabs[next], true); }
  });
}
const menu = document.getElementById("menu-toggle");
const nav = document.getElementById("main-nav");
function closeMenu() { menu.setAttribute("aria-expanded", "false"); nav.classList.remove("open"); }
menu.addEventListener("click", () => {
  const open = menu.getAttribute("aria-expanded") !== "true";
  menu.setAttribute("aria-expanded", String(open)); nav.classList.toggle("open", open);
});
for (const link of nav.querySelectorAll("a")) link.addEventListener("click", closeMenu);
document.addEventListener("keydown", event => { if (event.key === "Escape") closeMenu(); });

const dialog = document.getElementById("image-dialog");
let opener;
for (const button of document.querySelectorAll("[data-image]")) button.addEventListener("click", () => {
  opener = button;
  const image = document.getElementById("dialog-image");
  image.src = button.dataset.image; image.alt = button.dataset.caption;
  document.getElementById("dialog-caption").textContent = button.dataset.caption;
  dialog.showModal(); document.body.classList.add("modal-open"); document.getElementById("close-dialog").focus();
});
document.getElementById("close-dialog").addEventListener("click", () => dialog.close());
dialog.addEventListener("click", event => { if (event.target === dialog) dialog.close(); });
dialog.addEventListener("close", () => { document.body.classList.remove("modal-open"); if (opener) opener.focus(); });

document.getElementById("copy-install").addEventListener("click", async () => {
  const button = document.getElementById("copy-install");
  const status = document.getElementById("copy-status");
  try {
    await navigator.clipboard.writeText(document.getElementById("install-command").textContent.trim());
    button.textContent = "Copied ✓"; status.textContent = "Commands copied to clipboard.";
    setTimeout(() => { button.textContent = "Copy commands"; status.textContent = ""; }, 2500);
  } catch {
    status.textContent = "Select the command block to copy it manually.";
    const range = document.createRange(); range.selectNodeContents(document.getElementById("install-command"));
    const selection = window.getSelection(); selection.removeAllRanges(); selection.addRange(range);
  }
});
