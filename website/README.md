# GeoCrop presentation website

An English, standalone static presentation of GeoCrop Engine. It adds a public-facing product overview
alongside the existing engine and web workspace; neither is replaced. The page runs without a build
tool, JavaScript framework, external font, analytics service or CDN.

## Preview locally

From the repository root:

```bash
python3 tools/check_presentation_site.py
python3 -m http.server 8080 --directory website
```

Open the local server on port 8080. This presentation server does not run the engine and needs no token.
The separate engine workspace is started with `geocrop serve`; see `docs/web-interface.md`.

## Contents

* English overview, workflow and evidence/limitation sections.
* Accessible interface tabs for the web workspace, CLI, QGIS and the Jupyter client.
* Actual web, terminal and QGIS screenshots, with keyboard-accessible enlarged views.
* Copyable main-branch setup commands and links to the existing documentation.
* The self-contained QGIS plugin ZIP, including its license and notices.
* Responsive desktop/mobile layout and reduced-motion support.

## Screenshot evidence

`assets/screenshots.json` records the source commit, capture method, files, checksums and data status.
Screenshots were taken from actual running applications, not generated UI mockups or fabricated logs:

* **Web:** Chromium viewing an authenticated local engine server with an existing successful synthetic job.
  The current localized web UI is shown as it runs; the presentation copy and captions are English.
* **CLI:** an actual Xfce Terminal window running `geocrop --version` and `geocrop train --help`.
* **QGIS:** the actual QGIS 3.40.6 desktop with the GeoCrop provider, synthetic class raster and parcel
  evidence GeoPackage. Normal native QGIS styles were configured for the demonstration.

No credentials, personal parcel data or third-party satellite imagery are included. All mapped/results
data shown are synthetic software checks. No external accuracy, model superiority, exchangeability
or legal compliance is claimed. QGIS/Jupyter and other named projects retain their own licenses and
trademarks; integration does not imply their endorsement.

## Publish

The complete `website/` folder can be uploaded to any static host. Relative asset paths support both
a domain root and a project subpath, including GitHub Pages. To use GitHub Pages, publish this directory
through your chosen Pages build/artifact workflow. No hosting service is provisioned or deployed by
this change. Source/docs links and installation commands target the public main branch.

For an offline copy, archive the complete directory rather than only `index.html`. The screenshot
viewer and tabs work locally; clipboard access may be restricted by the browser, in which case the
command block is selected for manual copying.

## Validate

`check_presentation_site.py` validates local assets, anchors, accessibility references, screenshot hashes
and the QGIS archive. Browser checks should cover desktop/mobile overflow, navigation, all interface
tabs, keyboard tab navigation, screenshot dialogs and plugin download integrity. These checks validate
the site, not scientific accuracy of the engine.
