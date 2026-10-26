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

* English overview: the processing chain, the web workspace, validation (spatial CV, locked holdout,
  LORO/LOYO, conformal sets, AOA), reason-coded parcel decisions and S1+S2 markers, pretrained adapters,
  active learning, preflight/acceptance/reliability, plugins and the layered architecture.
* The design language of the web workspace (`src/geocrop/web_static/style.css`): the same colour tokens,
  system font stack, hairlines, translucent surfaces, inline SVG icons, light and dark appearance with a
  toggle (stored under the same `geocrop-theme` key), reduced motion and a phone layout.
* Actual workspace captures (light and dark map, New analysis sheet, command palette, terminal, phone)
  and the QGIS capture, with keyboard-accessible enlarged views.
* Accessible interface tabs (web, QGIS, Jupyter, REST `/api/v1`) and copyable install commands.
* The self-contained QGIS plugin ZIP, including its licence and notices.

The public EuroCrops Slovenia benchmark is quoted exactly as in `docs/benchmarks/eurocrops-si-2021-pomurje.v1.md`,
with its caveat: a public transfer task on declared labels, not Republic of Srpska data.

## Screenshot evidence

`assets/screenshots.json` records the source commit, capture method, files, checksums, data status and
attribution. Screenshots were taken from actual running applications, not generated UI mockups:

* **Web and terminal:** headless Chromium (Playwright, 1440×900 at 2×, phone 390×844) driving an
  authenticated local `geocrop serve` after a real `geocrop quicklook --bbox 19.10 44.72 19.16 44.76
  --start 2023-06-01 --end 2023-08-31` job and a synthetic `geocrop demo`. The only alteration is that the
  absolute workspace path label was replaced by `~/studies/semberija`. Images are lossy WebP.
  Imagery: contains modified Copernicus Sentinel data 2023; basemap Sentinel-2 cloudless 2023 by EOX.
* **QGIS:** the actual QGIS 3.40.6 desktop with the GeoCrop provider, synthetic class raster and parcel
  evidence GeoPackage. Normal native QGIS styles were configured for the demonstration.

No credentials or personal parcel data are included. The quicklook shows imagery only, not a classification;
no external accuracy, model superiority or legal compliance is claimed beyond the documented public benchmark.
QGIS/Jupyter and other named projects retain their own licences and trademarks; integration does not imply
their endorsement.

## Publish

The complete `website/` folder can be uploaded to any static host. Relative asset paths support both
a domain root and a project subpath, including GitHub Pages. Source/docs links and installation
commands target the public main branch.

### Cloudflare (current deployment)

The site is published as Cloudflare static assets at <https://geocrop.calm-silence-b9f5.workers.dev>.
After `npx wrangler login` (or with `CLOUDFLARE_API_TOKEN` set), run:

```bash
tools/deploy_website.sh
```

It runs the site check and uploads `website/`; `website/_headers` sets the security headers. Deployment
is manual and makes the page public.

### GitHub Pages (alternative)

1. Ensure `website/`, `tools/check_presentation_site.py` and `.github/workflows/pages.yml` are on `main`.
2. Open the repository **Settings → Pages → Build and deployment → Source** and select **GitHub Actions**.
3. Open **Actions → Deploy presentation to GitHub Pages → Run workflow**, select `main` and run it.
4. After the deployment succeeds, open <https://zeljkodzafic.github.io/geocrop-engine/>.

The workflow is manual only: rerun it after changes to the website on `main`. It checks local links,
screenshots and the bundled plugin before publishing. The repository's Actions settings must allow
GitHub Actions; any configured `github-pages` environment approval still applies. Adding the workflow
does not enable repository Pages settings or perform a deployment by itself.

GitHub Pages serves the static presentation, images and plugin download. Start the engine API/web
workspace separately on your own computer or server; Pages does not execute Python services.

For an offline copy, archive the complete directory rather than only `index.html`. The screenshot
viewer and tabs work locally; clipboard access may be restricted by the browser, in which case the
command block is selected for manual copying.

## Validate

`check_presentation_site.py` validates local assets, anchors, accessibility references, screenshot hashes
and the QGIS archive. Browser checks should cover desktop/mobile overflow, navigation, all interface
tabs, keyboard tab navigation, screenshot dialogs and plugin download integrity. These checks validate
the site, not scientific accuracy of the engine.
