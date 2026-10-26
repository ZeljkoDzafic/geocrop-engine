#!/usr/bin/env bash
# Publish the presentation site (website/) to Cloudflare as static assets.
# Requires a Cloudflare login (`npx wrangler login`) or CLOUDFLARE_API_TOKEN. The site becomes PUBLIC.
set -euo pipefail
cd "$(dirname "$0")/.."
python tools/check_presentation_site.py
npx --yes wrangler deploy --assets website --name "${GEOCROP_SITE_NAME:-geocrop}" --compatibility-date 2026-10-01
