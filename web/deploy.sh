#!/usr/bin/env bash
# Deploy the site to t01 (tiktoken.golia.jp).
#
# The box, TLS and the Caddy vhost are owned by goliajp/devops — the vhost is a
# `caddy_sites` row (id `tiktoken`) rendered into t01's managed Caddyfile, and
# the DNS CNAME is a `dns` row. Neither is ours to re-create here; this script
# only refreshes the static content those point at.
#
#   usage: web/deploy.sh [--check]
#     --check   build and verify only; do not upload
set -euo pipefail

cd "$(dirname "$0")"

HOST=t01
ROOT=/apps/tiktoken/web
URL=https://tiktoken.golia.jp

check_only=false
[[ "${1:-}" == "--check" ]] && check_only=true

echo "→ building"
npm run build

# The wasm carries all 17 vocabularies; a truncated or half-written upload
# would still serve HTTP 200, so pin the digest and compare it post-deploy.
wasm_local=$(ls dist/assets/*.wasm)
digest=$(shasum -a 256 "$wasm_local" | awk '{print $1}')
echo "→ wasm $(basename "$wasm_local") sha256=${digest:0:16}… ($(du -h "$wasm_local" | cut -f1))"

if $check_only; then
  echo "✓ build ok (--check: nothing uploaded)"
  exit 0
fi

echo "→ uploading to $HOST:$ROOT"
ssh "$HOST" "mkdir -p $ROOT"
# --delete keeps the target an exact mirror so stale hashed assets don't pile up.
rsync -a --delete dist/ "$HOST:$ROOT/"

echo "→ verifying $URL"
# Content-hashed filenames mean a stale CDN/browser cache can't mask a bad
# upload, but the origin can still be wrong — verify what the origin returns.
code=$(curl -s -o /dev/null -w '%{http_code}' "$URL/")
[[ "$code" == "200" ]] || { echo "✗ $URL returned $code"; exit 1; }

wasm_path=$(basename "$wasm_local")
ctype=$(curl -sSI "$URL/assets/$wasm_path" | tr -d '\r' | awk -F': ' 'tolower($1)=="content-type"{print $2}')
[[ "$ctype" == "application/wasm" ]] || { echo "✗ wasm content-type is '$ctype', expected application/wasm"; exit 1; }

live_digest=$(curl -s "$URL/assets/$wasm_path" | shasum -a 256 | awk '{print $1}')
[[ "$live_digest" == "$digest" ]] || { echo "✗ served wasm digest differs from the build"; exit 1; }

echo "✓ deployed — $URL (wasm verified byte-identical)"
