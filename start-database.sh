#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

set -a
source soulmem.env
set +a

exec surreal start \
  --user "$SURREAL_USERNAME" \
  --pass "$SURREAL_PASSWORD" \
  "$SURREAL_PATH"
