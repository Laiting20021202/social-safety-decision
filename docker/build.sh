#!/usr/bin/env bash
set -euo pipefail

repository=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repository"
# Keep the two BuildKit sessions separate. Combined Compose Bake builds can
# reject non-ASCII checkout paths in their shared session key.
docker compose build control
docker compose --profile isaac build isaac
