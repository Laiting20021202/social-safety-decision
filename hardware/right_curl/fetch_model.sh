#!/usr/bin/env bash
# Reconstruct the v1.0 model used in both Docker backends, without accessing the NUC.
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
model=${OPENARM_DESCRIPTION_DIR:-$root/.openarm-vendor/nuc-v10-description}
mkdir -p "$(dirname "$model")"
if [[ -e "$model" ]]; then
  (cd "$model" && sha256sum --check --quiet "$here/site-v10.sha256") || {
    echo "Existing model differs from the recorded experiment: $model" >&2; exit 1;
  }
  exit 0
fi
work=$(mktemp -d "$(dirname "$model")/.chest-model.XXXXXXXX")
trap 'rm -rf -- "$work"' EXIT
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$work:/work" -v "$here:/recipe:ro" \
  --entrypoint bash "${OPENARM_FETCH_IMAGE:-social-safety-decision-control:humble}" -euc '
    cd /work
    git init -q
    git fetch -q --depth=1 https://github.com/enactic/openarm_description.git ee06e2eeb4b147ce7c5c34ab56b1120104a7f7c0
    git checkout -q --detach FETCH_HEAD
    git apply /recipe/site-v10.patch
    sha256sum --check --quiet /recipe/site-v10.sha256
  '
mv -T "$work" "$model"
