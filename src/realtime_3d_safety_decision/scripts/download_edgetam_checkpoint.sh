#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT_REVISION="14b1b75185fee05a4e4ee1c797b2761d035c7ccf"
CHECKPOINT_SHA256="ed2d4850b8792c239689b043c47046ec239b6e808a3d9b6ae676c803fd8780df"
CHECKPOINT_SIZE="56116523"
CHECKPOINT_URL="https://huggingface.co/facebook/EdgeTAM/resolve/${CHECKPOINT_REVISION}/edgetam.pt"
CHECKPOINT_PATH="${1:-${EDGETAM_CHECKPOINT:-${ROOT_DIR}/models/edgetam/edgetam.pt}}"

mkdir -p "$(dirname "${CHECKPOINT_PATH}")"

if [[ -f "${CHECKPOINT_PATH}" ]]; then
  EXISTING_SHA256="$(sha256sum "${CHECKPOINT_PATH}" | awk '{print $1}')"
  if [[ "${EXISTING_SHA256}" == "${CHECKPOINT_SHA256}" ]]; then
    echo "Official EdgeTAM checkpoint already verified: ${CHECKPOINT_PATH}"
    exit 0
  fi
  echo "Existing checkpoint has the wrong checksum; refusing to overwrite it:" >&2
  echo "  ${CHECKPOINT_PATH}" >&2
  echo "  expected ${CHECKPOINT_SHA256}" >&2
  echo "  actual   ${EXISTING_SHA256}" >&2
  exit 1
fi

TEMP_PATH="$(mktemp "${CHECKPOINT_PATH}.download.XXXXXX")"
cleanup() {
  rm -f "${TEMP_PATH}"
}
trap cleanup EXIT

if command -v curl >/dev/null 2>&1; then
  curl --fail --location --retry 3 --output "${TEMP_PATH}" "${CHECKPOINT_URL}"
elif command -v wget >/dev/null 2>&1; then
  wget --tries=3 --output-document="${TEMP_PATH}" "${CHECKPOINT_URL}"
else
  echo "Install curl or wget to download the EdgeTAM checkpoint." >&2
  exit 1
fi

ACTUAL_SIZE="$(stat -c '%s' "${TEMP_PATH}")"
if [[ "${ACTUAL_SIZE}" != "${CHECKPOINT_SIZE}" ]]; then
  echo "EdgeTAM checkpoint size mismatch: expected ${CHECKPOINT_SIZE}, got ${ACTUAL_SIZE}" >&2
  exit 1
fi

ACTUAL_SHA256="$(sha256sum "${TEMP_PATH}" | awk '{print $1}')"
if [[ "${ACTUAL_SHA256}" != "${CHECKPOINT_SHA256}" ]]; then
  echo "EdgeTAM checkpoint checksum mismatch:" >&2
  echo "  expected ${CHECKPOINT_SHA256}" >&2
  echo "  actual   ${ACTUAL_SHA256}" >&2
  exit 1
fi

chmod 0644 "${TEMP_PATH}"
mv "${TEMP_PATH}" "${CHECKPOINT_PATH}"
trap - EXIT

echo "Downloaded official EdgeTAM checkpoint revision ${CHECKPOINT_REVISION}"
echo "Path: ${CHECKPOINT_PATH}"
echo "SHA256: ${CHECKPOINT_SHA256}"
