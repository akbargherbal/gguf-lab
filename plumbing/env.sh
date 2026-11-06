#!/usr/bin/env bash
# Shared environment for the plumbing scripts (source me).
#
#   . plumbing/env.sh
#
# BINARIES_GS points at the llama.cpp binary cache. The default is the prefix
# built by the music_captioning project: arch-tagged tarballs with a sha256 and
# a manifest, produced once on a CPU-only VM. Override it to use your own bucket
# or your own build prefix.

PLUMBING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$PLUMBING_DIR/.." && pwd)"
export PLUMBING_DIR REPO_ROOT

: "${BINARIES_GS:=gs://akbar-december-2024-backup/music_captioning/binaries}"
export BINARIES_GS

if command -v gcloud >/dev/null 2>&1; then
  gcs_cp()  { gcloud storage cp "$@"; }
  gcs_cat() { gcloud storage cat "$@"; }
  gcs_ls()  { gcloud storage ls "$@"; }
  export GCS_CLI=gcloud
elif command -v gsutil >/dev/null 2>&1; then
  gcs_cp()  { gsutil cp "$@"; }
  gcs_cat() { gsutil cat "$@"; }
  gcs_ls()  { gsutil ls "$@"; }
  export GCS_CLI=gsutil
else
  echo "plumbing/env.sh: need 'gcloud' or 'gsutil' on PATH" >&2
  export GCS_CLI=
fi
