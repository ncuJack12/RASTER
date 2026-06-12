#!/usr/bin/env bash
set -euo pipefail

TOKEN="05dab9941ca9418e8894"
BASE_URL="https://cloud.tsinghua.edu.cn"
MANIFEST_NAME="RASTER_DATASET_MANIFEST_20260612.json"

OUTPUT_ROOT="."
CACHE_DIR=".raster_data"
DATASETS=()
FORCE=0
KEEP_ARCHIVES=0
DOWNLOAD_ONLY=0
LIST_ONLY=0

usage() {
  cat <<'USAGE'
Usage:
  scripts/download_raster_data.sh [options]

Options:
  --dataset NAME[,NAME...]   Download selected datasets. Can be repeated.
  --all                      Download all datasets in the manifest. This is the default.
  --list                     List datasets in the remote manifest and exit.
  --output-root DIR          Extract into DIR. Default: current directory.
  --cache-dir DIR            Store downloaded parts in DIR. Default: .raster_data.
  --download-only            Download and verify parts, but do not extract.
  --keep-archives            Keep downloaded split archives after extraction.
  --force                    Redownload files even if a local verified copy exists.
  -h, --help                 Show this help.

The extracted layout is:
  data/<dataset>/...
  generated_queries/order_range_raw_attr/<dataset>/...
USAGE
}

die() {
  echo "error: $*" >&2
  exit 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

urlencode_path() {
  python3 - "$1" <<'PY'
import sys
from urllib.parse import quote
print(quote(sys.argv[1], safe=""))
PY
}

download_shared_file() {
  local remote_name="$1"
  local local_path="$2"
  local encoded
  encoded="$(urlencode_path "/${remote_name}")"
  mkdir -p "$(dirname "$local_path")"
  curl --noproxy '*' -fL --retry 5 --retry-delay 3 -C - \
    --progress-bar \
    "${BASE_URL}/d/${TOKEN}/files/?p=${encoded}&dl=1" \
    -o "$local_path"
}

manifest_path() {
  echo "${CACHE_DIR}/${MANIFEST_NAME}"
}

load_manifest() {
  mkdir -p "$CACHE_DIR"
  local manifest
  manifest="$(manifest_path)"
  if [[ "$FORCE" -eq 1 || ! -s "$manifest" ]]; then
    echo "Downloading manifest: ${MANIFEST_NAME}"
    rm -f "$manifest"
    download_shared_file "$MANIFEST_NAME" "$manifest"
  fi
  python3 -m json.tool "$manifest" >/dev/null
}

list_datasets() {
  python3 - "$(manifest_path)" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
for name, info in sorted(manifest["datasets"].items()):
    part_count = len(info["parts"])
    size = sum(part["size"] for part in info["parts"])
    print(f"{name}\tparts={part_count}\tarchive_bytes={size}")
PY
}

selected_datasets() {
  python3 - "$(manifest_path)" "${DATASETS[@]}" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
available = sorted(manifest["datasets"])
raw = sys.argv[2:]
requested = []
for item in raw:
    requested.extend(part for part in item.replace(",", " ").split() if part)
if not requested:
    requested = available

bad = sorted(set(requested) - set(available))
if bad:
    raise SystemExit(f"unknown dataset(s): {', '.join(bad)}")
for name in requested:
    print(name)
PY
}

dataset_parts() {
  local dataset="$1"
  python3 - "$(manifest_path)" "$dataset" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
dataset = sys.argv[2]
for part in manifest["datasets"][dataset]["parts"]:
    print(f"{part['name']}\t{part['size']}\t{part['sha256']}")
PY
}

verify_file() {
  local path="$1"
  local expected_size="$2"
  local expected_sha="$3"
  [[ -f "$path" ]] || return 1
  local actual_size
  actual_size="$(stat -c '%s' "$path")"
  [[ "$actual_size" == "$expected_size" ]] || return 1
  local actual_sha
  actual_sha="$(sha256sum "$path" | awk '{print $1}')"
  [[ "$actual_sha" == "$expected_sha" ]]
}

download_dataset() {
  local dataset="$1"
  local dataset_cache="${CACHE_DIR}/${dataset}"
  local part_paths=()
  mkdir -p "$dataset_cache"

  echo "Dataset: ${dataset}"
  while IFS=$'\t' read -r name size sha; do
    local local_path="${dataset_cache}/${name}"
    part_paths+=("$local_path")
    if [[ "$FORCE" -eq 0 ]] && verify_file "$local_path" "$size" "$sha"; then
      echo "  ok: ${name}"
      continue
    fi
    echo "  downloading: ${name}"
    rm -f "$local_path"
    download_shared_file "$name" "$local_path"
    verify_file "$local_path" "$size" "$sha" || die "checksum failed: ${name}"
  done < <(dataset_parts "$dataset")

  if [[ "$DOWNLOAD_ONLY" -eq 1 ]]; then
    return
  fi

  mkdir -p "$OUTPUT_ROOT"
  echo "  extracting into: ${OUTPUT_ROOT}"
  cat "${part_paths[@]}" | zstd -dc | tar -xf - -C "$OUTPUT_ROOT"

  if [[ "$KEEP_ARCHIVES" -eq 0 ]]; then
    rm -f "${part_paths[@]}"
    rmdir "$dataset_cache" 2>/dev/null || true
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      [[ $# -ge 2 ]] || die "--dataset requires an argument"
      DATASETS+=("$2")
      shift 2
      ;;
    --all)
      DATASETS=()
      shift
      ;;
    --list)
      LIST_ONLY=1
      shift
      ;;
    --output-root)
      [[ $# -ge 2 ]] || die "--output-root requires an argument"
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --cache-dir)
      [[ $# -ge 2 ]] || die "--cache-dir requires an argument"
      CACHE_DIR="$2"
      shift 2
      ;;
    --download-only)
      DOWNLOAD_ONLY=1
      shift
      ;;
    --keep-archives)
      KEEP_ARCHIVES=1
      shift
      ;;
    --force)
      FORCE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

need_cmd curl
need_cmd python3
need_cmd sha256sum
need_cmd stat
need_cmd zstd
need_cmd tar

load_manifest

if [[ "$LIST_ONLY" -eq 1 ]]; then
  list_datasets
  exit 0
fi

mapfile -t SELECTED < <(selected_datasets)
for dataset in "${SELECTED[@]}"; do
  download_dataset "$dataset"
done

echo "Done."
