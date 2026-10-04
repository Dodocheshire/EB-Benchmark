#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
setup_args=(); data_args=(); skip_data=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) setup_args+=(--dry-run); data_args+=(--dry-run); shift ;;
        --skip-system) setup_args+=(--skip-system); shift ;;
        --skip-data) skip_data=1; shift ;;
        --conda-root) setup_args+=(--conda-root "$2"); data_args+=(--conda-root "$2"); shift 2 ;;
        --deps-root) setup_args+=(--deps-root "$2"); shift 2 ;;
        -h|--help) echo 'Usage: bash install.sh [--dry-run] [--skip-system] [--skip-data] [--conda-root PATH] [--deps-root PATH]'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
bash "$REPO/scripts/setup_ubuntu.sh" "${setup_args[@]}"
if [[ "$skip_data" == 0 ]]; then bash "$REPO/scripts/download_datasets.sh" "${data_args[@]}"; fi
