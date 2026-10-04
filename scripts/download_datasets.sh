#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/setup_common.sh"
DATASETS=all
CACHE="${EB_DATASET_CACHE:-$REPO/.cache/datasets}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --conda-root) CONDA_ROOT="$2"; shift 2 ;;
        --dataset) DATASETS="$2"; shift 2 ;;
        --cache-dir) CACHE="$2"; shift 2 ;;
        -h|--help) echo 'Usage: bash scripts/download_datasets.sh [--dataset all|alfred|habitat|manipulation] [--dry-run] [--conda-root PATH] [--cache-dir PATH]'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
case "$DATASETS" in all|alfred|habitat|manipulation) ;; *) echo 'Unknown dataset selection.' >&2; exit 2 ;; esac
locate_conda
if [[ "$DRY_RUN" == 0 ]]; then
    [[ -x "$CONDA" ]] || { echo 'Run scripts/setup_ubuntu.sh first.' >&2; exit 1; }
    command -v git-lfs >/dev/null || { echo 'Install git-lfs first.' >&2; exit 1; }
fi
run mkdir -p "$CACHE"
run git -C "$REPO" lfs install --local
run git -C "$REPO" lfs pull
fetch_dataset() {
    local name="$1" revision="$2" destination="$3" source_subdir="$4"
    local checkout="$CACHE/$name"
    # Keep the downloaded checkout even if copying fails. Never delete downloaded data.
    if [[ ! -d "$checkout/.git" ]]; then
        run env GIT_LFS_SKIP_SMUDGE=1 git clone "https://huggingface.co/datasets/EmbodiedBench/$name" "$checkout"
    fi
    if [[ "$DRY_RUN" == 0 ]] && ! git -C "$checkout" cat-file -e "$revision^{commit}" 2>/dev/null; then run git -C "$checkout" fetch origin "$revision"; fi
    run git -C "$checkout" checkout --detach "$revision"
    run git -C "$checkout" lfs pull
    run git -C "$checkout" lfs fsck
    if [[ -e "$destination" && ! -d "$destination" ]]; then
        echo "Destination is not a directory: $destination; preserved the checkout at $checkout" >&2; exit 1
    fi
    run mkdir -p "$destination"
    run rsync -a --exclude=.git "$checkout/$source_subdir/" "$destination/"
    if [[ "$DRY_RUN" == 0 ]]; then printf '%s\n' "$revision" > "$destination/.dataset_revision"; fi
}
if [[ "$DATASETS" == all || "$DATASETS" == alfred ]]; then
    fetch_dataset EB-ALFRED af55cc721725ed058830fd7a8aee66c1f69efbaa "$REPO/embodiedbench/envs/eb_alfred/data/json_2.1.0" .
fi
if [[ "$DATASETS" == all || "$DATASETS" == habitat ]]; then
    conda_python embench -m habitat_sim.utils.datasets_download --uids rearrange_task_assets --data-path "$REPO/embodiedbench/envs/eb_habitat/data"
fi
if [[ "$DATASETS" == all || "$DATASETS" == manipulation ]]; then
    fetch_dataset EB-Manipulation 4b6c210a3c7539330c2f0d228e778a9400061025 "$REPO/embodiedbench/envs/eb_manipulation/data" data
    if [[ "$DRY_RUN" == 0 ]]; then cd "$REPO"; fi
    conda_python embench_man -c 'from ultralytics import YOLO; YOLO("yolo11n.pt"); print("YOLO weights ready")'
fi
echo 'Datasets ready. Navigation task JSONs are included in the repository.'
echo 'AI2-THOR downloads its simulator build into ~/.ai2thor on the first environment launch.'
