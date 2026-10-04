#!/usr/bin/env bash
# Shared by setup/download scripts. No global shell configuration is modified.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DRY_RUN=0
CONDA_ROOT=""
run() {
    printf '+ '; printf '%q ' "$@"; printf '\n'
    if [[ "$DRY_RUN" != 1 ]]; then "$@"; fi
}
locate_conda() {
    if [[ -z "$CONDA_ROOT" ]]; then
        if [[ -n "${CONDA_EXE:-}" ]]; then CONDA_ROOT="$(dirname "$(dirname "$CONDA_EXE")")"
        elif command -v conda >/dev/null 2>&1; then CONDA_ROOT="$(conda info --base)"
        elif [[ -x "$HOME/miniforge3/bin/conda" ]]; then CONDA_ROOT="$HOME/miniforge3"
        elif [[ -x /opt/conda/bin/conda ]]; then CONDA_ROOT=/opt/conda
        else CONDA_ROOT="$HOME/miniforge3"; fi
    fi
    CONDA="$CONDA_ROOT/bin/conda"
}
conda_python() { run "$CONDA" run --no-capture-output -n "$1" python "${@:2}"; }
