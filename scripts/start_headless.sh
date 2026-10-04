#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/setup_common.sh"
DISPLAY_ID="${1:-1}"
[[ "$DISPLAY_ID" =~ ^[0-9]+$ ]] || { echo 'Display must be a nonnegative integer.' >&2; exit 2; }
locate_conda
if DISPLAY=":$DISPLAY_ID" xdpyinfo >/dev/null 2>&1; then echo "DISPLAY=:$DISPLAY_ID already available"; exit 0; fi
session="embench-xorg-$DISPLAY_ID"
if tmux has-session -t "$session" 2>/dev/null; then echo "Session $session already exists; inspect its log."; exit 0; fi
mkdir -p "$REPO/running/setup"
printf -v command 'cd %q && exec %q run --no-capture-output -n embench python -m embodiedbench.envs.eb_alfred.scripts.startx %q >> %q 2>&1' \
    "$REPO" "$CONDA" "$DISPLAY_ID" "$REPO/running/setup/xorg-$DISPLAY_ID.log"
tmux new-session -d -s "$session" "$command"
echo "Started $session; log: $REPO/running/setup/xorg-$DISPLAY_ID.log"
echo 'Xorg requires NVIDIA-device permission. Containers need graphics/display driver capabilities.'
