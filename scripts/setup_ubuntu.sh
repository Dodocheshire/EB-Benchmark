#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/setup_common.sh"
SKIP_SYSTEM=0
DEPS_ROOT="${EB_DEPS_ROOT:-$REPO/.deps}"
MINIFORGE_VERSION=24.11.3-2
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --skip-system) SKIP_SYSTEM=1; shift ;;
        --conda-root) CONDA_ROOT="$2"; shift 2 ;;
        --deps-root) DEPS_ROOT="$2"; shift 2 ;;
        -h|--help) echo 'Usage: bash scripts/setup_ubuntu.sh [--dry-run] [--skip-system] [--conda-root PATH] [--deps-root PATH]'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || { echo 'Ubuntu x86_64 is required.' >&2; exit 1; }
if [[ "$SKIP_SYSTEM" == 0 ]]; then
    SUDO=(); [[ "$EUID" == 0 ]] || SUDO=(sudo)
    run "${SUDO[@]}" apt-get update
    run "${SUDO[@]}" apt-get install -y git git-lfs curl wget rsync tmux build-essential python3-dev \
        pciutils xorg x11-utils libopengl0 libgl1 libgl1-mesa-dri libglu1-mesa libegl1 libvulkan1 vulkan-tools \
        libx11-6 libxext6 libxi6 libxrender1 libxkbcommon-x11-0 libxcb-xinerama0 libxcb-randr0 \
        libxcb-shape0 libxcb-xfixes0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util0 fonts-ubuntu ffmpeg
fi
locate_conda
run mkdir -p "$DEPS_ROOT"
if [[ ! -x "$CONDA" ]]; then
    installer="$DEPS_ROOT/Miniforge3-$MINIFORGE_VERSION-Linux-x86_64.sh"
    url="https://github.com/conda-forge/miniforge/releases/download/$MINIFORGE_VERSION/$(basename "$installer")"
    run curl -fL --retry 5 -o "$installer" "$url"
    run curl -fL --retry 5 -o "$installer.sha256" "$url.sha256"
    if [[ "$DRY_RUN" == 0 ]]; then (cd "$DEPS_ROOT" && sha256sum -c "$(basename "$installer").sha256"); fi
    run bash "$installer" -b -p "$CONDA_ROOT"
fi
export CONDA_CHANNEL_PRIORITY=flexible
for environment in embench embench_nav embench_man; do
    if [[ "$DRY_RUN" == 0 ]] && "$CONDA" env list --json | python3 -c 'import json,sys,os; sys.exit(not any(os.path.basename(p)==sys.argv[1] for p in json.load(sys.stdin)["envs"]))' "$environment"; then
        echo "Reusing existing $environment"
    else
        run "$CONDA" create -y -n "$environment" --override-channels -c conda-forge \
            python=3.9 numpy=1.26.4 pip 'setuptools<81' wheel libstdcxx-ng libgcc-ng
    fi
    # CPU Torch is deliberate: models run remotely; simulator rendering still uses NVIDIA OpenGL.
    conda_python "$environment" -m pip install 'torch==2.4.0+cpu' 'torchvision==0.19.0+cpu' --index-url https://download.pytorch.org/whl/cpu
    conda_python "$environment" -m pip install -r "$REPO/requirements_remote.txt"
    conda_python "$environment" -m pip install -e "$REPO"
done
conda_python embench_nav -m pip install 'ai2thor==5.0.0' 'gym==0.26.2'
conda_python embench -m pip install 'ai2thor==2.1.0' 'gym==0.23.0' 'numpy-quaternion==2023.0.4' 'numba==0.59.1'
run "$CONDA" install -y -n embench --override-channels -c conda-forge -c aihabitat habitat-sim=0.3.0 withbullet headless

checkout_dependency() {
    local url="$1" path="$2" revision="$3"
    if [[ ! -d "$path/.git" ]]; then run git clone "$url" "$path"; fi
    if [[ "$DRY_RUN" == 0 ]] && ! git -C "$path" cat-file -e "$revision^{commit}" 2>/dev/null; then run git -C "$path" fetch origin "$revision"; fi
    run git -C "$path" checkout --detach "$revision"
}
checkout_dependency https://github.com/facebookresearch/habitat-lab.git "$DEPS_ROOT/habitat-lab" afe4058a7f8aa5ab71a133575cdaa79f0308af6a
conda_python embench -m pip install --no-deps -e "$DEPS_ROOT/habitat-lab/habitat-lab"

coppelia="$DEPS_ROOT/CoppeliaSim_Pro_V4_1_0_Ubuntu20_04"
if [[ ! -f "$coppelia/libcoppeliaSim.so" ]]; then
    archive="$DEPS_ROOT/CoppeliaSim_Pro_V4_1_0_Ubuntu20_04.tar.xz"
    run curl -fL --retry 5 -C - -o "$archive" https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Pro_V4_1_0_Ubuntu20_04.tar.xz
    run tar -xf "$archive" -C "$DEPS_ROOT"
fi
export COPPELIASIM_ROOT="$coppelia"
export LD_LIBRARY_PATH="$coppelia${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export QT_QPA_PLATFORM_PLUGIN_PATH="$coppelia"
run "$CONDA" install -y -n embench_man --override-channels -c conda-forge 'libtiff=4.3'
conda_python embench_man -m pip install 'gymnasium==1.0.0' 'ultralytics==8.3.61' 'open3d==0.18.0' 'pyquaternion==0.9.9' 'natsort==8.4.0' 'cffi==1.17.1' 'html-testRunner==1.2.1'
checkout_dependency https://github.com/stepjam/PyRep.git "$DEPS_ROOT/PyRep" 8f420be8064b1970aae18a9cfbc978dfb15747ef
conda_python embench_man -m pip install --no-build-isolation -e "$DEPS_ROOT/PyRep"
conda_python embench_man -m pip install -e "$REPO/embodiedbench/envs/eb_manipulation" --config-settings editable_mode=compat
run cp "$REPO/embodiedbench/envs/eb_manipulation/simAddOnScript_PyRep.lua" "$coppelia/"
if [[ "$DRY_RUN" == 0 ]]; then
    prefix="$("$CONDA" run -n embench_man python -c 'import sys; print(sys.prefix)')"
    mkdir -p "$prefix/etc/conda/activate.d" "$prefix/etc/conda/deactivate.d"
    # Restore previous variables when leaving embench_man; do not contaminate the other environments.
    {
        echo 'export _EB_OLD_COPPELIA=${COPPELIASIM_ROOT-} _EB_OLD_LD=${LD_LIBRARY_PATH-} _EB_OLD_QT=${QT_QPA_PLATFORM_PLUGIN_PATH-}'
        printf 'export COPPELIASIM_ROOT=%q\n' "$coppelia"
        echo 'export LD_LIBRARY_PATH="$COPPELIASIM_ROOT:$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"'
        echo 'export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"'
    } > "$prefix/etc/conda/activate.d/embodiedbench-simulator.sh"
    cat > "$prefix/etc/conda/deactivate.d/embodiedbench-simulator.sh" <<'EOF'
export COPPELIASIM_ROOT="${_EB_OLD_COPPELIA-}" LD_LIBRARY_PATH="${_EB_OLD_LD-}" QT_QPA_PLATFORM_PLUGIN_PATH="${_EB_OLD_QT-}"
unset _EB_OLD_COPPELIA _EB_OLD_LD _EB_OLD_QT
EOF
fi
if [[ "$DRY_RUN" == 0 ]]; then
    conda_python embench_nav -c 'import ai2thor, torch; from embodiedbench.planner.remote_model import RemoteModel; print("Navigation imports OK")'
    conda_python embench -c 'import habitat, habitat_sim, ai2thor; print("ALFRED/Habitat imports OK")'
    conda_python embench_man -c 'import pyrep, amsolver, ultralytics; from embodiedbench.envs.eb_manipulation.EBManEnv import EBManEnv; print("Manipulation imports OK")'
fi
echo 'Environment setup completed. Next: bash scripts/download_datasets.sh'
