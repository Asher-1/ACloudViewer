#!/usr/bin/env bash
# Reload NVIDIA kernel modules when Vulkan enters device-lost state.
#
# Cannot run from a desktop session: nvidia_drm holds the primary display.
# Use from a TTY (Ctrl+Alt+F3), log in, then:
#   sudo ./reload_nvidia_for_vulkan_bench.sh
#
# Safer first step (no display stop): kill stale GPU compute clients only.

set -euo pipefail

echo "== GPU compute clients =="
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv 2>/dev/null || true

if [[ "${1:-}" == "--kill-compute" ]]; then
    mapfile -t _pids < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')
    if ((${#_pids[@]})); then
        echo "Killing compute PIDs: ${_pids[*]}"
        kill -9 "${_pids[@]}" || true
        sleep 1
    fi
    nvidia-smi --query-compute-apps=pid --format=csv 2>/dev/null || true
    exit 0
fi

if [[ "${1:-}" == "--full" ]]; then
    if [[ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]]; then
        echo "ERROR: stop the display manager from a TTY first (gdm/lightdm/sddm)."
        echo "  sudo systemctl stop gdm   # or lightdm / sddm"
        exit 1
    fi
    echo "Unloading NVIDIA modules..."
    rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia
    echo "Loading NVIDIA modules..."
    modprobe nvidia
    modprobe nvidia_uvm
    echo "Done."
    exit 0
fi

cat <<'EOF'
Usage:
  ./reload_nvidia_for_vulkan_bench.sh --kill-compute   # safe: drop hung bench / ACloudViewer GPU use
  ./reload_nvidia_for_vulkan_bench.sh --full           # TTY only, after systemctl stop gdm

Why rmmod fails on desktop:
  nvidia_drm (refcnt ~9)  = X/Wayland compositor
  nvidia    (refcnt ~249) = all GL/Vulkan clients + driver state
EOF
