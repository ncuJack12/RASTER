#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
HOST_REPO=${A100_CHROOT_REPO:-$(cd -- "$SCRIPT_DIR/.." && pwd)}
ROOTFS=${A100_CHROOT_ROOTFS:-}
CHROOT_WORKDIR=${A100_CHROOT_WORKDIR:-/workspace/RASTER}
CHROOT_CONDA_PREFIX=${A100_CHROOT_CONDA_PREFIX:-}

if [[ -z "$ROOTFS" ]]; then
  echo "set A100_CHROOT_ROOTFS to the chroot rootfs path" >&2
  exit 2
fi

if [[ $# -eq 0 ]]; then
  echo "usage: $0 <command> [args...]" >&2
  exit 2
fi

exec unshare -m -- bash -c '
set -euo pipefail
mount --make-rprivate /

ROOTFS=$1
HOST_REPO=$2
CHROOT_WORKDIR=$3
CHROOT_CONDA_PREFIX=$4
shift 4

mkdir -p "$ROOTFS$CHROOT_WORKDIR"
mount --bind "$HOST_REPO" "$ROOTFS$CHROOT_WORKDIR"
mount --rbind /dev "$ROOTFS/dev"
mount --rbind /proc "$ROOTFS/proc"
mount --rbind /sys "$ROOTFS/sys"
mount --bind /tmp "$ROOTFS/tmp"

exec chroot "$ROOTFS" /bin/bash -lc "
set -euo pipefail
if [[ -n \"$CHROOT_CONDA_PREFIX\" ]]; then
  export PATH=\"$CHROOT_CONDA_PREFIX/bin:/usr/local/cuda/bin:/usr/bin:/bin:\${PATH:-}\"
  export LD_LIBRARY_PATH=\"$CHROOT_CONDA_PREFIX/lib:$CHROOT_CONDA_PREFIX/targets/x86_64-linux/lib:/usr/lib/x86_64-linux-gnu:\${LD_LIBRARY_PATH:-}\"
else
  export PATH=/usr/local/cuda/bin:/usr/bin:/bin:\${PATH:-}
  export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:\${LD_LIBRARY_PATH:-}
fi
cd \"$CHROOT_WORKDIR\"
exec \"\$@\"
" bash "$@"
' bash "$ROOTFS" "$HOST_REPO" "$CHROOT_WORKDIR" "$CHROOT_CONDA_PREFIX" "$@"
