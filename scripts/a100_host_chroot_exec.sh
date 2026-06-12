#!/usr/bin/env bash
set -euo pipefail

ROOTFS=${A100_CHROOT_ROOTFS:-/rpool/data/subvol-125-disk-0}
WJY_SRC=${A100_CHROOT_WJY_SRC:-/hdZFS/subvol-125-disk-1}

if [[ $# -eq 0 ]]; then
  echo "usage: $0 <command> [args...]" >&2
  exit 2
fi

exec unshare -m -- bash -c '
set -euo pipefail
mount --make-rprivate /

ROOTFS=$1
WJY_SRC=$2
shift 2

mount --bind "$WJY_SRC" "$ROOTFS/wjy"
mount --rbind /dev "$ROOTFS/dev"
mount --rbind /proc "$ROOTFS/proc"
mount --rbind /sys "$ROOTFS/sys"
mount --bind /tmp "$ROOTFS/tmp"

exec chroot "$ROOTFS" /bin/bash -lc "
set -euo pipefail
export PATH=/wjy/conda-envs/cuvs-build-129/bin:/usr/local/cuda/bin:/usr/bin:/bin:\${PATH:-}
export LD_LIBRARY_PATH=/wjy/conda-envs/cuvs-build-129/lib:/wjy/conda-envs/cuvs-build-129/targets/x86_64-linux/lib:/usr/lib/x86_64-linux-gnu:\${LD_LIBRARY_PATH:-}
cd /wjy/cuvs
exec \"\$@\"
" bash "$@"
' bash "$ROOTFS" "$WJY_SRC" "$@"
