#!/usr/bin/env bash
# Materialize default RealCam-Vid test subsets as symlink clip dirs.
# Source clips live in the archive tree (large); this script only links them.
#
# default_loop prefers clips_revisit (multi_revisit 481-frame schedules),
# then native clips_loop, then ping-pong clips_long.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
ARCH="${REALCAMVID_ARCHIVE:-$ROOT/artifacts/archive_pre_paper/bench/realcamvid}"
SUBSETS="$SCRIPT_DIR/subsets"

if [[ ! -d "$ARCH/clips" ]]; then
  echo "Archive not found: $ARCH/clips" >&2
  echo "Set REALCAMVID_ARCHIVE to the converted RealCam-Vid tree." >&2
  exit 1
fi

LOOP_IDS="$SUBSETS/default_loop.txt"

find_src() {
  local clip="$1"
  shift
  local pool
  for pool in "$@"; do
    if [[ -d "$ARCH/$pool/$clip" && -f "$ARCH/$pool/$clip/image.jpg" && -f "$ARCH/$pool/$clip/poses.npy" ]]; then
      echo "$ARCH/$pool/$clip"
      return 0
    fi
  done
  return 1
}

is_loop_id() {
  grep -Fxq "$1" "$LOOP_IDS"
}

link_subset() {
  local name="$1" list="$2"
  local dest="$SCRIPT_DIR/clips_${name}"
  rm -rf "$dest"
  mkdir -p "$dest"
  local n=0
  while IFS= read -r clip || [[ -n "$clip" ]]; do
    [[ -z "$clip" || "$clip" =~ ^# ]] && continue
    local src=""
    if [[ "$name" == "default_loop" ]] || { [[ "$name" == "default_all" ]] && is_loop_id "$clip"; }; then
      src="$(find_src "$clip" clips_revisit clips_loop clips_long clips || true)"
    else
      src="$(find_src "$clip" clips clips_long clips_loop || true)"
    fi
    if [[ -z "$src" ]]; then
      echo "[warn] skip missing clip: $clip" >&2
      continue
    fi
    ln -s "$src" "$dest/$clip"
    n=$((n + 1))
  done < "$list"
  echo "[$name] linked $n clips -> $dest"
}

link_subset "default_loop" "$SUBSETS/default_loop.txt"
link_subset "default_random" "$SUBSETS/default_random.txt"
link_subset "default_all" "$SUBSETS/default_all.txt"

echo "Done. Use clips_dir=$SCRIPT_DIR/clips_default_loop (or _random / _all)."
