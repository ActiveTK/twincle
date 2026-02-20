#!/bin/bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

log() { echo "[onstart] $*"; }

log "nvidia-smi:" || true
nvidia-smi || true

log "Installing dependencies"
apt-get update -y
apt-get install -y git curl build-essential pkg-config libssl-dev ca-certificates

log "Installing Rust"
if ! command -v cargo >/dev/null 2>&1; then
  curl https://sh.rustup.rs -sSf | sh -s -- -y
  source "$HOME/.cargo/env"
fi

log "Cloning repo"
rm -rf twincle
# shallow clone to save time
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/ActiveTK/twincle
cd twincle

log "Building"
cargo build --release

log "Benchmark"
./target/release/twincle --benchmark --benchmark-seconds 10 --limit 1000000000000

log "Done"
