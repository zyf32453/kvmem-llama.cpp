#!/usr/bin/env bash
# Replay maintained diffs without commits or Git identity changes.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LLAMA="${KVMEM_LLAMA_DIR:-$ROOT/llama.cpp}"
PATCH="$ROOT/patches/llama-kvmem-current.patch"
GRAPH="$ROOT/patches/cuda-graph-decode.patch"
GDN_OUTPUT="$ROOT/patches/gdn-output-fusion.patch"
GRAPH_REACTIVATION="$ROOT/patches/cuda-graph-reactivation.patch"
RDNA2_FATTN="$ROOT/patches/0005-hip-rdna2-quantized-kv-fa-vec.patch"
QWEN3MOE_CAPTURE="$ROOT/patches/0006-qwen3moe-capture.patch"
HISTORICAL_ACTIVE="$ROOT/patches/0007-historical-active-hook.patch"
HISTORICAL_CELLS="$ROOT/patches/0008-historical-cell-index-reservation.patch"
HISTORICAL_SLOTS="$ROOT/patches/0009-historical-slot-info-return.patch"
HISTORICAL_MICROBATCH="$ROOT/patches/0010-historical-microbatch-pool.patch"
BUDGET_UPGRADE="$ROOT/patches/reasoning-budget-upgrade.patch"
REPLAY_UPGRADE="$ROOT/patches/replayssm-upgrade.patch"
UPGRADE="$ROOT/patches/multimodal-upgrade.patch"
cd "$LLAMA"

# An incremental patch may also fit an older, incomplete tree. Check that
# its result contains the entire current patch before changing live files.
can_upgrade() {
    local upgrade="$1" check_dir added removed path rc=1
    git apply --check "$upgrade" 2>/dev/null || return 1
    check_dir="$(mktemp -d)"
    while IFS=$'\t' read -r added removed path; do
        if [[ -f "$path" ]]; then
            mkdir -p "$check_dir/$(dirname "$path")"
            cp -p -- "$path" "$check_dir/$path"
        fi
    done < <(git apply --numstat "$PATCH")
    if (cd "$check_dir" && git apply "$upgrade" && git apply --reverse --check "$PATCH") 2>/dev/null; then
        rc=0
    fi
    rm -rf -- "$check_dir"
    return "$rc"
}

if git apply --ignore-space-change --reverse --check "$GDN_OUTPUT" 2>/dev/null || git apply --reverse --check "$GRAPH" 2>/dev/null; then
    echo "KVMem patches already applied"
elif git apply --reverse --check "$PATCH" 2>/dev/null; then
    echo "KVMem patches already applied"
elif git apply --check "$PATCH" 2>/dev/null; then
    git apply "$PATCH"
    echo "applied current KVMem patch to pinned llama.cpp"
elif can_upgrade "$BUDGET_UPGRADE"; then
    git apply "$BUDGET_UPGRADE"
    echo "upgraded existing KVMem tree with reasoning budget fix"
elif can_upgrade "$REPLAY_UPGRADE"; then
    git apply "$REPLAY_UPGRADE"
    echo "upgraded existing KVMem tree with ReplaySSM support"
elif can_upgrade "$UPGRADE"; then
    git apply "$UPGRADE"
    echo "upgraded existing KVMem tree with multimodal and ReplaySSM support"
else
    echo "llama.cpp differs from the supported pin or KVMem baseline; no files changed" >&2
    echo "inspect local changes before replaying $PATCH" >&2
    exit 1
fi

if git apply --ignore-space-change --reverse --check "$GDN_OUTPUT" 2>/dev/null || git apply --reverse --check "$GRAPH" 2>/dev/null; then
    echo "CUDA graph decode patch already applied"
else
    git apply --check "$GRAPH"
    git apply "$GRAPH"
    echo "applied CUDA graph decode patch"
fi

if git apply --ignore-space-change --reverse --check "$GDN_OUTPUT" 2>/dev/null; then
    echo "GDN output fusion patch already applied"
else
    git apply --ignore-space-change --check "$GDN_OUTPUT"
    git apply --ignore-space-change "$GDN_OUTPUT"
    echo "applied GDN output fusion patch"
fi

if git apply --reverse --check "$GRAPH_REACTIVATION" 2>/dev/null; then
    echo "CUDA graph reactivation fix already applied"
else
    git apply --check "$GRAPH_REACTIVATION"
    git apply "$GRAPH_REACTIVATION"
    echo "applied CUDA graph reactivation fix"
fi

if git apply --ignore-space-change --reverse --check "$RDNA2_FATTN" 2>/dev/null; then
    echo "RDNA2 quantized-KV Flash Attention patch already applied"
else
    git apply --ignore-space-change --check "$RDNA2_FATTN"
    git apply --ignore-space-change "$RDNA2_FATTN"
    echo "applied RDNA2 quantized-KV Flash Attention patch"
fi

if git apply --ignore-space-change --reverse --check "$QWEN3MOE_CAPTURE" 2>/dev/null; then
    echo "Qwen3 MoE capture patch already applied"
else
    git apply --ignore-space-change --check "$QWEN3MOE_CAPTURE"
    git apply --ignore-space-change "$QWEN3MOE_CAPTURE"
    echo "applied Qwen3 MoE Q/K/V capture patch"
fi

if git apply --reverse --check "$HISTORICAL_ACTIVE" 2>/dev/null; then
    echo "historical active hook already applied"
else
    git apply --check "$HISTORICAL_ACTIVE"
    git apply "$HISTORICAL_ACTIVE"
    echo "applied historical active admission hook"
fi

if git apply --reverse --check "$HISTORICAL_CELLS" 2>/dev/null; then
    echo "historical cell index reservation already applied"
else
    git apply --check "$HISTORICAL_CELLS"
    git apply "$HISTORICAL_CELLS"
    echo "applied historical cell index reservation"
fi

if git apply --reverse --check "$HISTORICAL_SLOTS" 2>/dev/null; then
    echo "historical slot info return already applied"
else
    git apply --check "$HISTORICAL_SLOTS"
    git apply "$HISTORICAL_SLOTS"
    echo "applied historical slot info return"
fi

if git apply --reverse --check "$HISTORICAL_MICROBATCH" 2>/dev/null; then
    echo "historical microbatch pool already applied"
else
    git apply --check "$HISTORICAL_MICROBATCH"
    git apply "$HISTORICAL_MICROBATCH"
    echo "applied historical microbatch pool"
fi
