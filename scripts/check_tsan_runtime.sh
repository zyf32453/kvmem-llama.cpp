#!/usr/bin/env bash
# Linux host-test preflight. Never suppress a sanitizer report or change the
# product's timed waits to compensate for a broken detector.
# Usage: bash scripts/check_tsan_runtime.sh [compiler] [evidence-directory]
# An extracted toolchain may require its library directory in LD_LIBRARY_PATH.
# KVMEM_TSAN_FIXED_LAYOUT=1 opts into setarch -R for these two child processes.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
compiler="${1:-c++}"
out="${2:-$(mktemp -d "${TMPDIR:-/tmp}/kvmem-tsan-runtime.XXXXXX")}"
mkdir -p -- "$out"
out="$(cd -- "$out" && pwd)"
runner=()
if [[ "${KVMEM_TSAN_FIXED_LAYOUT:-0}" == 1 ]]; then
    runner=(setarch "$(uname -m)" -R)
fi
"$compiler" --version > "$out/compiler.txt"
uname -a > "$out/kernel.txt"
for variant in synchronized race; do
    definitions=()
    if [[ "$variant" == race ]]; then definitions=(-DKVMEM_TSAN_PROBE_RACE); fi
    "$compiler" -std=c++17 -O1 -g -pthread -fsanitize=thread -fno-omit-frame-pointer \
        "${definitions[@]}" "$root/tests/tsan-runtime-probe.cpp" -o "$out/$variant" \
        > "$out/$variant-build.log" 2>&1
    ldd "$out/$variant" > "$out/$variant-libraries.txt"
    set +e
    # Override inherited options: a suppression file or report_bugs=0 would
    # invalidate both the compatibility check and the positive control.
    timeout 15 "${runner[@]}" env TSAN_OPTIONS=halt_on_error=1:exitcode=66 \
        "$out/$variant" > "$out/$variant.log" 2>&1
    result=$?
    set -e
    printf '%s\n' "$result" > "$out/$variant.exit"
    if [[ "$variant" == synchronized ]]; then
        if [[ "$result" != 0 ]] || ! grep -q '^condition-variable handoff: passed$' "$out/$variant.log"; then
            cat "$out/$variant.log"
            printf 'TSan preflight FAILED: synchronized timed wait; inspect %s\n' "$out" >&2
            exit 1
        fi
    elif [[ "$result" != 66 ]] || ! grep -q 'WARNING: ThreadSanitizer: data race' "$out/$variant.log"; then
        cat "$out/$variant.log"
        printf 'TSan preflight FAILED: intentional race was not correctly reported; inspect %s\n' "$out" >&2
        exit 1
    fi
done
printf 'TSan preflight passed: timed wait works and intentional data race is detected. Evidence: %s\n' "$out"
