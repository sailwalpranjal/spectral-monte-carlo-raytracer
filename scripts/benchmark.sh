#!/usr/bin/env bash
# Reproducible throughput check. Builds the renderer, runs a few fixed
# configurations several times, and prints one line per run with the exit code,
# the wall time the renderer reports, and the "MSamples/sec" line it prints.
# "Samples" here is width * height * samples-per-pixel as the program counts it,
# not a count of ray bounces.
#
#   bash scripts/benchmark.sh            # 3 runs per configuration
#   RUNS=5 bash scripts/benchmark.sh
set -u
cd "$(dirname "$0")/.."
RUNS="${RUNS:-3}"
OUT="$(mktemp -d)"
BIN="$OUT/raytracer"

g++ -std=c++14 -O3 -fopenmp -pthread spectral_raytracer.cpp -o "$BIN" || exit 1
echo "compiler: $(g++ --version | head -1)"
echo "threads:  $(nproc 2>/dev/null || echo unknown)"
echo "config                      run  exit  render_ms  MSamples/sec"

for cfg in "64 64 4" "128 128 16" "256 256 16"; do
  set -- $cfg
  for i in $(seq 1 "$RUNS"); do
    "$BIN" --scene cornell --width "$1" --height "$2" --samples "$3" --output "$OUT/img.ppm" > "$OUT/log.txt" 2>&1
    rc=$?
    ms=$(grep -o "([0-9]* ms)" "$OUT/log.txt" | tr -d '() ms')
    perf=$(grep -o "[0-9.]* MSamples/sec" "$OUT/log.txt" | cut -d' ' -f1)
    printf "%-27s %-4s %-5s %-10s %s\n" "${1}x${2} @ ${3} spp" "$i" "$rc" "${ms:--}" "${perf:--}"
  done
done
rm -rf "$OUT"
