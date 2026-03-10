#!/bin/bash
# Run performance benchmarks and generate a Markdown report.
# Usage: ./scripts/run_benchmarks.sh [bench_name]
#   bench_name: optional benchmark name (default: regression_benchmark)

set -e
cd "$(dirname "$0")/.."

BENCH_NAME="${1:-regression_benchmark}"
REPORT_FILE="/tmp/perf_report.md"
BENCH_OUTPUT="/tmp/bench_output.txt"

echo "# PandRS Performance Report" > "${REPORT_FILE}"
echo "Generated: $(date)" >> "${REPORT_FILE}"
echo "" >> "${REPORT_FILE}"
echo "Benchmark: \`${BENCH_NAME}\`" >> "${REPORT_FILE}"
echo "" >> "${REPORT_FILE}"

echo "Running benchmarks (bench: ${BENCH_NAME})..."
cargo bench --bench "${BENCH_NAME}" 2>&1 | tee "${BENCH_OUTPUT}"

echo "## Results" >> "${REPORT_FILE}"
echo "" >> "${REPORT_FILE}"
echo '```' >> "${REPORT_FILE}"
# Extract criterion timing lines and bencher-style output
grep -E "(test |bench:| time:|ns/iter|μs/iter|ms/iter)" "${BENCH_OUTPUT}" >> "${REPORT_FILE}" || true
echo '```' >> "${REPORT_FILE}"

echo "" >> "${REPORT_FILE}"
echo "## Raw Output" >> "${REPORT_FILE}"
echo '```' >> "${REPORT_FILE}"
tail -100 "${BENCH_OUTPUT}" >> "${REPORT_FILE}"
echo '```' >> "${REPORT_FILE}"

echo ""
echo "=== Performance Report ==="
cat "${REPORT_FILE}"
echo ""
echo "Report saved to: ${REPORT_FILE}"
