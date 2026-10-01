#!/usr/bin/env bash
#
# run_all_tests.sh - builds and runs the whole test suite, grouped by kind.
#
# Usage:
#   ./scripts/run_all_tests.sh [--build-dir=DIR] [--sanitizers] [--quick]
#
# Exit code is non-zero if any group fails, which makes it usable as a CI gate.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

BUILD_DIR="${ROOT_DIR}/build"
SANITIZERS="no"
QUICK="no"

for arg in "$@"; do
    case "${arg}" in
        --build-dir=*) BUILD_DIR="${arg#*=}" ;;
        --sanitizers)  SANITIZERS="yes" ;;
        --quick)       QUICK="yes" ;;
        -h|--help)
            sed -n '2,10p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) echo "unknown option: ${arg}" >&2; exit 2 ;;
    esac
done

if [[ ! -d "${BUILD_DIR}" ]]; then
    echo "build directory ${BUILD_DIR} does not exist; run ./scripts/build.sh first" >&2
    exit 1
fi

FAILED=0
declare -a RESULTS=()

run_group() {
    local name="$1"; shift
    local regex="$1"; shift

    if [[ "${QUICK}" == "yes" && "${name}" == "property" ]]; then
        RESULTS+=("${name}: SKIPPED (--quick)")
        return 0
    fi

    echo
    echo "================================================================"
    echo "  ${name}"
    echo "================================================================"

    if ctest --test-dir "${BUILD_DIR}" --output-on-failure -R "${regex}" "${@:2}"; then
        RESULTS+=("${name}: PASS")
    else
        RESULTS+=("${name}: FAIL")
        FAILED=1
    fi
}

run_group "unit: semiring"  "unit_semiring"   --timeout 300
run_group "unit: matrix"    "unit_matrix"     --timeout 300
run_group "unit: spgemm"    "unit_spgemm"     --timeout 900
run_group "unit: energy"    "unit_energy"     --timeout 300
run_group "unit: autotuner" "unit_autotuner"  --timeout 600
run_group "smoke: examples" "smoke_examples"  --timeout 300
run_group "property"        "property"        --timeout 1800
run_group "regression"      "regression"      --timeout 1800

# Integration tests need MPI.
if ctest --test-dir "${BUILD_DIR}" -N 2>/dev/null | grep -q integration_distributed; then
    run_group "integration (MPI)" "integration_distributed" --timeout 1800
else
    RESULTS+=("integration (MPI): SKIPPED (not built)")
fi

echo
echo "================================================================"
echo "  summary"
echo "================================================================"
for r in "${RESULTS[@]}"; do
    printf '  %s\n' "${r}"
done
echo

if [[ "${FAILED}" -ne 0 ]]; then
    echo "RESULT: FAIL"
    exit 1
fi
echo "RESULT: PASS"
