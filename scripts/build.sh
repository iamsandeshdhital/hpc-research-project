#!/usr/bin/env bash
#
# build.sh - configure and build the HPC Research Project.
#
# Usage:
#   ./scripts/build.sh [--build-type=Release|Debug|RelWithDebInfo]
#                      [--cuda=on|off] [--mpi=on|off] [--sanitizers=on|off]
#                      [--build-dir=DIR] [--jobs=N] [--install] [--clean]
#
# Examples:
#   ./scripts/build.sh                                     # defaults
#   ./scripts/build.sh --build-type=Debug --sanitizers=on  # debug build
#   ./scripts/build.sh --cuda=on --mpi=on --install        # full HPC build

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

BUILD_TYPE="Release"
ENABLE_CUDA="off"
ENABLE_MPI="off"
ENABLE_SANITIZERS="off"
BUILD_DIR="${ROOT_DIR}/build"
JOBS="$(nproc 2>/dev/null || echo 4)"
DO_INSTALL="no"
DO_CLEAN="no"

for arg in "$@"; do
    case "${arg}" in
        --build-type=*) BUILD_TYPE="${arg#*=}" ;;
        --cuda=*)       ENABLE_CUDA="${arg#*=}" ;;
        --mpi=*)        ENABLE_MPI="${arg#*=}" ;;
        --sanitizers=*) ENABLE_SANITIZERS="${arg#*=}" ;;
        --build-dir=*)  BUILD_DIR="${arg#*=}" ;;
        --jobs=*)       JOBS="${arg#*=}" ;;
        --install)      DO_INSTALL="yes" ;;
        --clean)        DO_CLEAN="yes" ;;
        -h|--help)
            sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "unknown option: ${arg}" >&2
            exit 2
            ;;
    esac
done

if [[ "${DO_CLEAN}" == "yes" ]]; then
    echo "==> removing ${BUILD_DIR}"
    rm -rf "${BUILD_DIR}"
fi

if ! command -v cmake >/dev/null 2>&1; then
    cat >&2 <<'EOF'
error: cmake not found.

  Debian/Ubuntu : sudo apt-get install cmake ninja-build build-essential
  Fedora/RHEL   : sudo dnf install cmake ninja-build gcc-c++
  macOS         : brew install cmake ninja
EOF
    exit 1
fi

GENERATOR="Unix Makefiles"
if command -v ninja >/dev/null 2>&1; then
    GENERATOR="Ninja"
fi

echo "==> configuring"
echo "    build type : ${BUILD_TYPE}"
echo "    CUDA       : ${ENABLE_CUDA}"
echo "    MPI        : ${ENABLE_MPI}"
echo "    sanitizers : ${ENABLE_SANITIZERS}"
echo "    build dir  : ${BUILD_DIR}"
echo "    generator  : ${GENERATOR}"
echo

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -G "${GENERATOR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DENABLE_CUDA=$([[ "${ENABLE_CUDA}" == "on" ]] && echo ON || echo OFF) \
    -DENABLE_MPI=$([[ "${ENABLE_MPI}" == "on" ]] && echo ON || echo OFF) \
    -DENABLE_SANITIZERS=$([[ "${ENABLE_SANITIZERS}" == "on" ]] && echo ON || echo OFF) \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

echo
echo "==> building with ${JOBS} jobs"
cmake --build "${BUILD_DIR}" --parallel "${JOBS}"

if [[ "${DO_INSTALL}" == "yes" ]]; then
    echo
    echo "==> installing"
    cmake --install "${BUILD_DIR}"
fi

echo
echo "==> done"
echo "    binaries : ${BUILD_DIR}/benchmarks, ${BUILD_DIR}/examples"
echo "    tests    : ctest --test-dir ${BUILD_DIR} --output-on-failure"
