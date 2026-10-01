#!/usr/bin/env bash
#
# run_scaling_study.sh - sweeps process counts and reports strong/weak scaling.
#
# Usage:
#   ./scripts/run_scaling_study.sh --nodes=1,2,4,8 [--mode=strong|weak]
#                                  [--n=200000] [--density=1e-5] [--out=results]
#
# Produces <out>/scaling.csv with columns:
#   procs,n,nnz_c,time_ms,strong_efficiency,speedup
#
# Strong scaling: n is held fixed, so efficiency = T(1) / T(P).
# Weak scaling:   n grows with P, so efficiency should stay near 1.0.
#
# Note: this requires an MPI launcher and enough cores to run the largest
# node count. Use --oversubscribe on a workstation to fake more ranks, but
# understand that the numbers then reflect core oversubscription, not scaling.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

NODES="1,2,4,8"
MODE="strong"
BASE_N=200000
DENSITY=1e-5
REPS=5
OUT_DIR="${ROOT_DIR}/scaling_results"
BIN_DIR="${ROOT_DIR}/build/benchmarks"

for arg in "$@"; do
    case "${arg}" in
        --nodes=*)   NODES="${arg#*=}" ;;
        --mode=*)    MODE="${arg#*=}" ;;
        --n=*)       BASE_N="${arg#*=}" ;;
        --density=*) DENSITY="${arg#*=}" ;;
        --reps=*)    REPS="${arg#*=}" ;;
        --out=*)     OUT_DIR="${arg#*=}" ;;
        --bin-dir=*) BIN_DIR="${arg#*=}" ;;
        -h|--help)
            sed -n '2,22p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) echo "unknown option: ${arg}" >&2; exit 2 ;;
    esac
done

BIN="${BIN_DIR}/scaling_study"
if [[ ! -x "${BIN}" ]]; then
    echo "error: ${BIN} not found. Build first: ./scripts/build.sh --mpi=on" >&2
    exit 1
fi

MPIEXEC="${MPIEXEC:-mpirun}"
MPIEXEC_EXTRA="${MPIEXEC_EXTRA:---allow-run-as-root --oversubscribe}"

mkdir -p "${OUT_DIR}"
CSV="${OUT_DIR}/scaling.csv"

echo "scaling study: mode=${MODE} nodes=${NODES} n=${BASE_N} density=${DENSITY}"
echo "machine:"
{ command -v lscpu >/dev/null 2>&1 && lscpu | grep -E "Model name|^CPU\(s\)"; } || true
echo

printf 'procs,n,nnz_c,time_ms\n' > "${CSV}"

IFS=',' read -ra NODE_LIST <<< "${NODES}"
for p in "${NODE_LIST[@]}"; do
    p="$(echo "${p}" | tr -d '[:space:]')"
    [[ -z "${p}" ]] && continue

    echo "--- running with ${p} rank(s) ---"
    # Each run writes its own CSV; we append the payload here.
    tmp="$(mktemp)"
    ${MPIEXEC} ${MPIEXEC_EXTRA} -np "${p}" "${BIN}" \
        --n "${BASE_N}" --density "${DENSITY}" --reps "${REPS}" --mode "${MODE}"

    if [[ -f "scaling_np${p}.csv" ]]; then
        tail -n +2 "scaling_np${p}.csv" >> "${CSV}"
        rm -f "scaling_np${p}.csv"
    fi
    rm -f "${tmp}"
done

# ---------------------------------------------------------------------------
# Post-process: compute efficiency and speedup relative to P=1.
# ---------------------------------------------------------------------------
python3 - "${CSV}" "${MODE}" <<'PY'
import csv, sys

path, mode = sys.argv[1], sys.argv[2]
rows = []
with open(path) as f:
    for r in csv.DictReader(f):
        rows.append({k: float(v) for k, v in r.items()})

rows.sort(key=lambda r: r['procs'])
if not rows:
    print("no data collected")
    sys.exit(1)

base = rows[0]['time_ms']
for r in rows:
    r['speedup'] = base / r['time_ms'] if r['time_ms'] else 0.0
    if mode == 'strong':
        r['efficiency'] = r['speedup'] / r['procs']
    else:
        # Weak scaling: perfect means time does not change with P.
        r['efficiency'] = base / r['time_ms'] if r['time_ms'] else 0.0

out = path.replace('.csv', '_annotated.csv')
with open(out, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['procs', 'n', 'nnz_c', 'time_ms',
                                      'speedup', 'efficiency'])
    w.writeheader()
    for r in rows:
        w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v)
                    for k, v in r.items()})

print()
hdr = f"{'procs':>6} {'n':>12} {'nnz(C)':>12} {'time[ms]':>11} {'speedup':>9} {'eff':>7}"
print(hdr)
print('-' * len(hdr))
for r in rows:
    print(f"{int(r['procs']):>6} {int(r['n']):>12} {int(r['nnz_c']):>12} "
          f"{r['time_ms']:>11.3f} {r['speedup']:>9.2f} {r['efficiency']:>7.3f}")
print()
print(f"wrote {path} and {out}")
PY

echo
echo "Attach this machine description to the numbers you report:"
echo "  lscpu; nvidia-smi -L; ${MPIEXEC} --version | head -1"
