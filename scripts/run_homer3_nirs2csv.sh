#!/usr/bin/env bash
# Convert all .nirs files in data/additional-raw/ to HbO/HbR/HbT CSVs.
# Uses nirs2csv_homer3.m which calls the ACTUAL Homer3 functions
# (hmrR_Intensity2OD → hmrR_BandpassFilt → hmrR_OD2Conc).
#
# Default PPF=[6,6] matches the scale of existing data/raw CSVs (Molar units).
# Override with: PPF="1 1" bash scripts/run_homer3_nirs2csv.sh
#
# Output: {basename}_HbO.csv, {basename}_HbR.csv, {basename}_HbT.csv
#         saved alongside each .nirs file in data/additional-raw/
#
# Usage:
#   bash scripts/run_homer3_nirs2csv.sh               # ppf=[6,6] default
#   PPF="1 1" bash scripts/run_homer3_nirs2csv.sh     # ppf=[1,1] (Molar*mm)

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TOOLCHAIN_DIR="${PROJECT_ROOT}/references/synology_0331/clinical data/rawdata/data_NIRx/Toolchain"
DATA_DIR="${PROJECT_ROOT}/data/additional-raw/anxiety"

# PPF: override via env var PPF="6 6" or PPF="1 1"
PPF_ARG="${PPF:-6 6}"
PPF_MATLAB="[${PPF_ARG}]"

if [[ ! -d "${TOOLCHAIN_DIR}" ]]; then
    echo "ERROR: Toolchain directory not found: ${TOOLCHAIN_DIR}" >&2
    exit 1
fi
if [[ ! -d "${DATA_DIR}" ]]; then
    echo "ERROR: data/additional-raw not found: ${DATA_DIR}" >&2
    exit 1
fi

echo "=== Homer3 nirs2csv_homer3 batch converter ==="
echo "Toolchain : ${TOOLCHAIN_DIR}"
echo "Data dir  : ${DATA_DIR}"
echo "PPF       : ${PPF_MATLAB}"
echo ""

NIRS_COUNT=$(find "${DATA_DIR}" -name "*.nirs" | wc -l)
echo "Found ${NIRS_COUNT} .nirs file(s) to process."
echo ""

matlab -batch "addpath('${TOOLCHAIN_DIR}'); nirs2csv_homer3('${DATA_DIR}', ${PPF_MATLAB});"

echo ""
echo "=== Done. CSVs written alongside .nirs files in ${DATA_DIR} ==="
