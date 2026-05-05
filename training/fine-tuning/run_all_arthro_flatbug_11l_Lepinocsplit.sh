#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

SKIP_EXISTING="${SKIP_EXISTING:-1}" MODES="transfer" bash training/fine-tuning/run_matrix_lepinoc_folds.sh
