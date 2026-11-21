#!/usr/bin/env bash
# Weekly run of Caleb's manual-odor pipeline (config/config_caleb.yaml).
#
# Cron runs this at 00:00 Sunday ("Saturday night"); the config_new nightly is
# OFF that night (its cron line is Mon-Sat). Steps:
#   1. import every finished, fully-pushed session from Caleb's secured area
#      (/securedstorage/DATAsec/Caleb/Caleb/Experiment 2 Manual Odor Testing)
#   2. run the pipeline on config_caleb.yaml -- it mirrors each batch to
#      /securedstorage/DATAsec/Caleb/Data and prunes local videos YOLO finished.
# Step 2 holds the config_new pipeline lock too, so it starts only after any
# config_new run is done (waits up to 12 h) and the two never overlap.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RAW="/securedstorage/DATAsec/Caleb/Caleb/Experiment 2 Manual Odor Testing"
DEST="/home/ramanlab/Documents/cole/Data/flys_Caleb"

for candidate in "${CONDA_BASE:-}" "${HOME}/anaconda3" "${HOME}/miniconda3"; do
    if [ -n "${candidate}" ] && [ -f "${candidate}/etc/profile.d/conda.sh" ]; then
        # shellcheck disable=SC1091
        source "${candidate}/etc/profile.d/conda.sh"; break
    fi
done
conda activate "${YOLO_ENV_NAME:-yolo-env}"

cd "${PROJECT_ROOT}"
echo "[caleb-weekly] $(date '+%F %T') start"
if [ ! -d "${RAW}" ]; then
    echo "[caleb-weekly] raw session folder missing: ${RAW}" >&2
    exit 1
fi
python3 scripts/tools/import_manual_odor_sessions.py --src "${RAW}" --dest "${DEST}"
# config_caleb keeps its own cache (and so its own run lock); also hold the
# config_new lock so the two pipelines never share the GPU: wait (up to 12 h)
# for a config_new run to finish, and make a hand-started one wait for us.
PROD_LOCK="/home/ramanlab/Documents/cole/cache/run_workflows.lock"
echo "[caleb-weekly] waiting for the config_new pipeline lock (${PROD_LOCK})"
flock -w 43200 "${PROD_LOCK}" \
    python3 scripts/pipeline/run_workflows.py --config config/config_caleb.yaml
echo "[caleb-weekly] $(date '+%F %T') done"
