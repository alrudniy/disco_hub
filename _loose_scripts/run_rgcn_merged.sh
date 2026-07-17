#!/usr/bin/env bash
set -euo pipefail
export DH_DATA_ROOT=/workspace/dh_merged
export PYTHONPATH=/workspace/dh_rgcn_pipeline
cd /workspace/dh_rgcn_pipeline
/venv/main/bin/python -u 06_train_rgcn.py --dim 128 --hidden 128 --epochs 20
echo "=== TRAINING DONE, RUNNING BASELINE CHECK ON MERGED GRAPH ==="
exec /venv/main/bin/python -u baseline_check.py --out /workspace/rgcn_baseline_merged.json
