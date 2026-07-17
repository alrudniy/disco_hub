#!/usr/bin/env bash
set -euo pipefail
export DH_DATA_ROOT=/workspace/dh_data
export PYTHONPATH=/workspace/dh_rgcn_pipeline
cd /workspace/dh_rgcn_pipeline
exec /venv/main/bin/python -u 06_train_rgcn.py --dim 128 --hidden 128 --epochs 20
