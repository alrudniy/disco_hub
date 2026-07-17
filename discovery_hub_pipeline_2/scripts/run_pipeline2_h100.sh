#!/usr/bin/env bash
# run_pipeline2_h100.sh -- THE unattended orchestrator. Runs the full P0->P1 critical
# path on a rented Vast H100, writes a DONE sentinel when finished so the Drew-side
# watchdog can pull results and destroy the box. Idempotent-ish: each stage writes a
# versioned artifact and a .done marker; re-running skips completed stages.
#
# Usage (on the H100, after push_pipeline2_to_h100.sh has synced code+data):
#   nohup bash run_pipeline2_h100.sh > $DH2_ROOT/logs/run.log 2>&1 &
#
# Required env (sourced from dh2_env.sh which push_* writes):
#   DH_DATA_ROOT, DH2_ROOT, DH2_MODEL_4B, DH2_MODEL_8B, DH2_MODEL_06B,
#   DH2_TEACHER_LLM_API_KEY, DH2_TEACHER_LLM_MODEL, PYTHON
set -uo pipefail

PY="${PYTHON:-/venv/main/bin/python}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAGES="$ROOT/stages"
: "${DH2_ROOT:?set DH2_ROOT}"
LOGS="$DH2_ROOT/logs"; mkdir -p "$LOGS"
MARK="$DH2_ROOT/.markers"; mkdir -p "$MARK"

# hygiene: Vast boxes leak HF offline vars that break the reranker cache
unset HF_HUB_OFFLINE HF_HOME 2>/dev/null || true
# reduce CUDA fragmentation OOM during training (recommended by torch's OOM message)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run () {  # run <marker> <logfile> <cmd...>
  local marker="$MARK/$1.done"; shift
  local log="$LOGS/$1.log"; shift
  if [[ -f "$marker" ]]; then echo "[skip] $log already done"; return 0; fi
  echo "[run ] $* (log: $log)"
  if "$@" > "$log" 2>&1; then touch "$marker"; echo "[ok  ] $log";
  else echo "[FAIL] $log -- see log; aborting"; echo "FAILED at $log" > "$DH2_ROOT/FAILED"; exit 1; fi
}

echo "===== pipeline_2 unattended run @ $(date -u) ====="
echo "python: $PY"; "$PY" -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())" || true

# 0) validate the deployed 4B model loads clean before anything expensive
run validate validate "$PY" -m dh2.validate validate --model "$DH2_MODEL_4B" --expected-dim 2560

# P0: candidate pool -> teacher scores -> LLM adjudicate -> qrels
run p0_1 p0_1 "$PY" "$STAGES/p0_1_build_candidate_pool.py" \
    --vectors-8b "${DH2_VECTORS_8B:-}" --ids-8b "${DH2_IDS_8B:-}"
run p0_2 p0_2 "$PY" "$STAGES/p0_2_score_teachers.py"
run p0_3 p0_3 "$PY" "$STAGES/p0_3_llm_adjudicate.py"
run p0_4 p0_4 "$PY" "$STAGES/p0_4_build_eval_qrels.py" --held-out-only

# P1: train labels -> screen bakeoff on 0.6B
run p1_1 p1_1 "$PY" "$STAGES/p1_1_build_train_labels.py"
run p1_2_screen p1_2_screen "$PY" "$STAGES/p1_2_run_bakeoff.py" --stage screen

# screen-eval every arm, choose the winner by exact R@10 (+ nDCG tiebreak)
echo "[run ] screen-eval each arm"
SCREEN_JSON="$DH2_ROOT/bakeoff/screen_models.json"
if [[ -f "$SCREEN_JSON" ]]; then
  # Pass ROOT/DH2_ROOT/PY as ARGV -- they are shell vars here and are NOT in the
  # Python subprocess environment (that caused a KeyError:'ROOT' and skipped the winner).
  "$PY" - "$SCREEN_JSON" "$ROOT" "$DH2_ROOT" "$PY" <<'PYIN' > "$LOGS/screen_eval.log" 2>&1
import json,sys,subprocess
from pathlib import Path
screen_json, root, dh2_root, py = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
sys.path.insert(0, root)
models=json.loads(Path(screen_json).read_text())
best=None
for name,mdir in models.items():
    subprocess.run([py, f"{root}/stages/p1_3_eval_and_compare.py","--model",mdir,
                    "--name",f"screen_{name}"], check=False)
    rep=Path(dh2_root)/"reports"/f"eval_screen_{name}.json"
    if rep.exists():
        r=json.loads(rep.read_text())
        r10=r["exact_origin"].get("recall@10",{}).get("mean",0)
        nd=r["graded_utility"].get("ndcg@10",{}).get("mean",0)
        if best is None or (r10,nd)>(best[1],best[2]): best=(name,r10,nd)
Path(dh2_root,"bakeoff","winner.json").write_text(json.dumps({"winner":best[0] if best else None,"r10":best[1] if best else 0}))
print("WINNER:", best)
PYIN
fi

WINNER="$("$PY" -c "import json,os;print(json.load(open(os.path.join('$DH2_ROOT','bakeoff','winner.json')))['winner'])" 2>/dev/null || echo '')"
echo "[info] bakeoff winner: ${WINNER:-<none>}"

# P1 final: train winner on 4B, then embed/index/eval
if [[ -n "$WINNER" && "$WINNER" != "None" ]]; then
  run p1_2_final p1_2_final "$PY" "$STAGES/p1_2_run_bakeoff.py" --stage final --winner "$WINNER"
  FINAL_MODEL="$("$PY" -c "import json,os;print(json.load(open(os.path.join('$DH2_ROOT','bakeoff','final_model.json')))['model'])")"
  run p1_3_final p1_3_final "$PY" "$STAGES/p1_3_eval_and_compare.py" --model "$FINAL_MODEL" --name "final_${WINNER}_4b"
fi

echo "===== pipeline_2 COMPLETE @ $(date -u) ====="
date -u > "$DH2_ROOT/DONE"        # sentinel the Drew watchdog polls for
echo "wrote $DH2_ROOT/DONE"
