"""
dh2.config2 -- configuration for DiscoveryHub pipeline_2 (retrieval-quality R&D).

pipeline_2 is ADDITIVE: it imports the base `discovery_hub.config` (paths, models,
query instruction, RETRIEVAL) so there is ONE source of truth for the corpus/serving
config, and layers on the P0/P1 knobs (candidate pooling, teacher scoring, LLM
adjudication, objective bakeoff, promotion gates).

Every path lives under DH2_ROOT (default: <DH_DATA_ROOT>/pipeline2) so pipeline_2
artifacts never overwrite the production embeddings/index/reports.

Env vars (all overridable; nothing hardcoded):
  DH_DATA_ROOT            base data root (inherited from base config)
  DH2_ROOT               pipeline_2 artifact root (default <DH_DATA_ROOT>/pipeline2)

  # models already deployed / available (base config owns EMBED_MODEL etc.)
  DH2_MODEL_4B           path to the deployed 4B model dir (student + one teacher-pool source)
  DH2_MODEL_8B           path to the 8B model dir (second candidate-pool source)
  DH2_MODEL_06B          path to the 0.6B model dir (bakeoff screening backbone)

  # teachers
  DH2_TEACHER_BGE        BGE cross-encoder (default from base RERANK_MODEL)
  DH2_TEACHER_QWEN       Qwen3-Reranker-8B repo id or local path (H100 teacher)

  # LLM gray-zone judge (1min.ai) -- DECOUPLED from the online RAG LLM (DH_LLM_*)
  DH2_TEACHER_LLM_API_KEY   1min.ai API key
  DH2_TEACHER_LLM_MODEL     model string (default 'claude-sonnet-4-6')
  DH2_TEACHER_LLM_BASE_URL  default https://api.1min.ai
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# ONE source of truth: reuse the base pipeline's config wholesale.
from discovery_hub import config as base  # noqa: F401  (re-exported for callers)

# --------------------------------------------------------------------------- #
# Roots
# --------------------------------------------------------------------------- #
DATA_ROOT: Path = base.DATA_ROOT
DH2_ROOT: Path = Path(os.environ.get("DH2_ROOT", str(DATA_ROOT / "pipeline2"))).resolve()

# pipeline_2 subdirs (kept entirely separate from production dirs)
POOL_DIR = DH2_ROOT / "candidate_pool"     # p0_1 -> candidate_pool_v*.jsonl
TEACHER_DIR = DH2_ROOT / "teacher_scores"  # p0_2 -> teacher_scores_{bge,qwen}_v*.jsonl
ADJUD_DIR = DH2_ROOT / "adjudication"      # p0_3 -> llm_grayzone_judgments_v*.jsonl
QRELS_DIR = DH2_ROOT / "qrels"             # p0_4 -> qrels_{exact_origin,scout_utility}_v*.jsonl
LABELS_DIR = DH2_ROOT / "train_labels"     # p1_1 -> multi_positive_labels_v*.jsonl
BAKEOFF_DIR = DH2_ROOT / "bakeoff"         # p1_2 -> models + per-arm dirs
REPORTS2_DIR = DH2_ROOT / "reports"        # p1_3 -> eval_*.md, comparison
MANIFEST_DIR = DH2_ROOT / "manifests"      # §9.2 artifact manifests
LOGS_DIR = DH2_ROOT / "logs"

DH2_DIRS = [POOL_DIR, TEACHER_DIR, ADJUD_DIR, QRELS_DIR, LABELS_DIR,
            BAKEOFF_DIR, REPORTS2_DIR, MANIFEST_DIR, LOGS_DIR]


def ensure_dirs() -> None:
    """Create every pipeline_2 output directory. Safe to call repeatedly."""
    DH2_ROOT.mkdir(parents=True, exist_ok=True)
    for d in DH2_DIRS:
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Model locations (defaults match this session's on-Drew layout; override freely)
# --------------------------------------------------------------------------- #
_MODELS = DATA_ROOT.parent / "models"      # e.g. /home/alex/discovery_hub/models
MODEL_4B = os.environ.get("DH2_MODEL_4B", str(_MODELS / "qwen3-dh-ft-4b"))
MODEL_8B = os.environ.get("DH2_MODEL_8B", str(_MODELS / "qwen3-dh-ft-8b"))
MODEL_06B = os.environ.get("DH2_MODEL_06B", str(_MODELS / "qwen3-dh-ft"))

# --------------------------------------------------------------------------- #
# Teachers
# --------------------------------------------------------------------------- #
TEACHER_BGE = os.environ.get("DH2_TEACHER_BGE", base.RERANK_MODEL)  # BAAI/bge-reranker-v2-m3
TEACHER_QWEN = os.environ.get("DH2_TEACHER_QWEN", "Qwen/Qwen3-Reranker-8B")

# LLM gray-zone judge (z.ai, OpenAI-compatible), decoupled from online DH_LLM_*
# z.ai endpoint shape: {base}/chat/completions, Bearer auth, choices[0].message.content
TEACHER_LLM_API_KEY = os.environ.get("DH2_TEACHER_LLM_API_KEY", "")
TEACHER_LLM_MODEL = os.environ.get("DH2_TEACHER_LLM_MODEL", "glm-4.6")
TEACHER_LLM_BASE_URL = os.environ.get("DH2_TEACHER_LLM_BASE_URL",
                                      "https://api.z.ai/api/paas/v4")
# low temperature => deterministic, consistent grading (not creative variety)
TEACHER_LLM_TEMPERATURE = float(os.environ.get("DH2_TEACHER_LLM_TEMPERATURE", "0.1"))
# fast mode: grade-only output (~2-3s/call) vs full evidence string (~15s/call).
# Default ON for high-volume gray-zone judging. Set DH2_TEACHER_LLM_FAST=0 for evidence.
TEACHER_LLM_FAST = os.environ.get("DH2_TEACHER_LLM_FAST", "1") not in ("0", "false", "False")


# --------------------------------------------------------------------------- #
# P0/P1 knobs
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PoolConfig:
    # candidate pool per query (spec P1: 64-128 unique candidates)
    per_source_topk: int = 40          # top-k pulled from each retrieval channel
    deep_window_start: int = 30        # deeper dense window [start,end) for medium-hard negs
    deep_window_end: int = 100
    max_candidates: int = 128          # cap unique candidates per query
    eval_pool_per_query: int = 50      # P0 eval pool depth (30-50 in spec)


@dataclass(frozen=True)
class TeacherConfig:
    bge_batch_size: int = 64
    qwen_batch_size: int = 16          # 8B reranker is heavier
    rerank_max_length: int = 512
    # agreement / gray-zone routing on cross-encoder scores (both ~[0,1] after sigmoid)
    # Widened from the initial 0.15/0.30/0.75: at those values ~69% of pairs routed to the
    # LLM (teachers treated as "disagreeing" on most pairs), which is both miscalibrated and
    # far too much LLM volume. These wider bands auto-resolve the many pairs where the two
    # cross-encoders broadly agree, sending only genuinely ambiguous pairs to the judge.
    agree_high_delta: float = 0.25     # |bge - qwen| <= this AND same side of mid => high agreement
    grayzone_lo: float = 0.40          # scores in [lo,hi] on either teacher => ambiguous -> LLM
    grayzone_hi: float = 0.65
    # positive-aware false-negative flag (reported, NOT auto-applied) -- from the audit
    posaware_margin: float = 0.05      # flag neg if teacher(neg) >= teacher(pos)*(1-margin)


@dataclass(frozen=True)
class GradeConfig:
    # graded relevance scale (spec P0)
    #   3 directly_actionable | 2 clearly_relevant | 1 tangential | 0 irrelevant
    # mapping from merged teacher/LLM signal -> grade (thresholds on [0,1] relevance)
    g3: float = 0.85
    g2: float = 0.60
    g1: float = 0.35


@dataclass(frozen=True)
class BakeoffConfig:
    screen_queries: int = 10000        # 0.6B screening subset (spec: 10k-query matrix)
    epochs: int = 1
    batch_size_06b: int = 16
    batch_size_4b: int = 8             # LoRA on 80GB
    n_negatives: int = 4
    # Per-query cap on triples/candidates encoded in one step (OOM guard). The 4B carries
    # roughly 7x the activation footprint of the 0.6B, and OOMed on an 80GB H100 at 32
    # (LoRA forward: peft/tuners/lora/layer.py -> CUDA OOM), so it gets its own value.
    max_pairs_per_step: int = int(os.environ.get("DH2_STEP_CAP", "8"))       # 0.6B screening arms
    max_pairs_per_step_4b: int = int(os.environ.get("DH2_STEP_CAP_4B", "4"))     # 4B final training (LoRA)
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    listwise_T_teacher: float = 1.0
    listwise_T_student: float = 1.0
    marginmse_scale: float = 20.0      # student cosine scale for margin matching
    contrastive_aux_weight: float = 0.1  # small contrastive term alongside listwise


@dataclass(frozen=True)
class PromotionGates:
    # spec §5.2 default gates (revisable after CI analysis, but documented first)
    min_delta_r10_exact: float = 0.02
    min_delta_ndcg10_utility: float = 0.02
    max_source_r10_regression: float = 0.02
    require_no_mrr_regression: bool = True


POOL = PoolConfig()
TEACHER = TeacherConfig()
GRADE = GradeConfig()
BAKEOFF = BakeoffConfig()
GATES = PromotionGates()

# convenience: inputs pipeline_2 reads from the PRODUCTION pipeline (read-only)
PROD_DOCS = base.NORM_DIR / "docs.jsonl"
PROD_TRAIN_TRIPLES = DATA_ROOT / "finetune" / "train_triples.jsonl"
PROD_SYNTH_QUERIES = DATA_ROOT / "finetune" / "synthetic_queries.jsonl"
PROD_EVAL_QRELS = DATA_ROOT / "finetune" / "synthetic_queries.jsonl.eval.jsonl"
