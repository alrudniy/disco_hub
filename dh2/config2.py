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

# --- the semantic hedge (RASC) --------------------------------------------- #
# NOTE: MODEL_06B above is the FINE-TUNED 0.6B ("qwen3-dh-ft"). The cascade needs the
# UNTOUCHED stock checkpoint -- the arm the 2026-07-15 eval found to be the strongest
# low-overlap matcher (LOW R@10 0.2556 vs 0.1667 for the best trained arm). These are
# different models; do not collapse them.
MODEL_BASE_06B = os.environ.get("DH2_MODEL_BASE_06B", "Qwen/Qwen3-Embedding-0.6B")
# Query-side instruction for the UNTOUCHED stock 0.6B channel. The FT channels inherit
# base.QUERY_INSTRUCTION (a licensing-oriented pharma task the FT models were adapted
# around); this constant lets the stock 0.6B carry its own task string instead. NOTE:
# both are natural-language instructions in Qwen3's asymmetric format ("Instruct:
# {task}\nQuery: ...", docs un-prefixed), so this is a task-wording change, not a format
# fix -- the stock 0.6B was already being used in the correct asymmetric mode.
QWEN3_BASE_QUERY_INSTRUCTION = os.environ.get(
    "DH2_QWEN3_BASE_QUERY_INSTRUCTION",
    "Given a pharmaceutical research need, retrieve relevant technology "
    "disclosures, patents, or clinical findings.")
# optional reasoning-oriented discovery branch (gated: must earn its slot, see CASCADE)
MODEL_REASONEMBED = os.environ.get("DH2_MODEL_REASONEMBED", "")

# --- groupwise final pass (feature-flagged, gated) -------------------------- #
GROUPRANK_MODEL = os.environ.get("DH2_GROUPRANK_MODEL", "AQ-MedAI/Diver-GroupRank-32B")

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
    # BGE-reranker-v2-m3 is an XLM-R encoder with a HARD 512-position limit -- this field
    # feeds BgeTeacher and cannot be raised. The RASC recommendation's "set
    # rerank_max_length = 8192" is implemented as the separate Qwen-only field below;
    # raising the shared field would have broken BGE and ballooned the p0_2 scoring pass
    # over 329,761 pairs for no benefit.
    rerank_max_length: int = 512
    # Qwen3-Reranker-8B is a causal LM with a 32K window; 8K is the first deployment
    # target (RASC S3). Pair context = instruct + query + document view.
    qwen_rerank_max_length: int = int(os.environ.get("DH2_QWEN_RERANK_MAX_LEN", "8192"))
    qwen_use_flash_attention: bool = os.environ.get(
        "DH2_QWEN_FLASH_ATTN", "1") not in ("0", "false", "False")
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
    # Per-query cap on triples/candidates encoded in one step (OOM guard). Each triple
    # costs TWO encodes (pos + neg) with live gradient graphs, so the effective batch is
    # ~2x this. The 4B carries ~7x the activation footprint of the 0.6B and OOMed at 32.
    # NOTE: with dense multi-positive labels (~50 positives/query) EVERY query hits the
    # cap, so the cap IS the batch size -- tune it down if you OOM.
    # Env-overridable: DH2_STEP_CAP / DH2_STEP_CAP_4B
    max_pairs_per_step: int = int(os.environ.get("DH2_STEP_CAP", "8"))        # 0.6B arms
    max_pairs_per_step_4b: int = int(os.environ.get("DH2_STEP_CAP_4B", "4"))  # 4B final
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    listwise_T_teacher: float = 1.0
    listwise_T_student: float = 1.0
    marginmse_scale: float = 20.0      # student cosine scale for margin matching
    # BUG #4 (was: scale defined, documented, and NEVER referenced by the MarginMSE
    # module -- C matched raw cosine margins ~±0.1 against teacher margins ~±1.0 and
    # scored -0.0977 SIG). The fix needs a judgment call, so it is a knob, not a guess:
    #   "student_scale" -> (s_pos - s_neg) * scale  vs teacher margin.
    #                      Keeps MSE errors O(1), comparable to the cross-entropy arms at
    #                      the shared lr=2e-5. BUT a well-separated easy pair (cosine
    #                      margin 0.3 -> 6.0) overshoots a teacher max of 1.0, so the loss
    #                      actively PUSHES DOWN margins the model already got right.
    #   "teacher_scale" -> (t_pos - t_neg) / scale  vs raw cosine margin.
    #                      Preserves serving geometry (IndexFlatIP == cosine) and never
    #                      asks for an unreachable margin. BUT errors shrink to ~1e-3 and
    #                      C effectively trains at a 400x smaller LR than the other arms.
    #   "none"          -> the old (broken) behavior, kept only to reproduce the -0.0977.
    # Default is student_scale: it matches this field's original documented intent and
    # keeps the bakeoff's LR comparable across arms. NEITHER mode is validated -- C's
    # number is uninterpretable until both are ablated. See HANDOFF "Open decisions".
    marginmse_mode: str = os.environ.get("DH2_MARGINMSE_MODE", "student_scale")
    # BUG #5: teacher relevance enters softmax raw from [0,1] (max ratio e~2.7, near
    # uniform) while the student is x20 and peaked -> the KL taught D to FLATTEN its
    # scores (+0.0091 ns, no effect). Put the teacher in the same units as the student.
    listwise_scale_teacher: bool = os.environ.get(
        "DH2_LISTWISE_SCALE_TEACHER", "1") not in ("0", "false", "False")
    contrastive_aux_weight: float = 0.1  # small contrastive term alongside listwise


@dataclass(frozen=True)
class CascadeConfig:
    """Register-Aware Semantic Cascade (RASC-32B). Inference-only; no training.

    Retrieve from the untouched semantic model AND the fine-tuned domain model, union
    them (an uncapped union cannot have lower pre-rerank recall than either constituent),
    rerank with a correctly-prompted Qwen3-Reranker-8B, then optionally group-rerank.
    """
    # ---- Stage 1: channel depths (0 disables a channel) ----
    depth_base_06b: int = 100          # semantic hedge; protects low-overlap recall
    depth_ft_8b: int = 100             # strongest domain/capacity branch
    depth_ft_4b: int = 50              # optional diversity branch
    depth_reasonembed: int = 100       # optional; only if MODEL_REASONEMBED is set
    # Union cap. RETENTION IS ROUND-ROBIN ACROSS CHANNELS, never a global sort by one
    # model's score -- sorting the remainder by dense_4b is what evicted 8B-only and
    # semantic-only candidates and structurally favored the lexical-shortcut family.
    max_union: int = 400               # 0 = uncapped (recommended when latency allows)
    exact_identifier_channel: bool = True   # NCT/patent/CAS/compound codes only, no BM25

    # ---- Stage 2: source-aware document views (token budgets) ----
    qwen_view_tokens: int = 3000       # target 2k-4k per doc inside the 8K pair context
    group_view_tokens: int = 400       # 250-500 so several docs fit per group

    # ---- Stage 3: Qwen pointwise rerank ----
    qwen_shortlist: int = 40           # top-N kept after Qwen, fed to GroupRank

    # ---- Stage 4: GroupRank-32B (feature-flagged; OFF until it passes the gates) ----
    enable_grouprank: bool = os.environ.get("DH_ENABLE_GROUPRANK", "0") not in (
        "0", "false", "False")
    group_rank_size: int = 10          # documents per group
    group_rank_repeats: int = 2        # independent random partitions; scores averaged
    group_rank_seed: int = 20240611    # deterministic partitions for reproducible evals
    # Tensor-parallel GPUs for the 32B. 64 GB of BF16 weights: TP=1 fits a single H200
    # (141 GB) or B200 comfortably and is SIMPLER -- no NCCL, no multi-proc, no head-count
    # divisibility constraint. TP=2 for 2xH100-80GB; TP=4 only if that is the box you have.
    # vLLM requires TP to divide BOTH the attention-head and KV-head counts, so 1/2/4 are
    # the safe values -- confirm against the checkpoint's config.json before renting.
    group_rank_tp_size: int = int(os.environ.get("DH2_GR_TP_SIZE", "4"))
    group_rank_max_model_len: int = int(os.environ.get("DH2_GR_MAX_LEN", "16384"))
    group_rank_gpu_util: float = float(os.environ.get("DH2_GR_GPU_UTIL", "0.90"))


@dataclass(frozen=True)
class CascadeGates:
    """RASC promotion gates. Deliberately NOT exact-origin R@10 -- that benchmark scores
    paraphrase retrieval (origin docs carry +0.100 more query-word overlap than equally
    relevant alternatives), which is the bias the cascade exists to correct."""
    # Qwen cascade promotes only if ALL hold vs the incumbent dense path:
    max_low_overlap_utility_regression: float = 0.0   # low-overlap must not regress
    max_grade3_mrr_regression: float = 0.0            # grade-3 MRR must not regress
    min_delta_ndcg10_utility: float = 0.0             # graded nDCG@10 must improve
    max_source_regression_pp: float = 2.0             # no source loses >2 percentage pts
    require_union_recall_at_least_constituents: bool = True   # Recall@100 >= every branch
    # GroupRank promotes only if ALL hold vs Qwen-only ordering:
    grouprank_min_parse_success: float = 0.99
    grouprank_max_p95_latency_s: float = float(os.environ.get("DH2_GR_MAX_P95_S", "20"))


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
CASCADE = CascadeConfig()
CASCADE_GATES = CascadeGates()

# Master feature flag for the production path (stage 07). OFF => existing dense ordering.
REGISTER_AWARE_CASCADE = os.environ.get("DH_REGISTER_AWARE_CASCADE", "0") not in (
    "0", "false", "False")

# convenience: inputs pipeline_2 reads from the PRODUCTION pipeline (read-only)
PROD_DOCS = base.NORM_DIR / "docs.jsonl"
PROD_TRAIN_TRIPLES = DATA_ROOT / "finetune" / "train_triples.jsonl"
PROD_SYNTH_QUERIES = DATA_ROOT / "finetune" / "synthetic_queries.jsonl"
PROD_EVAL_QRELS = DATA_ROOT / "finetune" / "synthetic_queries.jsonl.eval.jsonl"
