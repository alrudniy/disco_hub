"""
dh2.train_bakeoff -- P1 objective bakeoff.

Runs the arms from the spec table on the multi-positive labeled data:
  A  MNRL (single designated positive)      -- locked control
  B  MNRL with flagged candidates removed   -- hard-filter control
  C  GPL MarginMSE (teacher score margins)  -- canonical soft-label baseline
  D  Listwise KL (+ small contrastive)      -- primary graded candidate
  E  LSEPair (explicit multi-positive)      -- primary multi-positive candidate
  F  Rand1LH (one random valid positive)    -- simple multi-positive control

Screening: run all arms on the 0.6B backbone over a query subset, eliminate weak arms,
then train the surviving arm(s) on 4B (LoRA). The trainer produces a merged model per arm
(using dh2.validate.merge_lora_checkpoint so no arm can emit a broken checkpoint).

The data packing (query -> padded candidate list with grades / positives / mask) is pure
numpy and unit-tested. The torch training loop is lazily imported (runs on the H100).
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from dh2 import config2 as C


# --------------------------------------------------------------------------- #
# Data packing (CPU-testable)
# --------------------------------------------------------------------------- #
@dataclass
class PackedQuery:
    query_id: str
    query: str
    doc_ids: list[str]
    teacher_rel: list[float]     # merged relevance [0,1] per candidate
    is_positive: list[bool]      # grade>=2 (multi-positive set)
    weights: list[float]         # training weights
    masked: list[bool]


def load_labels_grouped(labels_path: str | Path,
                        drop_masked: bool = True) -> dict[str, PackedQuery]:
    """Group merged-label JSONL by query_id into PackedQuery records."""
    g: dict[str, dict] = {}
    for line in Path(labels_path).open():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if drop_masked and r.get("masked"):
            continue
        qid = r["query_id"]
        b = g.setdefault(qid, {"query": r.get("query", ""), "d": [], "rel": [],
                               "pos": [], "w": [], "m": []})
        b["d"].append(r["document_id"])
        b["rel"].append(float(r.get("relevance", 0.0)))
        b["pos"].append(int(r.get("grade", 0)) >= 2 or bool(r.get("is_designated_positive")))
        b["w"].append(float(r.get("training_weight", 1.0)))
        b["m"].append(bool(r.get("masked")))
    out = {}
    for qid, b in g.items():
        if not any(b["pos"]):
            continue  # need at least one positive to train
        out[qid] = PackedQuery(qid, b["query"], b["d"], b["rel"], b["pos"], b["w"], b["m"])
    return out


def make_triples_for_marginmse(pq: PackedQuery, n_neg: int = 4,
                               hard_filter: bool = False,
                               fn_flag: dict[str, bool] | None = None,
                               seed: int = 0) -> list[tuple[str, str, float, float]]:
    """(pos_id, neg_id, teacher_pos, teacher_neg) triples for one query."""
    rng = np.random.default_rng(seed)
    pos_idx = [i for i, p in enumerate(pq.is_positive) if p]
    neg_idx = [i for i, p in enumerate(pq.is_positive) if not p]
    if hard_filter and fn_flag:
        neg_idx = [i for i in neg_idx if not fn_flag.get(pq.doc_ids[i], False)]
    if not pos_idx or not neg_idx:
        return []
    triples = []
    for pi in pos_idx:
        chosen = rng.choice(neg_idx, size=min(n_neg, len(neg_idx)), replace=False)
        for ni in chosen:
            triples.append((pq.doc_ids[pi], pq.doc_ids[int(ni)],
                            pq.teacher_rel[pi], pq.teacher_rel[int(ni)]))
    return triples


def pack_listwise(pq: PackedQuery, max_list: int = 32) -> dict:
    """Pad a query's candidate list to max_list for listwise/LSEPair training."""
    d = pq.doc_ids[:max_list]
    rel = pq.teacher_rel[:max_list]
    pos = pq.is_positive[:max_list]
    L = len(d)
    pad = max_list - L
    return {
        "doc_ids": d + [""] * pad,
        "teacher_rel": rel + [0.0] * pad,
        "is_positive": [int(p) for p in pos] + [0] * pad,
        "mask": [1] * L + [0] * pad,
    }


ARMS = {
    "A_mnrl_control": {"kind": "mnrl", "hard_filter": False},
    "B_mnrl_hardfilter": {"kind": "mnrl", "hard_filter": True},
    "C_gpl_marginmse": {"kind": "marginmse"},
    "D_listwise_kl": {"kind": "listwise"},
    "E_lsepair": {"kind": "lsepair"},
    "F_rand1lh": {"kind": "rand1lh"},
}


def iter_arms(only: list[str] | None = None) -> Iterator[tuple[str, dict]]:
    for name, spec in ARMS.items():
        if only and name not in only:
            continue
        yield name, spec


# --------------------------------------------------------------------------- #
# Torch training (lazy; runs on H100). Kept in one function so CPU import is clean.
# --------------------------------------------------------------------------- #
def train_arm(arm_name: str, arm_spec: dict, labels_path: str, base_model: str,
              out_dir: str, docs_text: dict[str, str], *, is_4b: bool,
              cfg: C.BakeoffConfig = C.BAKEOFF, device: str | None = None,
              max_queries: int = 0, fn_flag: dict[str, bool] | None = None) -> str:
    """Train one arm; return the merged model dir. Torch imported here only."""
    import torch
    from sentence_transformers import SentenceTransformer
    from dh2.losses import build_torch_losses
    from dh2.validate import merge_lora_checkpoint, validate_merged_model
    from discovery_hub.embedding import format_query
    from discovery_hub import config as base

    grouped = load_labels_grouped(labels_path)
    qids = list(grouped)
    if max_queries:
        qids = qids[:max_queries]

    st = SentenceTransformer(base_model, device=device,
                             model_kwargs={"torch_dtype": torch.bfloat16})
    st.max_seq_length = 512
    inner = st[0].auto_model

    # LoRA for 4B; full for 0.6B screening
    ckpt_dir = Path(out_dir) / "checkpoint"
    if is_4b:
        from peft import LoraConfig, get_peft_model
        lora = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
                          lora_dropout=cfg.lora_dropout, bias="none",
                          task_type="FEATURE_EXTRACTION",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                          "gate_proj", "up_proj", "down_proj"])
        st[0].auto_model = get_peft_model(inner, lora)

    losses = build_torch_losses()
    opt = torch.optim.AdamW(st.parameters(), lr=2e-5)
    kind = arm_spec["kind"]
    # size-aware per-step cap: the 4B has ~7x the activation footprint of the 0.6B and
    # OOMs on an 80GB H100 at the 0.6B value, so pick the cap from the model being trained.
    step_cap = cfg.max_pairs_per_step_4b if is_4b else cfg.max_pairs_per_step
    print(f"[train_arm] {arm_name}: is_4b={is_4b} step_cap={step_cap}", flush=True)

    import torch.nn.functional as _F

    def encode(texts):
        """TRAINING-time encode: runs the model forward WITH gradients (st.encode() uses
        torch.no_grad() and returns detached tensors -> loss.backward() would fail with
        'does not require grad'). Tokenize -> model forward -> pool -> L2-normalize,
        matching SentenceTransformer's pooling so scores are consistent."""
        feats = st.tokenize(list(texts))
        # tokenize() may include non-tensor entries; only move tensors to the device
        feats = {k: (v.to(st.device) if hasattr(v, "to") else v) for k, v in feats.items()}
        out = st(feats)                         # forward through all ST modules (grad ON)
        if "sentence_embedding" in out:
            emb = out["sentence_embedding"]     # pooled by the ST pooling module
        else:
            # fallback: mean-pool token embeddings with the attention mask
            tok = out["token_embeddings"]       # (N, T, dim)
            m = feats["attention_mask"].unsqueeze(-1).to(tok.dtype)  # (N, T, 1)
            emb = (tok * m).sum(1) / m.sum(1).clamp(min=1e-9)
        return _F.normalize(emb, p=2, dim=1)

    st.train()
    for epoch in range(cfg.epochs):
        rng = np.random.default_rng(cfg.__hash__() % (2**31) + epoch)
        rng.shuffle(qids)
        for qi, qid in enumerate(qids):
            pq = grouped[qid]
            # INVARIANT (README): query prefix byte-identical train vs inference.
            # DenseRetriever.encode_query() applies format_query() at eval time.
            q = format_query(pq.query, base.QUERY_INSTRUCTION)
            opt.zero_grad()
            if kind in ("mnrl", "marginmse"):
                trips = make_triples_for_marginmse(
                    pq, n_neg=cfg.n_negatives,
                    hard_filter=arm_spec.get("hard_filter", False),
                    fn_flag=fn_flag, seed=qi)
                if not trips:
                    continue
                # cap per-step triples: a query with many positives x n_neg could produce
                # hundreds of triples, each encoded with a full gradient graph -> OOM.
                if len(trips) > step_cap:
                    import random as _r
                    _r.Random(qi).shuffle(trips)
                    trips = trips[:step_cap]
                qv = encode([q])[0]
                pos_txt = [docs_text.get(p, "") for p, _, _, _ in trips]
                neg_txt = [docs_text.get(n, "") for _, n, _, _ in trips]
                pv = encode(pos_txt)
                nv = encode(neg_txt)
                s_pos = (pv @ qv)
                s_neg = (nv @ qv)
                if kind == "marginmse":
                    t_pos = torch.tensor([tp for _, _, tp, _ in trips], device=s_pos.device)
                    t_neg = torch.tensor([tn for _, _, _, tn in trips], device=s_pos.device)
                    loss = losses["MarginMSE"]()(s_pos, s_neg, t_pos, t_neg)
                else:  # mnrl: InfoNCE over pos vs its negs (single positive contrastive)
                    logits = torch.cat([s_pos.unsqueeze(1), s_neg.unsqueeze(1)], dim=1) * cfg.marginmse_scale
                    tgt = torch.zeros(len(trips), dtype=torch.long, device=logits.device)
                    loss = torch.nn.functional.cross_entropy(logits, tgt)
            else:  # listwise / lsepair / rand1lh
                packed = pack_listwise(pq, max_list=step_cap)
                ids = [d for d in packed["doc_ids"]]
                txt = [docs_text.get(d, "") if d else "" for d in ids]
                qv = encode([q])[0]
                dv = encode(txt)               # (L, dim)
                s = (dv @ qv).unsqueeze(0)      # (1, L)
                mask = torch.tensor([packed["mask"]], device=s.device)
                if kind == "listwise":
                    tr = torch.tensor([packed["teacher_rel"]], device=s.device)
                    loss = losses["ListwiseKL"](aux_weight=cfg.contrastive_aux_weight)(s, tr, mask)
                elif kind == "lsepair":
                    isp = torch.tensor([packed["is_positive"]], device=s.device)
                    loss = losses["LSEPair"]()(s, isp, mask)
                else:  # rand1lh: pick one random positive, contrastive vs negatives
                    isp = np.array(packed["is_positive"])
                    pos_pos = np.where(isp == 1)[0]
                    if len(pos_pos) == 0:
                        continue
                    keep_pos = int(np.random.default_rng(qi).choice(pos_pos))
                    sel = [keep_pos] + [i for i in range(len(ids)) if isp[i] == 0 and packed["mask"][i]]
                    logits = s[0, sel].unsqueeze(0) * cfg.marginmse_scale
                    tgt = torch.zeros(1, dtype=torch.long, device=logits.device)
                    loss = torch.nn.functional.cross_entropy(logits, tgt)
            loss.backward()
            opt.step()

    # save checkpoint then merge (4B) or save directly (0.6B)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    st.save(str(ckpt_dir))
    final = str(Path(out_dir) / "final")
    if is_4b:
        merge_lora_checkpoint(base_model, str(ckpt_dir), final,
                              lora_r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
                              lora_dropout=cfg.lora_dropout)
    else:
        st.save(final)
    validate_merged_model(final)
    return final
