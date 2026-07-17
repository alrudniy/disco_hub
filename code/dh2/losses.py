"""
dh2.losses -- objectives for the bakeoff (spec P1). All operate on a batch where each
query has a LIST of candidates with teacher relevance (graded), enabling soft-label and
multi-positive training instead of single-positive contrastive.

Implemented (sentence-transformers / torch):
  * MarginMSELoss      (GPL canonical baseline): match student score margins to teacher
                        score margins for (query, positive, negative) triples.
  * ListwiseKLLoss     (primary graded candidate): KL(softmax(teacher/T) || softmax(student/T))
                        over each query's candidate list, + small contrastive aux term.
  * LSEPairLoss        (primary multi-positive): log-sum-exp over the explicit positive set
                        vs confident negatives (multiple valid positives per query).

Design notes:
  * Student score = temperature-scaled cosine (embeddings L2-normalized at inference).
  * Teacher scores are the merged relevance in [0,1] from dh2.teacher_merge.
  * These are sentence_transformers-compatible loss modules (forward(features, labels)),
    but the bakeoff trainer uses a custom collator that packs candidate lists, so the
    losses here take explicit tensors and are unit-testable without a model.

Torch is imported lazily; the pure math is factored into functions tested on CPU.
"""
from __future__ import annotations

import math
from typing import Sequence


# --------------------------------------------------------------------------- #
# Pure-python reference implementations (unit-tested; mirror the torch versions)
# --------------------------------------------------------------------------- #
def _softmax(xs: Sequence[float], T: float = 1.0) -> list[float]:
    m = max(xs)
    exps = [math.exp((x - m) / T) for x in xs]
    s = sum(exps)
    return [e / s for e in exps]


def marginmse_ref(student_pos: float, student_neg: float,
                  teacher_pos: float, teacher_neg: float) -> float:
    """(student_margin - teacher_margin)^2 for one (q,pos,neg)."""
    sm = student_pos - student_neg
    tm = teacher_pos - teacher_neg
    return (sm - tm) ** 2


def listwise_kl_ref(student_scores: Sequence[float], teacher_scores: Sequence[float],
                    T_student: float = 1.0, T_teacher: float = 1.0) -> float:
    """KL(P_teacher || P_student) over one query's candidate list."""
    pt = _softmax(teacher_scores, T_teacher)
    ps = _softmax(student_scores, T_student)
    kl = 0.0
    for a, b in zip(pt, ps):
        if a > 0:
            kl += a * math.log(a / max(b, 1e-12))
    return kl


def lsepair_ref(student_scores: Sequence[float], is_positive: Sequence[bool],
                scale: float = 20.0) -> float:
    """Multi-positive log-sum-exp loss: pull ALL positives above ALL negatives.

    loss = -log( sum_p exp(s_p) / ( sum_p exp(s_p) + sum_n exp(s_n) ) ), scaled.
    Handles >=1 positive per query (the multi-positive case).
    """
    pos = [scale * s for s, p in zip(student_scores, is_positive) if p]
    neg = [scale * s for s, p in zip(student_scores, is_positive) if not p]
    if not pos:
        return 0.0
    mp = max(pos)
    num = sum(math.exp(p - mp) for p in pos)
    allx = pos + neg
    ma = max(allx)
    den = sum(math.exp(x - ma) for x in allx)
    # align the two logsumexp shifts
    lognum = mp + math.log(num)
    logden = ma + math.log(den)
    return -(lognum - logden)


# --------------------------------------------------------------------------- #
# Torch loss modules (used by the trainer)
# --------------------------------------------------------------------------- #
def build_torch_losses():
    """Return torch nn.Module versions. Imported lazily to keep CPU import clean."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class MarginMSE(nn.Module):
        def forward(self, s_pos, s_neg, t_pos, t_neg):
            return F.mse_loss(s_pos - s_neg, t_pos - t_neg)

    class ListwiseKL(nn.Module):
        def __init__(self, T_student=1.0, T_teacher=1.0, aux_weight=0.1, scale=20.0):
            super().__init__()
            self.Ts, self.Tt, self.aux, self.scale = T_student, T_teacher, aux_weight, scale

        def forward(self, student_scores, teacher_scores, mask=None):
            # student_scores,teacher_scores: (B, L); mask: (B,L) 1=valid
            s = student_scores * self.scale
            if mask is not None:
                s = s.masked_fill(mask == 0, -1e4)
                teacher_scores = teacher_scores.masked_fill(mask == 0, -1e4)
            log_ps = F.log_softmax(s / self.Ts, dim=-1)
            pt = F.softmax(teacher_scores / self.Tt, dim=-1)
            kl = F.kl_div(log_ps, pt, reduction="batchmean")
            if self.aux > 0:
                # small contrastive term: treat argmax-teacher as the positive
                tgt = pt.argmax(dim=-1)
                aux = F.cross_entropy(s, tgt)
                return kl + self.aux * aux
            return kl

    class LSEPair(nn.Module):
        def __init__(self, scale=20.0):
            super().__init__()
            self.scale = scale

        def forward(self, student_scores, is_positive, mask=None):
            s = student_scores * self.scale
            if mask is not None:
                s = s.masked_fill(mask == 0, -1e4)
            pos = s.masked_fill(is_positive == 0, -1e4)
            logsum_pos = torch.logsumexp(pos, dim=-1)
            logsum_all = torch.logsumexp(s, dim=-1)
            loss = -(logsum_pos - logsum_all)
            return loss.mean()

    return {"MarginMSE": MarginMSE, "ListwiseKL": ListwiseKL, "LSEPair": LSEPair}
