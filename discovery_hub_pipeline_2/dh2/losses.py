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
        """BUG #4 FIX. `marginmse_scale` was defined (20.0), documented as "student cosine
        scale for margin matching", used as a softmax temperature in the *mnrl* branch,
        and NEVER referenced here. C therefore matched raw cosine margins (~±0.1) against
        teacher margins (~±1.0): an error floor of ~0.8 that no amount of training could
        close, and the model was dragged toward an unreachable geometry. Measured
        -0.0977 R@10 SIG.

        The fix requires choosing WHICH side to rescale, and both choices cost something:

          student_scale  (s_pos - s_neg) * scale  vs  teacher margin
              MSE errors stay O(1), so C trains at a rate comparable to the
              cross-entropy arms under the shared lr=2e-5 -- which is what makes the
              bakeoff a fair comparison. BUT an easy pair the model already separates
              (cosine margin 0.3 -> 6.0) now overshoots the teacher's max of 1.0, so the
              gradient PUSHES DOWN a margin that was already correct.

          teacher_scale  (t_pos - t_neg) / scale  vs  raw cosine margin
              Preserves serving geometry (IndexFlatIP == cosine) and never asks for a
              margin the space cannot represent. BUT errors collapse to ~1e-3 and C
              effectively trains at a ~400x smaller LR than every other arm, so a null
              result would be uninterpretable.

        Default is student_scale (this field's original documented intent, and the
        fair-LR choice). NEITHER is validated. C's number means nothing until both are
        run -- see HANDOFF "Open decisions".
        """
        def __init__(self, scale=20.0, mode="student_scale"):
            super().__init__()
            if mode not in ("student_scale", "teacher_scale", "none"):
                raise ValueError(f"unknown marginmse mode: {mode}")
            self.scale, self.mode = scale, mode

        def forward(self, s_pos, s_neg, t_pos, t_neg):
            s_margin = s_pos - s_neg
            t_margin = t_pos - t_neg
            if self.mode == "student_scale":
                s_margin = s_margin * self.scale
            elif self.mode == "teacher_scale":
                t_margin = t_margin / self.scale
            return F.mse_loss(s_margin, t_margin)

    class ListwiseKL(nn.Module):
        """BUG #5 FIX. `teacher_rel` entered softmax raw from [0,1] -- a max probability
        ratio of e≈2.7, i.e. a nearly uniform target -- while the student was scaled x20
        and sharply peaked. Minimizing KL(teacher || student) against a flat target
        teaches the student to FLATTEN its own scores, which is the opposite of ranking.
        Measured +0.0091 ns: no effect, exactly as a flattening objective predicts.

        Fix: put the teacher in the SAME units as the student (x scale) before the
        softmax, so the temperatures mean what they say and the target is as peaked as
        the teacher's relevance actually is.
        """
        def __init__(self, T_student=1.0, T_teacher=1.0, aux_weight=0.1, scale=20.0,
                     scale_teacher=True):
            super().__init__()
            self.Ts, self.Tt, self.aux, self.scale = T_student, T_teacher, aux_weight, scale
            self.scale_teacher = scale_teacher

        def forward(self, student_scores, teacher_scores, mask=None):
            # student_scores,teacher_scores: (B, L); mask: (B,L) 1=valid
            s = student_scores * self.scale
            t = teacher_scores * (self.scale if self.scale_teacher else 1.0)
            if mask is not None:
                # -1e4 AFTER scaling: scaling a -1e4 fill would blow it to -2e5 and, in
                # bf16, straight to -inf -> NaN gradients.
                s = s.masked_fill(mask == 0, -1e4)
                t = t.masked_fill(mask == 0, -1e4)
            log_ps = F.log_softmax(s / self.Ts, dim=-1)
            pt = F.softmax(t / self.Tt, dim=-1)
            kl = F.kl_div(log_ps, pt, reduction="batchmean")
            if self.aux > 0:
                # small contrastive term: treat argmax-teacher as the positive
                tgt = pt.argmax(dim=-1)
                aux = F.cross_entropy(s, tgt)
                return kl + self.aux * aux
            return kl

    class LSEPair(nn.Module):
        """BUG #3 FIX: zero-positive guard, matching F_rand1lh and the lsepair_ref above.

        Was: `pos = s.masked_fill(is_positive == 0, -1e4)`. With NO positives in the
        packed window -- 32.7% of queries, because pack_listwise took the first 8
        candidates in pool order -- every entry became -1e4, so
        logsumexp(pos) ≈ -9998 vs logsum_all ≈ 12 and the loss was ≈ +10,010. Worse,
        masked_fill detaches those positions, so the gradient was +softmax(s): push every
        score down, hardest on the top-ranked document, with nothing pulling anything up.
        E scored 0.0545 -- 18% of baseline.

        Rows with no positives now contribute nothing and are excluded from the mean's
        denominator (not merely zeroed, which would still shrink the loss toward 0 and
        quietly scale down every other row's gradient).
        """
        def __init__(self, scale=20.0):
            super().__init__()
            self.scale = scale

        def forward(self, student_scores, is_positive, mask=None):
            s = student_scores * self.scale
            if mask is not None:
                s = s.masked_fill(mask == 0, -1e4)
            is_pos = is_positive.bool()
            if mask is not None:
                is_pos = is_pos & mask.bool()      # a padded slot is never a positive
            has_pos = is_pos.any(dim=-1)           # (B,)
            if not bool(has_pos.any()):
                # No row in this batch has a positive: return a real zero that still
                # carries a grad_fn, so .backward() is a no-op instead of a crash.
                return (s.sum() * 0.0)
            pos = s.masked_fill(~is_pos, -1e4)
            logsum_pos = torch.logsumexp(pos, dim=-1)
            logsum_all = torch.logsumexp(s, dim=-1)
            loss = -(logsum_pos - logsum_all)      # (B,)
            return loss[has_pos].mean()

    return {"MarginMSE": MarginMSE, "ListwiseKL": ListwiseKL, "LSEPair": LSEPair}
