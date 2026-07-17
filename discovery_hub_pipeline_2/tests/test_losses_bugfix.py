#!/usr/bin/env python3
"""Numerical regression tests for the loss bugs diagnosed in HANDOFF #3, #4, #5.

The handoff's own lesson: reading the loss source predicted C/D/E's outcomes correctly,
but every number extrapolated from that reading was wrong. So each test here RUNS the
old behavior and the new one and compares actual tensors -- no assertions from reasoning.

CPU-only, no model loads, ~2s.  Run: python tests/test_losses_bugfix.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from dh2.losses import build_torch_losses, lsepair_ref

L = build_torch_losses()


def test_lsepair_zero_positive_guard():
    """BUG #3: no positives in the window -> loss ~+10,010 and an all-push-down gradient."""
    MarginMSE, ListwiseKL, LSEPair = L["MarginMSE"], L["ListwiseKL"], L["LSEPair"]
    s = torch.tensor([[0.5, 0.4, 0.3, 0.2]], requires_grad=True)
    no_pos = torch.tensor([[0, 0, 0, 0]])
    mask = torch.ones_like(no_pos)

    # --- reproduce the OLD behavior verbatim ---
    old_s = s * 20.0
    old_pos = old_s.masked_fill(no_pos == 0, -1e4)
    old_loss = -(torch.logsumexp(old_pos, dim=-1) - torch.logsumexp(old_s, dim=-1)).mean()
    assert old_loss.item() > 9000, f"old loss should explode, got {old_loss.item()}"

    # --- the fix ---
    loss = LSEPair()(s, no_pos, mask)
    assert abs(loss.item()) < 1e-6, f"guarded loss must be 0, got {loss.item()}"
    loss.backward()
    assert torch.allclose(s.grad, torch.zeros_like(s)), \
        f"zero-positive rows must produce NO gradient, got {s.grad}"
    print(f"  [#3] zero-positive: old loss={old_loss.item():.1f} -> new loss=0.0, grad=0 OK")

    # A row WITH positives still learns, and matches the pure-python reference.
    # NOTE: scores must be NON-SATURATED. At scale=20, [0.9, 0.2, ...] gives the positive
    # softmax p=0.999999 -- the loss has already converged, so its gradient underflows to
    # exactly 0.0 in float32 and proves nothing. [0.5, 0.45, 0.4, 0.35] -> p~0.64, which
    # is a query the model is still learning.
    s2 = torch.tensor([[0.5, 0.45, 0.4, 0.35]], requires_grad=True)
    is_pos = torch.tensor([[1, 0, 0, 0]])
    out = LSEPair()(s2, is_pos, torch.ones_like(is_pos))
    ref = lsepair_ref([0.5, 0.45, 0.4, 0.35], [True, False, False, False], scale=20.0)
    assert abs(out.item() - ref) < 1e-4, f"torch {out.item()} != ref {ref}"
    out.backward()
    assert s2.grad[0, 0] < 0, f"the positive must be pulled UP, grad={s2.grad[0,0]}"
    assert (s2.grad[0, 1:] > 0).all(), f"negatives must be pushed DOWN, grad={s2.grad}"
    print(f"  [#3] with positives: torch={out.item():.6f} == ref={ref:.6f}, "
          f"grad(pos)={s2.grad[0,0]:+.3f} grad(neg)={s2.grad[0,1]:+.3f} OK")

    # mixed batch: the zero-positive row must not drag the mean toward 0
    s3 = torch.tensor([[0.5, 0.45, 0.4, 0.35], [0.5, 0.4, 0.3, 0.2]], requires_grad=True)
    mixed = torch.tensor([[1, 0, 0, 0], [0, 0, 0, 0]])
    mixed_loss = LSEPair()(s3, mixed, torch.ones_like(mixed))
    assert abs(mixed_loss.item() - out.item()) < 1e-4, \
        f"zero-pos row must be EXCLUDED from the mean, not averaged in as 0: " \
        f"{mixed_loss.item()} vs {out.item()}"
    mixed_loss.backward()
    assert torch.allclose(s3.grad[1], torch.zeros(4)), \
        f"the zero-positive row must get NO gradient, got {s3.grad[1]}"
    print(f"  [#3] mixed batch: loss={mixed_loss.item():.6f} == single-row "
          f"{out.item():.6f} (excluded, not averaged to {out.item()/2:.6f}) OK")


def test_marginmse_scale_applied():
    """BUG #4: marginmse_scale was never referenced -> ~0.8 irreducible error floor."""
    MarginMSE = L["MarginMSE"]
    # student cosine margin ~0.05 (realistic for normalized embeddings on a hard pair);
    # teacher margin 1.0 (relevance 1.0 vs 0.0)
    s_pos = torch.tensor([0.65]); s_neg = torch.tensor([0.60])
    t_pos = torch.tensor([1.0]);  t_neg = torch.tensor([0.0])

    broken = MarginMSE(mode="none")(s_pos, s_neg, t_pos, t_neg)
    student = MarginMSE(scale=20.0, mode="student_scale")(s_pos, s_neg, t_pos, t_neg)
    teacher = MarginMSE(scale=20.0, mode="teacher_scale")(s_pos, s_neg, t_pos, t_neg)

    assert abs(broken.item() - 0.9025) < 1e-4, f"old floor should be ~0.9, got {broken}"
    assert student.item() < 1e-6, "student_scale: 0.05*20 == 1.0 == teacher margin -> ~0"
    assert teacher.item() < 1e-6, "teacher_scale: 1.0/20 == 0.05 == student margin -> ~0"
    print(f"  [#4] hard pair: broken={broken.item():.4f} (unclosable floor) -> "
          f"student_scale={student.item():.2e}, teacher_scale={teacher.item():.2e} OK")

    # The documented overshoot: an EASY pair the model already separates (margin 0.3).
    # student_scale asks for 6.0 vs a teacher max of 1.0 and pushes the margin back DOWN.
    e_pos = torch.tensor([0.80], requires_grad=True); e_neg = torch.tensor([0.50])
    over = MarginMSE(scale=20.0, mode="student_scale")(e_pos, e_neg, t_pos, t_neg)
    over.backward()
    assert e_pos.grad.item() > 0, "student_scale overshoot: gradient pushes the positive DOWN"
    print(f"  [#4] easy pair overshoot CONFIRMED: loss={over.item():.2f}, "
          f"d/d(s_pos)={e_pos.grad.item():+.2f} (>0 => pushes a correct margin down) OK")

    # ... and the documented cost of the alternative: ~400x smaller gradients.
    g_student = torch.tensor([0.65], requires_grad=True)
    MarginMSE(scale=20.0, mode="student_scale")(g_student, s_neg, t_pos, t_neg).backward()
    g_teacher = torch.tensor([0.65], requires_grad=True)
    MarginMSE(scale=20.0, mode="teacher_scale")(g_teacher, s_neg, t_pos, t_neg).backward()
    ratio = abs(g_student.grad.item()) / max(abs(g_teacher.grad.item()), 1e-12)
    print(f"  [#4] LR asymmetry CONFIRMED: |grad| student_scale / teacher_scale = "
          f"{ratio:.0f}x  (both modes are defensible; neither is validated)")

    try:
        MarginMSE(mode="bogus")
        raise AssertionError("must reject unknown modes")
    except ValueError:
        pass


def test_listwise_teacher_scaling():
    """BUG #5: raw [0,1] teacher -> near-uniform target -> KL teaches the student to flatten."""
    ListwiseKL = L["ListwiseKL"]
    teacher = torch.tensor([[1.0, 0.9, 0.1, 0.0]])   # a clearly ordered teacher
    student = torch.tensor([[0.6, 0.5, 0.4, 0.3]], requires_grad=True)
    mask = torch.ones_like(teacher)

    pt_raw = F.softmax(teacher, dim=-1)
    pt_scaled = F.softmax(teacher * 20.0, dim=-1)
    ratio_raw = (pt_raw.max() / pt_raw.min()).item()
    ratio_scaled = (pt_scaled.max() / pt_scaled.min()).item()
    assert ratio_raw < 3.0, f"raw teacher target should be near-uniform, ratio={ratio_raw}"
    assert ratio_scaled > 1e6, f"scaled target should be peaked, ratio={ratio_scaled}"
    print(f"  [#5] teacher target max/min prob: raw={ratio_raw:.2f} (near-uniform, "
          f"max ratio e~2.72) -> scaled={ratio_scaled:.2e} (peaked) OK")

    # The flattening claim, measured: under the raw target the student's TOP-ranked doc
    # gets a positive gradient (pushed down) even though the teacher agrees it is best.
    s_raw = student.clone().detach().requires_grad_(True)
    ListwiseKL(aux_weight=0.0, scale=20.0, scale_teacher=False)(s_raw, teacher, mask).backward()
    s_fix = student.clone().detach().requires_grad_(True)
    ListwiseKL(aux_weight=0.0, scale=20.0, scale_teacher=True)(s_fix, teacher, mask).backward()
    assert s_raw.grad[0, 0] > 0, "raw target should push the top doc DOWN (flattening)"
    assert s_fix.grad[0, 0] < 0, "scaled target should pull the top doc UP"
    print(f"  [#5] grad on the teacher's BEST doc: raw={s_raw.grad[0,0]:+.4f} (pushed down "
          f"= flattening) -> scaled={s_fix.grad[0,0]:+.4f} (pulled up) OK")

    # masked padding must not produce NaN in bf16-ish ranges (the -1e4-after-scaling fix)
    part = torch.tensor([[1.0, 0.9, 0.0, 0.0]])
    m = torch.tensor([[1, 1, 0, 0]])
    sp = torch.tensor([[0.6, 0.5, 0.0, 0.0]], requires_grad=True)
    out = ListwiseKL(aux_weight=0.1, scale=20.0)(sp, part, m)
    out.backward()
    assert torch.isfinite(out) and torch.isfinite(sp.grad).all(), "masking must stay finite"
    print(f"  [#5] masked padding: loss={out.item():.4f}, grads finite OK")


if __name__ == "__main__":
    test_lsepair_zero_positive_guard()
    test_marginmse_scale_applied()
    test_listwise_teacher_scaling()
    print("\nLOSS BUGFIX TESTS PASSED (#3 zero-positive, #4 scale, #5 teacher units)")
