"""
dh2.overlap_regression -- measure lexical dependence WITHOUT a threshold.

    nDCG_i  ~  a + b * overlap_i        per query, per arm

`b` is how much an arm leans on word overlap. That is the project's central question said
directly -- did fine-tuning teach lexical matching? -- and it has no threshold, so it
cannot be broken by moving a constant.

WHY THIS REPLACES LOW/HIGH BUCKETING. The bucket is a threshold on a continuum. The split
is `np.median(overlap)` over the evaluated queries (a library default in bucket_queries,
not a derived constant; the pinned 0.375 is just the median of the 123-query slice under
the old tokenizer). Fixing the tokenizer moves 9.2% of queries across it. Re-deriving the
split does not fix that -- it relocates it. A slope uses every query at its true overlap
and throws none of the signal away.

Reads the `per_query` rows that evaluate_cascade now emits. Needs no GPU: once a run is
saved, every re-derivation is free forever.

Stdlib + numpy only. OLS with a bootstrap CI on b (the same 20240611 seed the rest of the
harness bootstraps with, so intervals are comparable).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SEED = 20240611


def fit(overlap: list[float], metric: list[float], n_boot: int = 2000,
        seed: int = SEED) -> dict:
    """OLS metric ~ a + b*overlap, with a bootstrap CI on the slope."""
    x = np.asarray(overlap, dtype=float)
    y = np.asarray(metric, dtype=float)
    n = len(x)
    if n < 3 or float(x.std()) == 0.0:
        return {"n": n, "slope": None, "note": "degenerate: <3 points or zero variance in overlap"}

    def _ols(xi, yi):
        b, a = np.polyfit(xi, yi, 1)
        return float(a), float(b)

    a, b = _ols(x, y)
    r = float(np.corrcoef(x, y)[0, 1])
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if float(x[idx].std()) == 0.0:
            continue
        bs.append(_ols(x[idx], y[idx])[1])
    lo, hi = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))) if bs else (None, None)
    return {"n": n, "intercept": round(a, 4), "slope": round(b, 4),
            "slope_ci95": [round(lo, 4), round(hi, 4)] if bs else None,
            "pearson_r": round(r, 4),
            # A slope whose CI excludes 0 means the arm's quality genuinely tracks lexical
            # overlap. A CI spanning 0 means it does not -- which is the GOOD outcome for a
            # system claiming to bridge a register gap.
            "lexically_dependent": bool(lo is not None and (lo > 0 or hi < 0))}


def fit_report(report_path: str | Path, metric: str = "ndcg@10") -> dict:
    """Fit from a saved eval report (needs the per_query rows)."""
    rep = json.loads(Path(report_path).read_text())
    arm = next(iter(rep.values())) if "per_query" not in rep else rep
    rows = arm.get("per_query")
    if not rows:
        raise SystemExit(
            f"{report_path}: no per_query rows. This report predates per-query saving, so the "
            "regression needs a GPU re-run. New runs carry them.")
    ov = [r["overlap"] for r in rows if metric in r]
    me = [r[metric] for r in rows if metric in r]
    out = fit(ov, me)
    out["metric"] = metric
    out["overlap_split_source"] = arm.get("overlap_split_source", "unknown")
    return out


def compare(report_a: str | Path, report_b: str | Path, metric: str = "ndcg@10",
            label_a: str = "A", label_b: str = "B") -> dict:
    """Two arms: does B lean on lexical overlap more or less than A?

    This is the claim to publish. "Fine-tuning changed the model's dependence on word
    overlap, slope X -> Y" survives a moved constant; "+64% on low-overlap queries" does not.
    """
    fa, fb = fit_report(report_a, metric), fit_report(report_b, metric)
    if fa.get("slope") is None or fb.get("slope") is None:
        return {"error": "degenerate fit", label_a: fa, label_b: fb}
    d = fb["slope"] - fa["slope"]
    return {
        "metric": metric, label_a: fa, label_b: fb,
        "slope_delta": round(d, 4),
        "verdict": (f"{label_b} leans MORE on lexical overlap than {label_a}" if d > 0
                    else f"{label_b} leans LESS on lexical overlap than {label_a}"),
        "caveat": ("Slope CIs here are marginal. To claim a DIFFERENCE, bootstrap the paired "
                   "per-query difference -- the arms are scored on the same queries, so "
                   "comparing two CIs by overlap is the wrong statistic."),
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 2:
        print(json.dumps(fit_report(sys.argv[1]), indent=2))
    elif len(sys.argv) >= 3:
        print(json.dumps(compare(sys.argv[1], sys.argv[2]), indent=2))
    else:
        print(__doc__)
        print("usage: overlap_regression.py <report.json> [<report_b.json>]")
