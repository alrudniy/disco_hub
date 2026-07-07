"""
Determinism & reproducibility controls (the investor's question, in code).

WHAT THIS MODULE DOES
    set_global_determinism() sets every knob we *can* control: Python/NumPy/torch
    seeds, deterministic cuDNN/cuBLAS algorithms, disables TF32, and pins the
    cuBLAS workspace. determinism_report() returns a dict describing the active
    configuration so 09_stability_harness.py can print exactly what was in force
    when a measurement was taken.

WHAT IT CANNOT DO (be honest about this with the investor)
    Even with everything below set, byte-identical LLM output is NOT guaranteed
    across runs, because production inference servers batch requests together and
    standard GPU kernels are not "batch-invariant": the floating-point reduction
    order -- and therefore the last-bit value of the logits -- depends on how many
    other requests were in the batch. Tiny float differences can flip the argmax
    in greedy decoding and change a token, which cascades. (Thinking Machines,
    "Defeating Nondeterminism in LLM Inference," 2025.) Batch-invariant kernels
    fix this but cost ~50% throughput.

    Seeds and these flags make *training on a fixed hardware/library stack*
    reproducible. They do not erase cross-GPU, cross-driver, or cross-library
    differences. So the honest, measurable claim is SEMANTIC / FUNCTIONAL
    stability (same evidence, same citations, same recommendation), not byte
    equality -- which is exactly what 09 measures.

IMPORTANT ENV CAVEATS
    * CUBLAS_WORKSPACE_CONFIG and PYTHONHASHSEED must be set BEFORE the process
      starts / before CUDA initializes to take full effect. We set them here too,
      but the Makefile and run scripts also export them up front. If you import
      this module first thing in __main__, you are fine for CUBLAS.
"""
from __future__ import annotations

import os
import random

# Set these as early as possible. They are read at interpreter / CUDA init.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONHASHSEED", "0")

import numpy as np  # noqa: E402  (after env vars on purpose)


def set_global_determinism(seed: int, strict: bool = True) -> dict:
    """
    Configure all reproducibility controls.

    Args:
        seed: the master seed applied to Python, NumPy, and (if present) torch.
        strict: if True, force torch deterministic algorithms (may raise on ops
            with no deterministic implementation, and is slower). Set False to
            warn-only for development.

    Returns:
        The determinism report dict.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch_configured = False
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Deterministic algorithm selection.
        torch.use_deterministic_algorithms(True, warn_only=not strict)
        # cuDNN: pick deterministic kernels and stop autotuning (autotuner picks
        # different algos run-to-run depending on timing).
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # TF32 trades precision for speed and changes results -> off for repro.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch_configured = True
    except ImportError:
        pass  # torch absent (e.g. mock-mode CPU box) -- numpy paths still repro.

    return determinism_report(seed=seed, strict=strict, torch_configured=torch_configured)


def determinism_report(seed: int, strict: bool, torch_configured: bool) -> dict:
    """A machine-readable snapshot of the active reproducibility configuration."""
    report = {
        "seed": seed,
        "strict_deterministic": strict,
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "numpy_version": np.__version__,
        "torch_configured": torch_configured,
        # Honesty fields surfaced in the stability report:
        "guarantees": {
            "byte_identical_same_stack": "embeddings & exact (FAISS Flat) "
            "retrieval: yes. LLM greedy decode: only with batch-invariant "
            "kernels + fixed batch size.",
            "byte_identical_cross_hardware": "no (float non-associativity across "
            "GPU arch / driver / library versions).",
            "semantic_stability": "yes, and measured in 09_stability_harness.py.",
        },
    }
    try:
        import torch

        report["torch_version"] = torch.__version__
        report["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            report["gpu_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        report["torch_version"] = None
        report["cuda_available"] = False
    return report
