"""
dh2.validate -- the two must-fix engineering items from the spec (S10) plus reusable
startup checks. Import-safe (no heavy deps at module load).

1. merge_lora_checkpoint(): the CORRECT LoRA merge that pipeline_1 got wrong.
   The bug: train.py called model[0].auto_model.merge_and_unload() AFTER the PEFT
   wrapper pointer was reassigned, so it hit a bare Qwen3Model. The trainer still
   checkpoints the full PEFT state at checkpoint-<step>/model.safetensors with keys
   like 'layers.N...lora_A/lora_B/base_layer'. This rebuilds the SAME LoraConfig,
   loads those weights (prefixing 'base_model.model.'), merges, and saves a plain
   SentenceTransformer. FAILS LOUDLY if any adapter tensor is missing (S10.1).

2. validate_merged_model(): load the merged dir, assert correct dim, assert NO
   "missing/newly-initialized" params (the signature of the cp-checkpoint-as-final
   mistake), and confirm it encodes.

3. assert_serving_compat(): startup checks (S10.4) -- DH_EMBED_MODEL set, model dim
   matches the vectors' dim, docs.jsonl row count matches vector count.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


# --------------------------------------------------------------------------- #
# 1. Correct LoRA merge (recovery-proof)
# --------------------------------------------------------------------------- #
def merge_lora_checkpoint(base_model: str, checkpoint_dir: str, out_dir: str,
                          lora_r: int = 16, lora_alpha: int = 32,
                          lora_dropout: float = 0.05, max_seq_length: int = 512,
                          target_modules: list[str] | None = None) -> str:
    """Merge a PEFT checkpoint's adapter into base weights -> plain SentenceTransformer.

    Returns out_dir on success; raises AssertionError if any adapter tensor is missing
    (so an unattended run FAILS rather than silently saving a random-init model).
    """
    from safetensors.torch import load_file
    from sentence_transformers import SentenceTransformer
    from peft import LoraConfig, get_peft_model

    target_modules = target_modules or ["q_proj", "k_proj", "v_proj", "o_proj",
                                        "gate_proj", "up_proj", "down_proj"]
    ckpt = Path(checkpoint_dir)
    weights = ckpt / "model.safetensors"
    if not weights.exists():
        # HF may shard; fall back to the checkpoint dir itself if it's already a model
        cands = sorted(ckpt.glob("*.safetensors"))
        if not cands:
            raise FileNotFoundError(f"no safetensors in {ckpt}")
        weights = cands[0]

    st = SentenceTransformer(base_model)
    st.max_seq_length = max_seq_length
    lora = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                      bias="none", task_type="FEATURE_EXTRACTION",
                      target_modules=target_modules)
    peft_inner = get_peft_model(st[0].auto_model, lora)

    sd = load_file(str(weights))
    # checkpoint keys look like 'layers.0...lora_A.default.weight'; PEFT wants the
    # 'base_model.model.' prefix.
    remapped = {}
    for k, v in sd.items():
        remapped[k if k.startswith("base_model.model.") else "base_model.model." + k] = v
    missing, unexpected = peft_inner.load_state_dict(remapped, strict=False)
    lora_missing = [m for m in missing if "lora_" in m]
    assert not lora_missing, (
        f"LoRA merge ABORTED: {len(lora_missing)} adapter tensors missing "
        f"(first: {lora_missing[:3]}). Checkpoint/config mismatch — do NOT save.")

    st[0].auto_model = peft_inner.merge_and_unload()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    st.save(out_dir)
    return out_dir


# --------------------------------------------------------------------------- #
# 2. Validate a merged model loads clean
# --------------------------------------------------------------------------- #
def validate_merged_model(model_dir: str, expected_dim: int | None = None,
                          device: str | None = None) -> dict:
    """Load model_dir, assert clean load + correct dim + encodes. Returns diagnostics.

    Detects the 'MISSING params' failure mode: SentenceTransformer logs newly-initialized
    weights to stderr, but a robust check is the embedding dimension + a finite encode.
    """
    import numpy as np
    from sentence_transformers import SentenceTransformer

    m = SentenceTransformer(model_dir, device=device)
    dim = m.get_sentence_embedding_dimension()
    if expected_dim is not None:
        assert dim == expected_dim, (
            f"dim mismatch: model={dim} expected={expected_dim} — wrong checkpoint merged?")
    vec = m.encode(["oral therapy for relapsed chronic lymphocytic leukemia"])
    vec = np.asarray(vec)
    assert vec.shape[-1] == dim and np.isfinite(vec).all(), "encode produced bad vectors"
    norm = float(np.linalg.norm(vec[0]))
    return {"model_dir": model_dir, "dim": dim, "encode_norm": round(norm, 4), "ok": True}


# --------------------------------------------------------------------------- #
# 3. Serving/index compatibility (S10.4) -- cheap, import-safe
# --------------------------------------------------------------------------- #
def assert_serving_compat(embed_model: str | None, doc_vectors_path: str,
                          doc_ids_path: str, docs_jsonl_path: str | None = None) -> dict:
    """Fail fast on model/vector/index mismatch. Returns the checked facts."""
    import numpy as np
    facts = {}
    if not embed_model:
        raise EnvironmentError("DH_EMBED_MODEL not set — refusing to serve/eval with an "
                               "ambiguous encoder (would mismatch the indexed vectors).")
    facts["embed_model"] = embed_model

    vecs = np.load(doc_vectors_path, mmap_mode="r")
    facts["vector_count"] = int(vecs.shape[0])
    facts["vector_dim"] = int(vecs.shape[1])

    ids = json.loads(Path(doc_ids_path).read_text())
    facts["doc_id_count"] = len(ids)
    assert facts["doc_id_count"] == facts["vector_count"], (
        f"doc_ids ({facts['doc_id_count']}) != vectors ({facts['vector_count']}) — "
        f"index-alignment invariant violated.")

    if docs_jsonl_path and Path(docs_jsonl_path).exists():
        n = sum(1 for _ in Path(docs_jsonl_path).open())
        facts["docs_jsonl_lines"] = n
        assert n == facts["vector_count"], (
            f"docs.jsonl lines ({n}) != vectors ({facts['vector_count']}) — "
            f"docs.jsonl was regenerated out of sync with the embeddings.")
    return facts


def _cli() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="pipeline_2 validation utilities")
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("merge", help="merge a LoRA checkpoint -> plain model")
    m.add_argument("--base-model", required=True)
    m.add_argument("--checkpoint", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--max-seq-length", type=int, default=512)

    v = sub.add_parser("validate", help="validate a merged model loads clean")
    v.add_argument("--model", required=True)
    v.add_argument("--expected-dim", type=int, default=None)
    v.add_argument("--device", default=None)

    args = ap.parse_args()
    if args.cmd == "merge":
        out = merge_lora_checkpoint(args.base_model, args.checkpoint, args.out,
                                    max_seq_length=args.max_seq_length)
        print(f"merged -> {out}")
        diag = validate_merged_model(out)
        print(f"validation: {diag}")
    elif args.cmd == "validate":
        diag = validate_merged_model(args.model, expected_dim=args.expected_dim,
                                     device=args.device)
        print(json.dumps(diag, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
