#!/usr/bin/env python3
"""p1_2 -- run the objective bakeoff. Screen arms on 0.6B, then train winner on 4B."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dh2 import config2 as C
from dh2 import train_bakeoff as TB
from discovery_hub.schema import read_docs


def _doc_text_map(docs_path: Path) -> dict:
    return {d.doc_id: (d.embedding_text or f"{d.title}\n{d.abstract}") for d in read_docs(docs_path)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", default=str(C.LABELS_DIR / "train_labels_v1.jsonl"))
    ap.add_argument("--docs", default=str(C.PROD_DOCS))
    ap.add_argument("--fn-flag", default=str(C.LABELS_DIR / "fn_flag_v1.json"))
    ap.add_argument("--stage", choices=["screen", "final"], default="screen",
                    help="screen: all arms on 0.6B subset; final: one arm on 4B")
    ap.add_argument("--arms", default="", help="comma list to restrict arms")
    ap.add_argument("--winner", default="", help="arm name to train on 4B in final stage")
    ap.add_argument("--base-model", default="", help="override backbone")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-queries", type=int, default=0)
    args = ap.parse_args()

    C.ensure_dirs()
    docs_text = _doc_text_map(Path(args.docs))
    fn_flag = json.loads(Path(args.fn_flag).read_text()) if Path(args.fn_flag).exists() else {}
    only = [a for a in args.arms.split(",") if a] or None

    if args.stage == "screen":
        base = args.base_model or C.MODEL_06B
        max_q = args.max_queries or C.BAKEOFF.screen_queries
        results = {}
        for name, spec in TB.iter_arms(only):
            out_dir = str(C.BAKEOFF_DIR / f"screen_{name}")
            print(f"[p1_2] SCREEN arm {name} on 0.6B ({base}) ...")
            final = TB.train_arm(name, spec, args.labels, base, out_dir, docs_text,
                                 is_4b=False, device=args.device, max_queries=max_q,
                                 fn_flag=fn_flag)
            results[name] = final
        Path(C.BAKEOFF_DIR / "screen_models.json").write_text(json.dumps(results, indent=2))
        print(f"[p1_2] screen models: {json.dumps(results, indent=2)}")
        print("[p1_2] next: eval each with p1_3, pick winner, run --stage final --winner <arm>")
    else:
        assert args.winner, "final stage needs --winner <arm_name>"
        base = args.base_model or C.MODEL_4B
        spec = dict(TB.ARMS[args.winner])
        out_dir = str(C.BAKEOFF_DIR / f"final_{args.winner}_4b")
        print(f"[p1_2] FINAL arm {args.winner} on 4B ({base}) ...")
        final = TB.train_arm(args.winner, spec, args.labels, base, out_dir, docs_text,
                             is_4b=True, device=args.device, max_queries=args.max_queries,
                             fn_flag=fn_flag)
        Path(C.BAKEOFF_DIR / "final_model.json").write_text(json.dumps({"winner": args.winner, "model": final}))
        print(f"[p1_2] final 4B model -> {final}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
