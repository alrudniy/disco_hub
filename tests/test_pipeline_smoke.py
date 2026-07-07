"""
End-to-end smoke test of the mock pipeline.

Runs all nine stages against a tiny synthetic corpus in a temp data root and
asserts the determinism properties the investor cares about:
  * embeddings are bit-identical across runs,
  * retrieval is exactly reproducible,
  * citation sets are stable.

Run with:  python -m pytest tests/ -q   (or: make smoke)
"""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load(name_py, mod_name):
    spec = importlib.util.spec_from_file_location(mod_name, ROOT / name_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(script, *args, env):
    r = subprocess.run([sys.executable, str(ROOT / script), *args],
                       capture_output=True, text=True, env=env, cwd=ROOT)
    assert r.returncode == 0, f"{script} failed:\n{r.stdout}\n{r.stderr}"
    return r.stdout


def test_pipeline_end_to_end(tmp_path):
    env = dict(os.environ)
    env["DH_DATA_ROOT"] = str(tmp_path / "data")
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    env["PYTHONHASHSEED"] = "0"

    # Stages 01-06 build the artifacts.
    _run("01_download_data.py", "--mock", "--mvp", "--limit", "60", env=env)
    _run("02_parse_normalize.py", env=env)
    _run("03_build_graph.py", "--validate", env=env)
    _run("04_generate_embeddings.py", "--mock", env=env)
    _run("05_build_index.py", env=env)
    _run("06_train_rgcn.py", "--mock", "--steps", "2", env=env)

    data = tmp_path / "data"
    assert (data / "normalized" / "docs.jsonl").exists()
    assert (data / "graph" / "graph_meta.json").exists()
    vecs = np.load(data / "embeddings" / "doc_vectors.npy")
    assert vecs.shape[0] > 0 and vecs.shape[1] == 1024
    # normalized vectors
    np.testing.assert_allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)

    # Stage 09 produces the stability report; assert perfect mock stability.
    os.environ["DH_DATA_ROOT"] = str(data)  # for in-process module import below
    _run("09_stability_harness.py", "--mock", "--runs", "3", env=env)
    report = json.loads((data / "reports" / "stability_report.json").read_text())
    assert report["embedding"]["exact_match_all_runs"] is True
    assert report["embedding"]["max_cosine_drift"] < 1e-6
    assert report["retrieval"]["mean_jaccard"] == 1.0
    assert report["retrieval"]["exact_order_match_rate"] == 1.0
    assert report["explanation"]["citation_set_jaccard"] == 1.0


def test_embedder_is_deterministic():
    emb_mod = _load("discovery_hub/embedding.py", "dh_embedding") \
        if False else None  # use package import instead
    sys.path.insert(0, str(ROOT))
    from discovery_hub.embedding import MockEmbedder
    e = MockEmbedder()
    a = e.encode(["kinase inhibitor", "monoclonal antibody"])
    b = e.encode(["kinase inhibitor", "monoclonal antibody"])
    assert np.array_equal(a, b)
