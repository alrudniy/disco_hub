# vast_h100 /workspace — code snapshot, 2026-07-17

Code-only snapshot of the rented H200 (vast_h100), taken because that box gets
reclaimed and none of this existed in git. Orphan branch: shares no history with
any other branch here, merges into nothing, overwrites nothing.

| dir | what | source |
|---|---|---|
| `dh_rgcn_pipeline/` | the 01_*–10_* pipeline line, R-GCN oriented | `/workspace/dh_rgcn_pipeline` |
| `discovery_hub_pipeline_2/` | the pipeline_2 line: prod/, stages/, dh2/, HANDOFF.md | `/workspace/discovery_hub_pipeline_2` |
| `_loose_scripts/` | scripts sitting bare in /workspace, belonging to no tree | `/workspace/*.{js,py,sh}` |

## Deliberately NOT included

- **`dh2_takehome/`** — 141 MB and **zero code files**. Its entire content is one
  data file, `data/multi_positive_labels_v1.jsonl` (140.1 MB), which exceeds
  GitHub's 100 MB hard limit. It is data, not source. **It is still only on the
  vast box and will be lost when that box is reclaimed** — if it matters, it needs
  object storage, not git.
- `dh_data/` (29 GB), `models/` (23 GB), `dh_data_v2/` (13 GB), `dh_merged/`
  (1.5 GB) — data and model weights.
- `.venv_preflight/` — 34 MB of numpy/scipy binaries.
- `dh_rgcn_pipeline/key-google-cloud-big-query.json` — a **live GCP
  service-account private key** (project `summer-dssi`). Excluded at rsync level,
  verified absent before commit. **It needs rotating** — it is also committed in
  the root commit of `discovery_hub_pipeline` and therefore already on GitHub.

Snapshot only. Nothing here was run, tested, or reviewed as part of taking it.
