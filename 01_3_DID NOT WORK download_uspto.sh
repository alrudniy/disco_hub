#!/bin/bash
# =============================================================================
# download_bulk.sh  --  pull ALL Discovery Hub sources in BULK onto Drew.
#
# Each source has a different bulk mechanism; this orchestrates all five. Sources
# run independently (one failing doesn't abort the rest). Rough disk footprint:
#   SBIR ~0.3 GB | ClinicalTrials ~few GB | OpenAlex ~330 GB gz (~1.6 TB unzip)
#   USPTO ~0.5 TB unzip | AUTM small   ->  ~2.1 TB total (fits 4 TB).
#
# IMPORTANT -- disk is not the only limit. Drew's 2x12 GB GPUs CANNOT embed/index
# all ~250M OpenAlex works or ~500 GB USPTO full text. config.py says to filter
# OpenAlex to the ~43M medicine+biology subset (or 1-5M for the MVP) BEFORE stage
# 04. So it's fine to land the raw bulk here, but you will filter before embedding.
#
# Run:  bash download_bulk.sh      (edit the CONFIG block first)
# =============================================================================
set -uo pipefail   # deliberately NOT -e: keep going if one source fails

# ----------------------------- CONFIG ---------------------------------------
DH_DATA_ROOT="${DH_DATA_ROOT:-$HOME/discovery_hub/data}"
REPO="${REPO:-$HOME/discovery_hub_pipeline}"
PY="${PY:-python3}"                      # set to the venv python if you use one
RAW="$DH_DATA_ROOT/raw"
SNAP="$DH_DATA_ROOT/openalex_snapshot"
mkdir -p "$RAW" "$SNAP"

# SBIR has no stable direct link -- the file is behind a download button. Either
# set SBIR_BULK_URL to that button's link (right-click -> copy link), or just
# download it in a browser and drop it at $RAW/sbir_bulk.json yourself.
SBIR_BULK_URL="${SBIR_BULK_URL:-}"

echo "DH_DATA_ROOT = $DH_DATA_ROOT"
echo "REPO         = $REPO"
echo


# ===== 4. USPTO -- ODP bulk datasets (needs ODP key) =========================
# Reuses the odp-bulk backend already in 01_download_data_v2.py.
echo "=== [4/5] USPTO (ODP bulk) ==="
if [ -n "${USPTO_ODP_API_KEY:-}" ]; then
    ( cd "$REPO" && DH_DATA_ROOT="$DH_DATA_ROOT" "$PY" 01_download_data_v2.py \
          --sources uspto --uspto-backend odp-bulk --odp-download ) \
        || echo "  USPTO ODP download FAILED"
else
    echo "  SKIP: USPTO_ODP_API_KEY not set."
    echo "        Get an ODP key at data.uspto.gov (USPTO.gov account + ID.me), then"
    echo "        export USPTO_ODP_API_KEY=... and re-run."
fi
echo "  NOTE: bulk Patent File Wrapper JSON != parse_uspto's shape -- needs a"
echo "        dedicated parser before it feeds stage 02."
echo


# ----------------------------- sizes ----------------------------------------
echo "=== disk usage ==="
du -sh "$RAW"/* "$SNAP" 2>/dev/null
echo
df -h "$DH_DATA_ROOT"
echo
echo "DONE. Parse-ready now: clinicaltrials.jsonl. Need a converter/parser:"
echo "  SBIR (field remap), OpenAlex (stream+filter from snapshot), USPTO (ODP shape)."
