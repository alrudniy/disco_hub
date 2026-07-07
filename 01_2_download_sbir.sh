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
SBIR_BULK_URL="https://data.www.sbir.gov/awarddatapublic/award_data.csv"

echo "DH_DATA_ROOT = $DH_DATA_ROOT"
echo "REPO         = $REPO"
echo

# ===== 1. SBIR -- single bulk file, ~290 MB, every award WITH abstracts ======
# Page: https://www.sbir.gov/data-resources  ->  Award Data  ->  WITH abstracts (JSON)
echo "=== [1/5] SBIR ==="
if [ -n "$SBIR_BULK_URL" ]; then
    echo "  downloading bulk file ..."
    curl -L --fail -o "$RAW/sbir_bulk.json" "$SBIR_BULK_URL" \
        && echo "  -> $RAW/sbir_bulk.json" \
        || echo "  download FAILED -- check SBIR_BULK_URL"
elif [ -f "$RAW/sbir_bulk.json" ]; then
    echo "  using existing $RAW/sbir_bulk.json"
else
    echo "  SKIP: no file. Download 'Award data WITH abstract (JSON)' from"
    echo "        https://www.sbir.gov/data-resources and save to $RAW/sbir_bulk.json"
fi
echo "  (then a converter remaps it -> $RAW/sbir.jsonl; see notes below)"
echo


# ----------------------------- sizes ----------------------------------------
echo "=== disk usage ==="
du -sh "$RAW"/* "$SNAP" 2>/dev/null
echo
df -h "$DH_DATA_ROOT"
echo
echo "DONE. Parse-ready now: clinicaltrials.jsonl. Need a converter/parser:"
echo "  SBIR (field remap), OpenAlex (stream+filter from snapshot), USPTO (ODP shape)."
