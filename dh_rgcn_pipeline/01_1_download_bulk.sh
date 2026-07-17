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

# ===== 2. ClinicalTrials.gov -- full pull via v2 API, no key needed ==========
# Output is already in parse_clinicaltrials's shape. The huge --limit just
# removes the MVP cap; the fetcher pages with pageToken and stops at the end.
# NOTE: 01_download_data_v2.py filters by condition (pharmaceutical/cancer/antibody).
#       For literally ALL ~590k studies, delete that query.cond filter in the file.

# AR : DOWNLOADED ONCE, SKIPPING FOR NOW
#echo "=== [2/5] ClinicalTrials.gov (full) ==="
#( cd "$REPO" && DH_DATA_ROOT="$DH_DATA_ROOT" "$PY" 01_download_data_v2.py \
#      --sources clinicaltrials --limit 1000000 ) \
#    || echo "  ClinicalTrials pull FAILED"
#echo

# ===== 3. OpenAlex -- full S3 snapshot (~330 GB gz works) ====================
# Needs the AWS CLI (no credentials; --no-sign-request). pip install awscli

#echo "=== [3/5] OpenAlex snapshot ==="
#if command -v aws >/dev/null 2>&1; then
#    echo "  syncing s3://openalex/data/works/ (~330 GB, long-running) ..."
#    aws s3 sync "s3://openalex/data/jsonl/works/" "$SNAP/works/" --no-sign-request \
#    || echo "  s3 sync FAILED / interrupted (re-run to resume)"
    # For EVERYTHING (authors, sources, institutions, concepts, topics) sync the
    # whole bucket instead:
    #   aws s3 sync "s3://openalex/data/" "$SNAP/" --no-sign-request
#else
#    echo "  SKIP: aws CLI not found. Install with:  pip install awscli"
#fi
#echo "  (gzipped JSONL part files. DO NOT flatten all 250M works into one file --"
#echo "   stream + filter to the medicine subset when building $RAW/openalex.jsonl)"
#echo

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

# ===== 5. AUTM -- students' scrape, no public bulk ===========================
echo "=== [5/5] AUTM ==="
if [ -f "$RAW/autm.jsonl" ]; then
    echo "  present: $RAW/autm.jsonl"
else
    echo "  MISSING: drop the students' scraped JSONL at $RAW/autm.jsonl"
fi
echo

# ----------------------------- sizes ----------------------------------------
echo "=== disk usage ==="
du -sh "$RAW"/* "$SNAP" 2>/dev/null
echo
df -h "$DH_DATA_ROOT"
echo
echo "DONE. Parse-ready now: clinicaltrials.jsonl. Need a converter/parser:"
echo "  SBIR (field remap), OpenAlex (stream+filter from snapshot), USPTO (ODP shape)."
