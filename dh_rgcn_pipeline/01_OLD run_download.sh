#!/bin/bash
# =============================================================================
# run_download.sh  --  Discovery Hub pipeline, stage 01 (real, keyless sources)
#
# Pulls ClinicalTrials.gov v2, SBIR.gov, and OpenAlex into JSONL on Anvil
# scratch. No API key required for any of these three. USPTO and AUTM are
# handled separately (see notes at the bottom).
#
# Submit with:   sbatch run_download.sh
# Watch with:    squeue -u $USER        # and: tail -f dh_download_<jobid>.out
# =============================================================================

#SBATCH -A cis251425              # <-- your allocation/account. Confirm the exact
                                  #     string with `mybalance` if sbatch rejects it.
#SBATCH -p shared                 # per-core billing (NOT wholenode = 128 cores)
#SBATCH -J dh_download
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4         # plenty; this is I/O-bound, not compute
#SBATCH --time=02:00:00           # generous; job ends as soon as it finishes
#SBATCH -o dh_download_%j.out     # stdout  (%j = job id)
#SBATCH -e dh_download_%j.err     # stderr
# --- optional email notifications: uncomment and set your address ---
# #SBATCH --mail-type=END,FAIL
# #SBATCH --mail-user=you@purdue.edu

set -euo pipefail

# ----------------------------------------------------------------------------
# 1. Where the repo lives. Adjust if you unzipped it somewhere other than
#    $PROJECT (e.g. $HOME/discovery_hub_pipeline).
# ----------------------------------------------------------------------------
REPO="$PROJECT/discovery_hub_pipeline"

# ----------------------------------------------------------------------------
# 2. Downloads go to scratch. config.py reads DH_DATA_ROOT (default ./data),
#    so this single line redirects every stage's output. ensure_dirs() creates
#    the raw/ subfolder automatically.
# ----------------------------------------------------------------------------
export DH_DATA_ROOT="$SCRATCH/discovery_hub/data"
export PYTHONHASHSEED=0           # parity with the Makefile's reproducibility env

# ----------------------------------------------------------------------------
# 3. Python from your conda env. Using the env's interpreter by full path is the
#    most robust option — no need to know the exact conda-env module name.
#    (The "module" equivalent is commented below if you prefer it.)
# ----------------------------------------------------------------------------
module purge
module load conda 2>/dev/null || true
PYTHON="$PROJECT/apps/dh/bin/python"
# Alternative, the module way:
#   module use "$PROJECT/etc/modules"
#   module load conda-env/dh-py3.11    # exact name from: module avail conda-env
#   PYTHON=python

# ----------------------------------------------------------------------------
# 4. Fail fast with a clear message if the env or deps aren't ready.
# ----------------------------------------------------------------------------
if [ ! -x "$PYTHON" ]; then
    echo "ERROR: $PYTHON not found. Was the env created at \$PROJECT/apps/dh?" >&2
    exit 1
fi
if ! "$PYTHON" -c "import requests" 2>/dev/null; then
    echo "ERROR: 'requests' is not installed in the env. Run:" >&2
    echo "   $PROJECT/apps/dh/bin/pip install -r $REPO/requirements.txt" >&2
    exit 1
fi

# ----------------------------------------------------------------------------
# 5. Run stage 01 against the three keyless sources (MVP volumes).
# ----------------------------------------------------------------------------
cd "$REPO"
echo "=== stage 01 download starting $(date) ==="
echo "    DH_DATA_ROOT = $DH_DATA_ROOT"
echo "    python       = $PYTHON"
echo

"$PYTHON" 01_download_data.py --sources clinicaltrials,sbir,openalex --mvp

# ----------------------------------------------------------------------------
# 6. Show what landed.
# ----------------------------------------------------------------------------
echo
echo "=== results in $DH_DATA_ROOT/raw/ ==="
ls -lh "$DH_DATA_ROOT/raw/" || true
echo
echo "=== line counts ==="
wc -l "$DH_DATA_ROOT"/raw/*.jsonl 2>/dev/null || true
echo
echo "=== stage 01 download done $(date) ==="

# =============================================================================
# USPTO  -- needs an ODP API key (data.uspto.gov/apis/getting-started, requires
#           a USPTO.gov + ID.me verified account), AND a new parser: the odp-bulk
#           backend downloads the whole Patent File Wrapper dataset in a schema
#           that stage 02's parse_uspto does not yet read. Defer until that's built.
#
# AUTM   -- no bulk API. Drop the students' pre-scraped JSONL in by hand:
#               $SCRATCH/discovery_hub/data/raw/autm.jsonl
#           parse_uspto/parse_autm already accept that flat shape.
# =============================================================================
