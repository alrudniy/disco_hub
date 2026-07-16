#!/usr/bin/env python3
"""
A compact, CSR-backed view of the merged knowledge graph -- the backbone of the
expertise-gap agent.

WHY THIS EXISTS. The expertise-gap question ("what does this company cover, what
does it not, and who covers the gap") is a graph-traversal question, and the
traversal has to be fast enough to sit inside an interactive agent. The merged
graph is 1,489,785 nodes / 4,520,029 edges. Holding that as Python dicts of lists
costs several GB and seconds to build; the same adjacency as CSR (an int64
indptr + an int32 indices array per relation) is ~300 MB on disk and loads in
about a second. Neighbours of a node are then a contiguous slice, with no
per-edge Python object anywhere.

WHAT THE DATA ACTUALLY SAYS (verified by a full scan of all 4.52M edges -- the
schema notes in the spec do not match the file, so these are the measured facts):

    edge key is "rel", NOT "relation"
    technology --located_at-->      facility       1,187,011
    technology --invented_by-->     inventor       1,164,355
    inventor   --affiliated_with--> inventor         718,544   <- co-inventor noise
    inventor   --affiliated_with--> organization     636,675
    technology --assigned_to-->     organization     550,574
    technology --investigated_by--> expert           262,551
    technology --assigned_to-->     inventor             319   <- noise, dropped

THREE THINGS THE SPEC GETS WRONG, AND WHAT WE DO INSTEAD:

  * The spec writes `org --assigned_to--> tech`. The direction in the data is the
    reverse: `technology --assigned_to--> organization`. A portfolio ("what does
    this org own") therefore needs a REVERSE index, which is `org_to_techs` below
    and is the single most important map in this file.

  * The spec's step 6 routes `expert --affiliated_with--> org`. THAT PATH DOES
    NOT EXIST: expert nodes have zero affiliated_with edges (full-scan verified).
    The only route from an expert to an org is through the technology they
    investigated -- see `orgs_of_expert()`, which implements it.

  * `affiliated_with` is not one relation. 718,544 of its edges are
    inventor->inventor, which is co-authorship/co-invention, NOT affiliation.
    Only the inventor->organization half is a real affiliation, so
    `inventor_to_orgs` filters on the destination's node type. Including the
    inventor->inventor edges would silently turn "who does this person work for"
    into "who has this person published with".

ORG IDENTITY IS FRAGMENTED and we only partly fix it. "bristol myers squibb"
(826 techs) and "bristol myers squibb co" are separate nodes with separate
portfolios; so are "yale university" / "univ yale" / "yale univ". We canonicalize
by stripping trailing corporate suffixes, which merges those. We deliberately do
NOT merge subsidiaries -- "Juno Therapeutics, Inc., a Bristol-Myers Squibb
Company" is a legally distinct entity, and folding it into the parent would
overstate the portfolio to someone who can check. The canonicalization is
persisted in the artifact so a reader can audit exactly what got merged.

Build (a few minutes, ~2 GB peak RSS):
    DH_DATA_ROOT=/home/alex/discovery_hub/data \
    DH_GRAPH_DIR=/home/alex/discovery_hub/data_merged/graph \
    DH_ARTIFACT_DIR=/home/alex/discovery_hub/data_merged/artifacts \
    venv/bin/python agents/graph_index.py --build
"""
from __future__ import annotations

import argparse
import json
import sys
from array import array
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from discovery_hub import config
from discovery_hub.entity_link import normalize

ARTIFACT_NAME = "graph_index.npz"
FORMAT_VERSION = 1

# Node types, frozen in this order: the int8 `ntype` code is this list's index,
# and it is written into the artifact, so appending is safe and reordering is not.
NTYPES = ("technology", "inventor", "organization", "expert", "facility")
NTYPE_TO_CODE = {t: i for i, t in enumerate(NTYPES)}

# Trailing tokens stripped to canonicalize an org label. These are legal-form
# suffixes: dropping them merges "bristol myers squibb co" into "bristol myers
# squibb" without merging two different companies, because a legal form is never
# the distinguishing part of a name. Stripping is iterative ("foo inc co" ->
# "foo"), and never strips the last remaining token.
_ORG_SUFFIXES = {"co", "inc", "corp", "corporation", "llc", "ltd", "plc",
                 "gmbh", "ag", "sa", "nv", "holdings", "company", "the"}


def canonical_org_key(label: str) -> str:
    """
    Canonical key for an org label: normalized, with leading "the" and trailing
    corporate suffixes stripped. Pure and deterministic.

    This is a SUFFIX rule, not a substring rule, which is what keeps subsidiaries
    separate: "adnexus a bristol myers squibb r d company" loses only "company"
    and stays its own entity, while "bristol myers squibb co" collapses onto
    "bristol myers squibb". Returns the normalized label unchanged if stripping
    would leave nothing.
    """
    toks = normalize(label).split()
    while toks and toks[0] == "the":
        toks = toks[1:]
    while len(toks) > 1 and toks[-1] in _ORG_SUFFIXES:
        toks = toks[:-1]
    return " ".join(toks) if toks else normalize(label)


# --------------------------------------------------------------------------- #
# String packing. A (1.49M,) numpy array of unicode strings is fixed-width and
# would cost >1 GB for the longest label; packing to one utf-8 blob + int64
# offsets costs the actual bytes and unpacks lazily.
# --------------------------------------------------------------------------- #
def _pack_strings(items: list[str]) -> tuple[np.ndarray, np.ndarray]:
    encoded = [s.encode("utf-8") for s in items]
    offsets = np.zeros(len(encoded) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(b) for b in encoded], dtype=np.int64)
    return np.frombuffer(b"".join(encoded), dtype=np.uint8), offsets


def _unpack_string(blob: np.ndarray, offsets: np.ndarray, i: int) -> str:
    return blob[offsets[i]:offsets[i + 1]].tobytes().decode("utf-8")


def _unpack_all(blob: np.ndarray, offsets: np.ndarray) -> list[str]:
    raw = blob.tobytes()
    return [raw[offsets[i]:offsets[i + 1]].decode("utf-8")
            for i in range(len(offsets) - 1)]


# --------------------------------------------------------------------------- #
# CSR construction
# --------------------------------------------------------------------------- #
def _build_csr(src: np.ndarray, dst: np.ndarray,
               n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """
    (src, dst) edge pairs -> CSR (indptr int64, indices int32), sorted by
    (src, dst) and deduplicated.

    Sorting by dst within src makes neighbour lists deterministic (ties broken by
    node row, i.e. by id), which stage 09 cares about. Dedup matters because the
    merge rewired 1.07M edges and can leave exact duplicates: a duplicated
    assigned_to edge would double-count a technology in a portfolio.
    """
    if len(src) == 0:
        return np.zeros(n_nodes + 1, dtype=np.int64), np.zeros(0, dtype=np.int32)
    order = np.lexsort((dst, src))
    src_s, dst_s = src[order], dst[order]
    keep = np.ones(len(src_s), dtype=bool)
    keep[1:] = (src_s[1:] != src_s[:-1]) | (dst_s[1:] != dst_s[:-1])
    src_s, dst_s = src_s[keep], dst_s[keep]
    indptr = np.zeros(n_nodes + 1, dtype=np.int64)
    np.cumsum(np.bincount(src_s, minlength=n_nodes), out=indptr[1:])
    return indptr, dst_s.astype(np.int32)


# The eight adjacency maps the expertise-gap agent traverses. Each is stored as
# "<name>_indptr" / "<name>_indices" in the npz.
_CSR_NAMES = ("org_to_techs", "tech_to_orgs", "tech_to_inventors",
              "inventor_to_techs", "inventor_to_orgs", "org_to_inventors",
              "tech_to_experts", "expert_to_techs")


# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #
def build(graph_dir: Path, artifact_dir: Path, out: Path | None = None) -> dict:
    """
    Stream nodes.jsonl + edges.jsonl and write the compact index. Edges are read
    line by line and accumulated into typed int32 arrays -- 4.5M edge dicts would
    be several GB and there is no reason for any of them to coexist.

    Returns a summary dict (also printed by the CLI).
    """
    out = out or (artifact_dir / ARTIFACT_NAME)
    nodes_path, edges_path = graph_dir / "nodes.jsonl", graph_dir / "edges.jsonl"
    node_ids_path = artifact_dir / "node_ids.json"

    # ---- Row order comes from node_ids.json, NOT from nodes.jsonl. -----------
    # That file is row-aligned to rgcn_node_emb.npy, so using its order is what
    # lets the agent index an R-GCN embedding row directly by node row. We assert
    # the alignment rather than assuming it.
    print(f"[1/4] reading row order from {node_ids_path}", flush=True)
    with node_ids_path.open("r", encoding="utf-8") as fh:
        node_ids: list[str] = json.load(fh)
    n_nodes = len(node_ids)
    row_of: dict[str, int] = {nid: i for i, nid in enumerate(node_ids)}
    if len(row_of) != n_nodes:
        raise ValueError(f"node_ids.json has duplicate ids: {n_nodes} ids, "
                         f"{len(row_of)} unique")

    # ---- Pass 1: node attributes --------------------------------------------
    print(f"[2/4] streaming {nodes_path}", flush=True)
    ntype = np.full(n_nodes, -1, dtype=np.int8)
    labels: list[str] = [""] * n_nodes
    order_mismatches = 0
    seen = 0
    with nodes_path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if not line.strip():
                continue
            nd = json.loads(line)
            nid = nd["node_id"]
            row = row_of.get(nid)
            if row is None:
                raise ValueError(f"nodes.jsonl node {nid!r} absent from node_ids.json")
            if row != i:
                order_mismatches += 1
            ntype[row] = NTYPE_TO_CODE[nd["ntype"]]
            labels[row] = nd.get("label") or nid.split(":", 1)[-1]
            seen += 1
    if seen != n_nodes:
        raise ValueError(f"nodes.jsonl has {seen} nodes, node_ids.json has {n_nodes}")
    if (ntype < 0).any():
        raise ValueError(f"{int((ntype < 0).sum())} nodes never got an ntype")

    tech_code = NTYPE_TO_CODE["technology"]
    org_code = NTYPE_TO_CODE["organization"]
    inv_code = NTYPE_TO_CODE["inventor"]
    exp_code = NTYPE_TO_CODE["expert"]

    # ---- doc_id <-> tech row -------------------------------------------------
    # A technology node_id is "technology:<doc_id>", so the doc_id is everything
    # after the first colon. (nodes.jsonl also carries an explicit doc_id field on
    # technology nodes; the split is used so the map is derivable from the row
    # order alone, and the two agree.)
    tech_rows = np.flatnonzero(ntype == tech_code).astype(np.int32)
    tech_doc_ids = [node_ids[r].split(":", 1)[1] for r in tech_rows]

    # ---- Pass 2: edges -------------------------------------------------------
    print(f"[3/4] streaming {edges_path}", flush=True)
    pairs: dict[str, tuple[array, array]] = {
        k: (array("i"), array("i")) for k in
        ("assigned_to", "invented_by", "affiliated_with", "investigated_by")
    }
    dropped_assigned_to_nonorg = 0
    dropped_affil_noninv_org = 0
    n_edges = 0
    with edges_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            e = json.loads(line)                 # NB: the key is "rel", not "relation"
            rel = e["rel"]
            bucket = pairs.get(rel)
            if bucket is None:                   # located_at: not traversed by this agent
                n_edges += 1
                continue
            s, d = row_of.get(e["src"]), row_of.get(e["dst"])
            n_edges += 1
            if s is None or d is None:
                raise ValueError(f"edge references unknown node: {e}")
            if rel == "assigned_to" and ntype[d] != org_code:
                # 319 technology --assigned_to--> inventor edges exist. An
                # assignee that is a person is not an org portfolio; drop them.
                dropped_assigned_to_nonorg += 1
                continue
            if rel == "affiliated_with" and not (ntype[s] == inv_code
                                                 and ntype[d] == org_code):
                # The 718,544 inventor->inventor edges land here: co-invention,
                # not affiliation. See the module docstring.
                dropped_affil_noninv_org += 1
                continue
            bucket[0].append(s)
            bucket[1].append(d)

    def _np(rel: str) -> tuple[np.ndarray, np.ndarray]:
        s, d = pairs[rel]
        return (np.frombuffer(s, dtype=np.int32).copy(),
                np.frombuffer(d, dtype=np.int32).copy())

    at_s, at_d = _np("assigned_to")        # technology -> organization
    ib_s, ib_d = _np("invented_by")        # technology -> inventor
    aw_s, aw_d = _np("affiliated_with")    # inventor   -> organization
    inv_s, inv_d = _np("investigated_by")  # technology -> expert

    print("[4/4] building CSR adjacency", flush=True)
    csr = {
        # The portfolio map: REVERSE of assigned_to, because the edge runs
        # tech -> org and the question runs org -> techs.
        "org_to_techs": _build_csr(at_d, at_s, n_nodes),
        "tech_to_orgs": _build_csr(at_s, at_d, n_nodes),
        "tech_to_inventors": _build_csr(ib_s, ib_d, n_nodes),
        "inventor_to_techs": _build_csr(ib_d, ib_s, n_nodes),
        "inventor_to_orgs": _build_csr(aw_s, aw_d, n_nodes),
        "org_to_inventors": _build_csr(aw_d, aw_s, n_nodes),
        "tech_to_experts": _build_csr(inv_s, inv_d, n_nodes),
        "expert_to_techs": _build_csr(inv_d, inv_s, n_nodes),
    }

    # ---- Org canonicalization (auditable: the key list ships in the artifact) -
    org_rows = np.flatnonzero(ntype == org_code).astype(np.int32)
    key_to_rows: dict[str, list[int]] = {}
    for r in org_rows:
        key_to_rows.setdefault(canonical_org_key(labels[r]), []).append(int(r))
    canon_keys = sorted(key_to_rows)                       # deterministic order
    canon_indptr = np.zeros(len(canon_keys) + 1, dtype=np.int64)
    canon_rows_list: list[int] = []
    canon_of_row = np.full(n_nodes, -1, dtype=np.int32)    # org row -> canon key id
    for ki, key in enumerate(canon_keys):
        rows = sorted(key_to_rows[key])
        canon_rows_list.extend(rows)
        canon_indptr[ki + 1] = len(canon_rows_list)
        for r in rows:
            canon_of_row[r] = ki

    # ---- Write ---------------------------------------------------------------
    node_ids_blob, node_ids_off = _pack_strings(node_ids)
    labels_blob, labels_off = _pack_strings(labels)
    doc_ids_blob, doc_ids_off = _pack_strings(tech_doc_ids)
    canon_blob, canon_off = _pack_strings(canon_keys)

    arrays: dict[str, np.ndarray] = {
        "format_version": np.array([FORMAT_VERSION], dtype=np.int32),
        "n_nodes": np.array([n_nodes], dtype=np.int64),
        "ntype": ntype,
        "ntype_names": np.array(list(NTYPES)),
        "node_ids_blob": node_ids_blob, "node_ids_off": node_ids_off,
        "labels_blob": labels_blob, "labels_off": labels_off,
        "tech_rows": tech_rows,
        "doc_ids_blob": doc_ids_blob, "doc_ids_off": doc_ids_off,
        "canon_keys_blob": canon_blob, "canon_keys_off": canon_off,
        "canon_indptr": canon_indptr,
        "canon_rows": np.array(canon_rows_list, dtype=np.int32),
        "canon_of_row": canon_of_row,
    }
    for name, (indptr, indices) in csr.items():
        arrays[f"{name}_indptr"] = indptr
        arrays[f"{name}_indices"] = indices

    out.parent.mkdir(parents=True, exist_ok=True)
    # Uncompressed on purpose: this artifact is on the interactive load path and
    # decompressing ~300 MB to save disk is the wrong trade for an agent.
    np.savez(out, **arrays)

    summary = {
        "out": str(out),
        "size_mb": round(out.stat().st_size / 1e6, 1),
        "n_nodes": n_nodes,
        "n_edges_read": n_edges,
        "node_order_mismatches_vs_nodes_jsonl": order_mismatches,
        "nodes_by_type": {t: int((ntype == c).sum()) for t, c in NTYPE_TO_CODE.items()},
        "edges_kept": {name: int(len(indices)) for name, (_, indices) in csr.items()},
        "dropped_assigned_to_non_org": dropped_assigned_to_nonorg,
        "dropped_affiliated_with_not_inventor_to_org": dropped_affil_noninv_org,
        "n_org_nodes": int(len(org_rows)),
        "n_canonical_org_keys": len(canon_keys),
    }
    return summary


# --------------------------------------------------------------------------- #
# Loader
# --------------------------------------------------------------------------- #
@dataclass
class GraphIndex:
    """
    Read-only accessor over the built artifact. Every neighbour lookup is an
    array slice; none of them allocate a Python object per edge.

    Rows are node ids in the SAME order as data_merged/artifacts/node_ids.json,
    so `rgcn_node_emb[row]` is this node's R-GCN embedding with no remapping.

    Entity linking (query text -> node id) is NOT here; it lives in
    discovery_hub/entity_link.py and stays there.
    """
    n_nodes: int
    ntype_codes: np.ndarray
    _node_ids: list[str]
    _labels: list[str]
    _row_of: dict[str, int]
    _csr: dict[str, tuple[np.ndarray, np.ndarray]]
    _tech_doc_id: dict[int, str]
    _row_of_doc_id: dict[str, int]
    _canon_keys: list[str]
    _canon_indptr: np.ndarray
    _canon_rows: np.ndarray
    _canon_of_row: np.ndarray

    # -- construction ------------------------------------------------------- #
    @classmethod
    def load(cls, path: Path | str | None = None) -> GraphIndex:
        path = Path(path) if path is not None else (config.ARTIFACT_DIR / ARTIFACT_NAME)
        with np.load(path, allow_pickle=False) as z:
            version = int(z["format_version"][0])
            if version != FORMAT_VERSION:
                raise ValueError(f"{path} is format v{version}, this code reads "
                                 f"v{FORMAT_VERSION}; rebuild with --build")
            node_ids = _unpack_all(z["node_ids_blob"], z["node_ids_off"])
            labels = _unpack_all(z["labels_blob"], z["labels_off"])
            tech_rows = z["tech_rows"]
            doc_ids = _unpack_all(z["doc_ids_blob"], z["doc_ids_off"])
            csr = {n: (z[f"{n}_indptr"], z[f"{n}_indices"]) for n in _CSR_NAMES}
            tech_doc_id = {int(r): d for r, d in zip(tech_rows, doc_ids)}
            return cls(
                n_nodes=int(z["n_nodes"][0]),
                ntype_codes=z["ntype"],
                _node_ids=node_ids,
                _labels=labels,
                _row_of={nid: i for i, nid in enumerate(node_ids)},
                _csr=csr,
                _tech_doc_id=tech_doc_id,
                _row_of_doc_id={d: int(r) for r, d in zip(tech_rows, doc_ids)},
                _canon_keys=_unpack_all(z["canon_keys_blob"], z["canon_keys_off"]),
                _canon_indptr=z["canon_indptr"],
                _canon_rows=z["canon_rows"],
                _canon_of_row=z["canon_of_row"],
            )

    # -- rows / identity ---------------------------------------------------- #
    def row(self, node_id: str) -> int | None:
        """Row for any node_id, or None if absent. None is a normal answer here."""
        return self._row_of.get(node_id)

    def org_row(self, node_id: str) -> int | None:
        """Row for an org node_id. None if unknown OR if it is not an org node."""
        r = self._row_of.get(node_id)
        if r is None or self.ntype(r) != "organization":
            return None
        return r

    def label(self, row: int) -> str:
        return self._labels[row]

    def node_id(self, row: int) -> str:
        return self._node_ids[row]

    def ntype(self, row: int) -> str:
        return NTYPES[int(self.ntype_codes[row])]

    def doc_id_of_tech(self, row: int) -> str | None:
        return self._tech_doc_id.get(int(row))

    def tech_row_of_doc_id(self, doc_id: str) -> int | None:
        return self._row_of_doc_id.get(doc_id)

    # -- traversal ---------------------------------------------------------- #
    def _neighbors(self, name: str, row: int) -> np.ndarray:
        indptr, indices = self._csr[name]
        return indices[indptr[row]:indptr[row + 1]]

    def portfolio(self, org_row: int) -> np.ndarray:
        """Technology rows assigned to this org node. THE portfolio primitive."""
        return self._neighbors("org_to_techs", org_row)

    def owners_of(self, tech_row: int) -> np.ndarray:
        return self._neighbors("tech_to_orgs", tech_row)

    def inventors_of(self, tech_row: int) -> np.ndarray:
        return self._neighbors("tech_to_inventors", tech_row)

    def techs_of_inventor(self, inventor_row: int) -> np.ndarray:
        return self._neighbors("inventor_to_techs", inventor_row)

    def orgs_of_inventor(self, inventor_row: int) -> np.ndarray:
        """Orgs an inventor is affiliated with. inventor->inventor edges excluded."""
        return self._neighbors("inventor_to_orgs", inventor_row)

    def inventors_of_org(self, org_row: int) -> np.ndarray:
        return self._neighbors("org_to_inventors", org_row)

    def experts_of(self, tech_row: int) -> np.ndarray:
        return self._neighbors("tech_to_experts", tech_row)

    def techs_of_expert(self, expert_row: int) -> np.ndarray:
        return self._neighbors("expert_to_techs", expert_row)

    def orgs_of_expert(self, expert_row: int) -> np.ndarray:
        """
        Orgs reachable from an expert, via
            expert <--investigated_by-- technology --assigned_to--> organization.

        The spec calls for `expert --affiliated_with--> org`. That edge does not
        exist for a single one of the 213,687 expert nodes (full-scan verified),
        so this two-hop is the ONLY route. It is a weaker signal than a real
        affiliation -- it says "this expert investigated something that org owns",
        which is sponsorship or collaboration, not employment. Callers must not
        render it as "works for".

        MEASURED, AND A REAL LIMIT: this returns a non-empty result for only
        ~13.4% of experts (534/4000 uniform sample, seed 0). The reason is
        structural, not a bug -- experts attach overwhelmingly to OpenAlex works,
        and OpenAlex technologies almost never carry an assigned_to org (~1% of a
        4000-tech sample), because papers have no assignee. The expert route to
        orgs is therefore thin; `inventor_to_orgs` (patent assignees, dense) is
        the load-bearing path for expertise-gap, and an agent that leans on this
        method for coverage will abstain most of the time. That is the honest
        shape of the data, and it should be stated in the output, not hidden.
        """
        techs = self.techs_of_expert(expert_row)
        if len(techs) == 0:
            return np.zeros(0, dtype=np.int32)
        indptr, indices = self._csr["tech_to_orgs"]
        spans = [indices[indptr[t]:indptr[t + 1]] for t in techs]
        return np.unique(np.concatenate(spans)) if spans else np.zeros(0, dtype=np.int32)

    # -- org canonicalization ----------------------------------------------- #
    def canonical_key(self, org_row: int) -> str | None:
        ki = int(self._canon_of_row[org_row])
        return None if ki < 0 else self._canon_keys[ki]

    def canonical_org_rows(self, org_row: int) -> np.ndarray:
        """
        Every org row sharing this row's canonical key -- i.e. the fragments of
        one legal name ("bristol myers squibb" + "bristol myers squibb co"). The
        union of their portfolios is the org's real coverage. Subsidiaries are
        NOT included; they are different companies.
        """
        ki = int(self._canon_of_row[org_row])
        if ki < 0:
            return np.array([org_row], dtype=np.int32)
        return self._canon_rows[self._canon_indptr[ki]:self._canon_indptr[ki + 1]]

    def canonical_portfolio(self, org_row: int) -> np.ndarray:
        """Union of the portfolios of every fragment of this org's canonical name."""
        spans = [self.portfolio(int(r)) for r in self.canonical_org_rows(org_row)]
        spans = [s for s in spans if len(s)]
        return np.unique(np.concatenate(spans)) if spans else np.zeros(0, dtype=np.int32)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--build", action="store_true", help="build the index artifact")
    ap.add_argument("--out", type=Path, default=None,
                    help=f"output path (default: ARTIFACT_DIR/{ARTIFACT_NAME})")
    args = ap.parse_args()
    if not args.build:
        ap.print_help()
        return 2

    print(f"graph dir   : {config.GRAPH_DIR}")
    print(f"artifact dir: {config.ARTIFACT_DIR}")
    summary = build(config.GRAPH_DIR, config.ARTIFACT_DIR, args.out)
    print("\n--- graph_index summary ---")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
