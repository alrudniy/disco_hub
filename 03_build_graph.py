#!/usr/bin/env python3
"""
03_build_graph.py  --  LAYER 1, step 3 of 6.

Turns the normalized DiscoveryDocs into a heterogeneous knowledge graph:

    nodes:  technology (the invention/doc), inventor, organization, expert, facility
    edges:  invented_by, assigned_to, investigated_by, located_at, affiliated_with

Writes data/graph/{nodes.jsonl, edges.jsonl, graph_meta.json}. These three files
are the contract consumed by 06_train_rgcn.py. Entity resolution here is simple
(normalized string match); at full scale, lean on PatentsView's pre-disambiguated
inventors/assignees and OpenAlex's disambiguated authors/institutions instead of
rolling your own.

  TARGET: Anvil CPU (RAM heavy at scale). networkx is used only to validate the
          graph; the R-GCN consumes the JSONL edge lists directly.

Usage:
  python 03_build_graph.py
"""
from __future__ import annotations

import argparse
import json
import re

import networkx as nx

from discovery_hub import config
from discovery_hub.schema import read_docs, NODE_TYPES, RELATIONS


def norm_entity(name: str) -> str:
    """Cheap entity-resolution key: lowercase, strip punctuation/extra space."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", name.lower())).strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--validate", action="store_true",
                    help="run networkx structural checks and print graph stats")
    args = ap.parse_args()

    config.ensure_dirs()
    docs_path = config.NORM_DIR / "docs.jsonl"

    # node_key -> (node_id, ntype, label)
    nodes: dict[str, dict] = {}
    edges: list[dict] = []

    def get_node(label: str, ntype: str, key_override: str | None = None) -> str:
        # Technologies are keyed by their doc_id (each invention/paper is unique);
        # people/orgs/facilities dedup on normalized string (that is where entity
        # resolution actually matters). Real data should use PatentsView/OpenAlex
        # pre-disambiguated ids here instead of string match.
        key = key_override or f"{ntype}:{norm_entity(label)}"
        if key not in nodes:
            nodes[key] = {"node_id": key, "ntype": ntype, "label": label}
        return key

    n_docs = 0
    for doc in read_docs(docs_path):
        n_docs += 1
        tech = get_node(doc.title or doc.doc_id, "technology",
                        key_override=f"technology:{doc.doc_id}")
        # technology carries its doc_id so retrieval can join back to the doc.
        nodes[tech]["doc_id"] = doc.doc_id
        for inv in doc.inventors:
            inv_n = get_node(inv, "inventor")
            edges.append({"src": tech, "dst": inv_n, "rel": "invented_by"})
            for org in doc.organizations:
                edges.append({"src": inv_n, "dst": get_node(org, "organization"),
                              "rel": "affiliated_with"})
        for org in doc.organizations:
            edges.append({"src": tech, "dst": get_node(org, "organization"),
                          "rel": "assigned_to"})
        for exp in doc.experts:
            edges.append({"src": tech, "dst": get_node(exp, "expert"),
                          "rel": "investigated_by"})
        for fac in doc.facilities:
            edges.append({"src": tech, "dst": get_node(fac, "facility"),
                          "rel": "located_at"})

    # De-duplicate edges (same triple can be emitted twice).
    edges = [dict(t) for t in {tuple(sorted(e.items())) for e in edges}]

    # Write artifacts.
    with (config.GRAPH_DIR / "nodes.jsonl").open("w") as fh:
        for nd in nodes.values():
            fh.write(json.dumps(nd) + "\n")
    with (config.GRAPH_DIR / "edges.jsonl").open("w") as fh:
        for e in edges:
            fh.write(json.dumps(e) + "\n")

    meta = {
        "node_types": list(NODE_TYPES),
        "relations": list(RELATIONS),
        "num_nodes": len(nodes),
        "num_edges": len(edges),
        "num_docs": n_docs,
        "nodes_by_type": {t: sum(1 for n in nodes.values() if n["ntype"] == t)
                          for t in NODE_TYPES},
        "edges_by_rel": {r: sum(1 for e in edges if e["rel"] == r) for r in RELATIONS},
    }
    with (config.GRAPH_DIR / "graph_meta.json").open("w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"Built graph: {meta['num_nodes']} nodes, {meta['num_edges']} edges "
          f"from {n_docs} docs -> {config.GRAPH_DIR}")
    print("  nodes by type:", meta["nodes_by_type"])
    print("  edges by rel :", meta["edges_by_rel"])

    if args.validate:
        G = nx.MultiDiGraph()
        for nd in nodes.values():
            G.add_node(nd["node_id"], **nd)
        for e in edges:
            G.add_edge(e["src"], e["dst"], rel=e["rel"])
        comps = nx.number_weakly_connected_components(G)
        iso = list(nx.isolates(G))
        print(f"  [validate] weakly-connected components: {comps}; isolates: {len(iso)}")
        if G.number_of_nodes():
            degs = dict(G.degree())
            top = sorted(degs.items(), key=lambda kv: kv[1], reverse=True)[:3]
            print("  [validate] highest-degree nodes:",
                  [(nodes[k]["label"], d) for k, d in top])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
