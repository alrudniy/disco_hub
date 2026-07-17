#!/usr/bin/env python3
"""
06_train_rgcn.py  --  LAYER 2, step 6 of 6.

Learns relational node embeddings over the heterogeneous knowledge graph so that
retrieval can blend graph structure (who-works-with-whom, who-assigned-what) with
text similarity -- and, crucially, MEASURES whether those embeddings actually
capture the graph via held-out link prediction (ROC-AUC, Hits@k, MRR). Without
that measurement you cannot tell if the R-GCN earns its place in the pipeline.

  TARGET: Anvil single A100. The real path uses NEIGHBOR SAMPLING
          (LinkNeighborLoader), so memory is bounded by the sampled subgraph, not
          the whole graph -- a ~2M-node graph trains in single-digit GPU-hours and
          fits comfortably in 12-40 GB. Full-batch (the old behavior) needed
          >40 GB and did not scale; that is fixed here.

EVALUATION PROTOCOL (no leakage)
  Edges are split into message-passing / val / test. Embeddings are computed from
  the message-passing edges ONLY; val drives early stopping (real path); test is
  scored once at the end. Pass --val-frac 0 --test-frac 0 to disable the split and
  train the final deployed embeddings on ALL edges (skips evaluation).

Two paths behind one interface:
  --mock   deterministic numpy relational message passing (per-relation mean
           aggregation -- corrected to divide each relation's neighbor sum by that
           relation's degree before summing across relations). No torch. It does
           not train, so its link-prediction score reflects structure alone; the
           real path trains and should exceed it.
  (real)   PyTorch Geometric RGCNConv encoder + dot-product decoder, trained with
           BCE over sampled positive/negative edges via LinkNeighborLoader. Lazy
           import; gated so the package works without torch.

Outputs:
  data/artifacts/rgcn_node_emb.npy   float32 [num_nodes, dim], L2-normalized
  data/artifacts/node_ids.json       ordered node_id list aligned to rows
  data/artifacts/rgcn_eval.json      link-prediction metrics on held-out test edges

Usage:
  python 06_train_rgcn.py --mock --steps 3
  python 06_train_rgcn.py --epochs 20 --hidden 128 --fanout 15   # real PyG on GPU
  python 06_train_rgcn.py --mock --val-frac 0 --test-frac 0      # final embeddings
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from discovery_hub import config
from discovery_hub import graph_eval
from discovery_hub.determinism import set_global_determinism
from discovery_hub.schema import RELATIONS


def load_graph():
    nodes = [json.loads(l) for l in
             (config.GRAPH_DIR / "nodes.jsonl").read_text().splitlines() if l]
    edges = [json.loads(l) for l in
             (config.GRAPH_DIR / "edges.jsonl").read_text().splitlines() if l]
    node_ids = [n["node_id"] for n in nodes]
    idx = {nid: i for i, nid in enumerate(node_ids)}
    return nodes, edges, node_ids, idx


def relational_mean_aggregate(h: np.ndarray, by_rel: dict) -> np.ndarray:
    """
    One hop of relational MEAN aggregation, summed across relations:

        agg_v = sum_r  mean_{u in N_r(v)} h_u

    For EACH relation r, average the features of each node's r-neighbors (the
    relation's neighbor sum divided by that relation's in-degree), THEN sum those
    per-relation means across relations.

    The mean is taken per relation BEFORE the cross-relation sum. The original
    code divided the running cross-relation accumulator by each relation's degree
    in turn, which re-divided the contributions of relations already added: a node
    receiving [1,1] from a degree-1 relation and a [1,1] mean from a degree-2
    relation ended up [1.5,1.5] instead of the correct [2,2]. Pulled out as a pure
    function so this exact property is unit-tested (tests/test_rgcn_agg.py) and the
    bug cannot silently return.
    """
    n = h.shape[0]
    agg = np.zeros_like(h)
    for pairs in by_rel.values():
        if not pairs:
            continue
        src = np.fromiter((p[0] for p in pairs), dtype=np.int64)
        dst = np.fromiter((p[1] for p in pairs), dtype=np.int64)
        per_rel = np.zeros_like(h)
        np.add.at(per_rel, src, h[dst])          # sum r-neighbor features
        deg = np.zeros(n, np.float32)
        np.add.at(deg, src, 1.0)
        deg[deg == 0] = 1.0
        per_rel /= deg[:, None]                   # -> mean for THIS relation only
        agg += per_rel                            # then sum across relations
    return agg


def train_mock(mp_edges, n: int, idx: dict, dim: int, steps: int) -> np.ndarray:
    """
    Deterministic RELATIONAL MEAN AGGREGATION -- a graph-smoothing stand-in for a
    trained R-GCN (think relational label propagation / a simplified GCN):

      h_v <- normalize( h_v + sum_r mean_{u in N_r(v)} h_u )    [both directions]

    No learned or random per-relation transform: a rotation would scramble the
    embedding space so an untrained dot-product decoder sees no link signal (an
    honest fact about untrained R-GCNs). Plain mean aggregation instead pulls
    graph-adjacent nodes together, giving weak-but-real link-predictive signal
    that the metric in main() can detect. Seeded init + fixed ops => identical
    embeddings every run.

    Per relation we still aggregate separately (relations differ in WHICH
    neighbors contribute), then sum across relations. Each relation's neighbor sum
    is divided by THAT relation's degree before summing -- the earlier code
    divided the running accumulator by every relation's degree in turn, double-
    dividing earlier relations; that bug is fixed here. Propagation is over a few
    hops only: more steps over-smooth and HURT link prediction (visible in the
    eval), so the default is intentionally small.
    """
    rng = np.random.default_rng(config.SEED)
    h = rng.standard_normal((n, dim)).astype(np.float32)
    h /= (np.linalg.norm(h, axis=1, keepdims=True) + 1e-9)

    by_rel: dict[str, list[tuple[int, int]]] = {r: [] for r in RELATIONS}
    for e in mp_edges:
        if e["src"] in idx and e["dst"] in idx and e["rel"] in by_rel:
            a, b = idx[e["src"]], idx[e["dst"]]
            # Propagate along BOTH directions (inverse relations), as standard
            # R-GCN does. Without this, entity nodes are pure sinks that never
            # aggregate their technologies, so no homophily forms.
            by_rel[e["rel"]].append((a, b))
            by_rel[e["rel"]].append((b, a))

    for _ in range(steps):
        h = h + relational_mean_aggregate(h, by_rel)
        h /= (np.linalg.norm(h, axis=1, keepdims=True) + 1e-9)
    return h.astype(np.float32)


def train_real(mp_edges, val_edges, n, idx, dim, epochs, hidden, device,
               fanout, batch_size, patience=3):
    """
    PyTorch Geometric R-GCN with NEIGHBOR SAMPLING and link-prediction training.

    Pattern (validate on Anvil GPU -- not exercised in CPU/CI):
      * Data holds edge_index + edge_type for the message-passing edges.
      * LinkNeighborLoader samples a k-hop subgraph around a minibatch of
        supervision edges and adds negative edges (neg_sampling_ratio=1).
      * The encoder embeds the subgraph's nodes (looked up by global id batch.n_id),
        runs two RGCNConv layers using the subgraph's edge types (edge_type[batch.e_id]),
        and the decoder scores supervision edges as a dot product of endpoints.
      * Validation ROC-AUC (held-out val edges) drives early stopping.
    Returns the full-graph node embeddings from the best-val epoch.
    """
    import torch
    from torch_geometric.data import Data
    from torch_geometric.loader import LinkNeighborLoader
    from torch_geometric.nn import RGCNConv

    rel_to_id = {r: i for i, r in enumerate(RELATIONS)}

    def to_eit(edge_dicts):
        keep = [(idx[e["src"]], idx[e["dst"]], rel_to_id[e["rel"]])
                for e in edge_dicts
                if e["src"] in idx and e["dst"] in idx and e["rel"] in rel_to_id]
        s = torch.tensor([a for a, _, _ in keep], dtype=torch.long)
        d = torch.tensor([b for _, b, _ in keep], dtype=torch.long)
        t = torch.tensor([c for _, _, c in keep], dtype=torch.long)
        return torch.stack([s, d]), t

    edge_index, edge_type = to_eit(mp_edges)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    data = Data(edge_index=edge_index, num_nodes=n)
    data.edge_type = edge_type  # carried through sampling; subset via batch.e_id

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(n, hidden)
            self.conv1 = RGCNConv(hidden, hidden, len(RELATIONS), num_bases=8)
            self.conv2 = RGCNConv(hidden, dim, len(RELATIONS), num_bases=8)

        def forward(self, node_ids, ei, et):
            x = self.emb(node_ids)
            x = torch.relu(self.conv1(x, ei, et))
            return self.conv2(x, ei, et)

    model = Encoder().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    loader = LinkNeighborLoader(
        data, num_neighbors=[fanout, fanout],
        edge_label_index=edge_index,
        edge_label=torch.ones(edge_index.size(1)),
        neg_sampling_ratio=1.0, batch_size=batch_size, shuffle=True)

    full_idx = torch.arange(n, device=dev)
    ei_dev, et_dev = edge_index.to(dev), edge_type.to(dev)

    @torch.no_grad()
    def full_encode():
        model.eval()
        return model(full_idx, ei_dev, et_dev).detach().cpu().numpy()

    val_pos = graph_eval.edges_to_pairs(val_edges, idx)
    all_pos = set(graph_eval.edges_to_pairs(list(mp_edges) + list(val_edges), idx))

    best_auc, best_Z, bad = -1.0, None, 0
    for ep in range(epochs):
        model.train()
        total = 0.0
        for batch in loader:
            batch = batch.to(dev)
            opt.zero_grad()
            et = et_dev[batch.e_id] if hasattr(batch, "e_id") else batch.edge_type
            z = model(batch.n_id, batch.edge_index, et)
            u = z[batch.edge_label_index[0]]
            v = z[batch.edge_label_index[1]]
            logits = (u * v).sum(-1)
            loss = loss_fn(logits, batch.edge_label.float())
            loss.backward()
            opt.step()
            total += float(loss)
        Z = full_encode()
        Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
        if val_pos:
            m = graph_eval.evaluate_link_prediction(Zn, val_pos, all_pos,
                                                     seed=config.SEED)
            auc = m["roc_auc"]
        else:
            auc = -1.0
        print(f"  epoch {ep:3d} loss {total:.4f} val_auc {auc:.4f}")
        if auc > best_auc:
            best_auc, best_Z, bad = auc, Zn, 0
        else:
            bad += 1
            if bad >= patience and val_pos:
                print(f"  early stop at epoch {ep} (best val_auc {best_auc:.4f})")
                break
    if best_Z is None:
        best_Z = full_encode()
        best_Z /= (np.linalg.norm(best_Z, axis=1, keepdims=True) + 1e-9)
    return best_Z.astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--dim", type=int, default=config.EMBED_DIM)
    ap.add_argument("--steps", type=int, default=2,
                    help="mock propagation hops (small on purpose; more over-smooths)")
    ap.add_argument("--epochs", type=int, default=20, help="real training epochs")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--fanout", type=int, default=15,
                    help="neighbors sampled per hop (real path)")
    ap.add_argument("--batch-size", type=int, default=2048,
                    help="supervision edges per minibatch (real path)")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    set_global_determinism(config.SEED)
    config.ensure_dirs()
    nodes, edges, node_ids, idx = load_graph()
    n = len(node_ids)

    mp_edges, val_edges, test_edges = graph_eval.split_edges(
        edges, val_frac=args.val_frac, test_frac=args.test_frac, seed=config.SEED)
    print(f"R-GCN on {n} nodes / {len(edges)} edges "
          f"(mock={args.mock}, dim={args.dim}); "
          f"split: mp={len(mp_edges)} val={len(val_edges)} test={len(test_edges)}")

    if args.mock:
        emb = train_mock(mp_edges, n, idx, args.dim, args.steps)
    else:
        emb = train_real(mp_edges, val_edges, n, idx, args.dim, args.epochs,
                         args.hidden, args.device, args.fanout, args.batch_size)

    np.save(config.ARTIFACT_DIR / "rgcn_node_emb.npy", emb)
    with (config.ARTIFACT_DIR / "node_ids.json").open("w") as fh:
        json.dump(node_ids, fh)
    print(f"Wrote node embeddings {emb.shape} -> {config.ARTIFACT_DIR}")

    # Held-out link-prediction evaluation (the "did it learn anything" metric).
    if test_edges:
        test_pairs = graph_eval.edges_to_pairs(test_edges, idx)
        all_pos = set(graph_eval.edges_to_pairs(edges, idx))
        metrics = graph_eval.evaluate_link_prediction(emb, test_pairs, all_pos,
                                                       seed=config.SEED)
        metrics["mock"] = args.mock
        metrics["num_nodes"] = n
        metrics["num_edges"] = len(edges)
        (config.ARTIFACT_DIR / "rgcn_eval.json").write_text(json.dumps(metrics, indent=2))
        print(f"  link-prediction (held-out test, {metrics['num_eval']} edges): "
              f"AUC={metrics['roc_auc']:.3f}  "
              f"Hits@10={metrics['hits@10']:.3f}  MRR={metrics['mrr']:.3f}")
        if args.mock:
            print("  (mock R-GCN does not train; AUC reflects structure alone. "
                  "The real PyG path trains and should score higher.)")
    else:
        print("  eval split disabled (--val-frac/--test-frac = 0); "
              "embeddings use all edges, no link-prediction metric computed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
