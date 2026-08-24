"""
Quick directional comparison of the two architectures in `model.py` on
OGBN-MAG (Paper venue/subject classification, ~1.9M nodes / 21M edges).

This is a standalone eval -- it does not reuse `lightning_module.py`
(AUROC/AP + class-weighted binary fraud loss) because OGBN-MAG is a balanced
349-way multiclass problem, not the imbalanced binary fraud task the rest of
this repo targets. It reuses only `model.build_model`, so the two
architectures under test are exactly the ones in `model.py`.

Caveat: OGBN-MAG's edges (cites / writes / affiliated_with / has_topic) carry
no numeric attributes, so HeteroEdgeGNN's edge_dim=None on every relation --
this measures the two attention/aggregation schemes, not the edge-feature
route that is `HeteroEdgeGNN`'s reason for existing on the fraud schema.

This is a quick sanity comparison (fixed step budget, not a full OGB
leaderboard run): a handful of epochs over a step-limited NeighborLoader,
evaluated on a val subsample. Treat the numbers as directional.
"""

import argparse
import time

import torch
import torch.nn.functional as F
import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG
from torch_geometric.loader import NeighborLoader

from model import build_model

# HGTConv's per-relation Linear picks pyg_lib's fused `segment_matmul` kernel
# when available, which has no MPS implementation and would otherwise force a
# silent, much slower CPU fallback for that op. Forcing the plain per-type
# matmul loop keeps everything on MPS.
torch_geometric.backend.use_segment_matmul = False


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_data(root):
    dataset = OGB_MAG(root=root, preprocess="metapath2vec", transform=T.ToUndirected())
    return dataset[0]


def make_loader(data, mask_name, batch_size, num_neighbors, num_hops, shuffle):
    per_hop = [num_neighbors] * num_hops
    return NeighborLoader(
        data,
        num_neighbors={et: per_hop for et in data.edge_types},
        batch_size=batch_size,
        input_nodes=("paper", data["paper"][mask_name]),
        shuffle=shuffle,
    )


@torch.no_grad()
def evaluate(model, loader, device, max_batches):
    model.eval()
    correct = total = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = batch.to(device)
        out = model(batch.x_dict, batch.edge_index_dict)
        n = batch["paper"].batch_size
        pred = out[:n].argmax(dim=-1)
        y = batch["paper"].y[:n]
        correct += int((pred == y).sum())
        total += n
    return correct / max(total, 1)


def train_one(conv, data, device, args):
    node_feat_dims = {nt: data[nt].x.size(-1) for nt in data.node_types}
    out_channels = int(data["paper"].y.max()) + 1

    model = build_model(
        conv=conv,
        node_feat_dims=node_feat_dims,
        metadata=data.metadata(),
        hidden_channels=args.hidden_channels,
        out_channels=out_channels,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        target_node_type="paper",
        dropout=0.2,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)

    train_loader = make_loader(data, "train_mask", args.batch_size, args.num_neighbors, args.num_layers, shuffle=True)
    val_loader = make_loader(data, "val_mask", args.batch_size, args.num_neighbors, args.num_layers, shuffle=True)

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = n_batches = 0
        for i, batch in enumerate(train_loader):
            if i >= args.train_batches:
                break
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x_dict, batch.edge_index_dict)
            n = batch["paper"].batch_size
            loss = F.cross_entropy(out[:n], batch["paper"].y[:n])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        val_acc = evaluate(model, val_loader, device, args.eval_batches)
        print(f"  [{conv}] epoch {epoch}/{args.epochs}  "
              f"train_loss={total_loss / max(n_batches, 1):.4f}  val_acc={val_acc:.4f}")

    elapsed = time.time() - t0
    test_loader = make_loader(data, "test_mask", args.batch_size, args.num_neighbors, args.num_layers, shuffle=True)
    test_acc = evaluate(model, test_loader, device, args.eval_batches)
    return {"conv": conv, "test_acc": test_acc, "seconds": elapsed}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/ogbn_mag")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-neighbors", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=2, help="model depth == loader hops; kept shallow for a quick run")
    p.add_argument("--train-batches", type=int, default=400, help="cap batches/epoch for a quick run")
    p.add_argument("--eval-batches", type=int, default=50)
    p.add_argument("--hidden-channels", type=int, default=64)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()
    print(f"device: {device}")

    data = load_data(args.root)
    print(data)

    results = []
    for conv in ["hgt", "transformer"]:
        print(f"\n=== {conv} ===")
        results.append(train_one(conv, data, device, args))

    print("\n=== summary (quick sanity run, not an OGB leaderboard number) ===")
    for r in results:
        print(f"  {r['conv']:>11}: test_acc={r['test_acc']:.4f}  ({r['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
