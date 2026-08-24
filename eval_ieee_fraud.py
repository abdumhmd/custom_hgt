"""
HGT vs HeteroEdgeGNN on the IEEE-CIS fraud graph (`build_ieee_fraud_graph.py`).

Unlike the OGBN-MAG comparison, this graph has real numeric edge attributes
(amount, time) on every relation, so this actually exercises the thing
`HeteroEdgeGNN` exists for in `model.py`: `TransformerConv(edge_dim=2)` reads
them directly, while `HGTConv` structurally cannot and only sees them via
`transaction`'s own node features (the same amount/time info, reached the
same way `data.reify_transfers` reaches it in the main fraud pipeline).

Full-batch (graph is ~608K nodes / ~3.8M edges after ToUndirected -- small
enough that a NeighborLoader isn't needed, unlike OGBN-MAG).
"""

import argparse

import torch
import torch.nn.functional as F
import torch_geometric
import torch_geometric.transforms as T
import torchmetrics

from model import build_model

# See eval_ogbn_mag.py: forces PyG's plain per-type matmul loop instead of
# pyg_lib's segment_matmul, which has no MPS kernel.
torch_geometric.backend.use_segment_matmul = False


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_data(cache_path):
    data = torch.load(cache_path, weights_only=False)
    return T.ToUndirected()(data)


@torch.no_grad()
def evaluate(model, data, mask):
    model.eval()
    out = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
    # torchmetrics' ROC/AP kernels are unreliable on MPS (observed: garbage
    # indices, out-of-bounds crashes) -- compute metrics on CPU.
    probs = out.softmax(dim=-1)[:, 1][mask].cpu()
    y = data["transaction"].y[mask].cpu()
    auroc = torchmetrics.functional.auroc(probs, y, task="binary").item()
    ap = torchmetrics.functional.average_precision(probs, y, task="binary").item()
    return auroc, ap


def train_one(conv, data, device, args):
    node_feat_dims = {nt: data[nt].x.size(-1) for nt in data.node_types}
    edge_feat_dims = {et: data[et].edge_attr.size(-1) for et in data.edge_types if "edge_attr" in data[et]}

    model = build_model(
        conv=conv,
        node_feat_dims=node_feat_dims,
        metadata=data.metadata(),
        edge_feat_dims=edge_feat_dims,
        hidden_channels=args.hidden_channels,
        out_channels=2,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        target_node_type="transaction",
        dropout=0.2,
    ).to(device)

    y = data["transaction"].y
    train_mask = data["transaction"].train_mask
    class_counts = torch.bincount(y[train_mask], minlength=2).float()
    class_weight = (class_counts.sum() / (2 * class_counts)).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
        loss = F.cross_entropy(out[train_mask], y[train_mask], weight=class_weight)
        loss.backward()
        optimizer.step()

        val_auroc, val_ap = evaluate(model, data, data["transaction"].val_mask)
        print(f"  [{conv}] epoch {epoch}/{args.epochs}  loss={loss.item():.4f}  "
              f"val_auroc={val_auroc:.4f}  val_ap={val_ap:.4f}")

    test_auroc, test_ap = evaluate(model, data, data["transaction"].test_mask)
    return {"conv": conv, "test_auroc": test_auroc, "test_ap": test_ap}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default=".cache/ieee_fraud_graph.pt")
    p.add_argument("--conv", choices=["hgt", "transformer", "both"], default="both")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--hidden-channels", type=int, default=64)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()
    print(f"device: {device}")

    data = load_data(args.cache).to(device)
    print(data)
    base_rate = data["transaction"].y[data["transaction"].test_mask].float().mean().item()
    print(f"test fraud base rate: {base_rate:.4f}")

    convs = ["hgt", "transformer"] if args.conv == "both" else [args.conv]
    results = []
    for conv in convs:
        print(f"\n=== {conv} ===")
        results.append(train_one(conv, data, device, args))

    print("\n=== summary ===")
    for r in results:
        print(f"  {r['conv']:>11}: test_auroc={r['test_auroc']:.4f}  test_ap={r['test_ap']:.4f}")


if __name__ == "__main__":
    main()
