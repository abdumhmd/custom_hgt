"""
HGT vs HeteroEdgeGNN on the IEEE-CIS fraud graph (`build_ieee_fraud_graph.py`).

Unlike the OGBN-MAG comparison, this graph has real numeric edge attributes
(amount, time) on every relation, so this actually exercises the thing
`HeteroEdgeGNN` exists for in `model.py`: `TransformerConv(edge_dim=2)` reads
them directly, while `HGTConv` structurally cannot and only sees them via
`transaction`'s own node features (the same amount/time info, reached the
same way `data.reify_transfers` reaches it in the main fraud pipeline).

Built on `lightning_module.py` rather than a hand-rolled loop: this task is
binary, class-weighted, AUROC/AP-scored, full-graph -- exactly what
`FraudHGTLightningModule` + `FraudFullGraphDataModule` already implement for
the TigerGraph schema, with no fraud-schema coupling beyond an overridable
`target_node_type` default. Mirrors `train.py`'s structure for the same
reason: consistency with the rest of the repo, checkpointing/logging for
free.

Full-batch (graph is ~608K nodes / ~3.8M edges after ToUndirected -- small
enough that a NeighborLoader isn't needed, unlike OGBN-MAG).
"""

import argparse

import pytorch_lightning as pl
import torch
import torch_geometric
import torch_geometric.transforms as T
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from features import build_edge_feat_dims, build_node_feat_dims
from lightning_module import FraudFullGraphDataModule, FraudHGTLightningModule

# See eval_ogbn_mag.py: forces PyG's plain per-type matmul loop instead of
# pyg_lib's segment_matmul, which has no MPS kernel.
torch_geometric.backend.use_segment_matmul = False


def load_data(cache_path):
    data = torch.load(cache_path, weights_only=False)
    return T.ToUndirected()(data)


def run_one(conv, data, args):
    node_feat_dims = build_node_feat_dims(data)
    edge_feat_dims = build_edge_feat_dims(data)

    y_train = data["transaction"].y[data["transaction"].train_mask]
    n_pos = (y_train == 1).sum().item()
    n_neg = (y_train == 0).sum().item()
    pos_weight = min(n_neg / max(n_pos, 1), 20.0)
    class_weights = torch.tensor([1.0, pos_weight])
    print(f"[{conv}] train fraud prevalence: {n_pos}/{n_pos + n_neg} -> pos_weight={pos_weight:.2f}")

    module = FraudHGTLightningModule(
        node_feat_dims=node_feat_dims,
        metadata=data.metadata(),
        hidden_channels=args.hidden_channels,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        lr=args.lr,
        target_node_type="transaction",
        class_weights=class_weights,
        conv=conv,
        edge_feat_dims=edge_feat_dims,
    )

    dm = FraudFullGraphDataModule(data, target_node_type="transaction")

    callbacks = [
        EarlyStopping(monitor="val_auroc", mode="max", patience=8),
        ModelCheckpoint(monitor="val_auroc", mode="max", filename=f"ieee-{conv}-{{epoch}}-{{val_auroc:.3f}}"),
    ]

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        # torchmetrics' binary ROC/AP kernels produce garbage/crash on MPS
        # (cumsum-based ops); CPU is the safe default for this script. Model
        # forward/backward is cheap enough here that this costs little.
        accelerator=args.accelerator,
        callbacks=callbacks,
        log_every_n_steps=1,
        enable_progress_bar=True,
    )

    trainer.fit(module, datamodule=dm)
    (test_metrics,) = trainer.test(module, datamodule=dm, ckpt_path="best")
    return {"conv": conv, "test_auroc": test_metrics["test_auroc"], "test_ap": test_metrics["test_ap"]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default=".cache/ieee_fraud_graph.pt")
    p.add_argument("--conv", choices=["hgt", "transformer", "both"], default="both")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--hidden-channels", type=int, default=64)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--accelerator", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    pl.seed_everything(args.seed, workers=True)

    data = load_data(args.cache)
    print(data)
    base_rate = data["transaction"].y[data["transaction"].test_mask].float().mean().item()
    print(f"test fraud base rate: {base_rate:.4f}")

    convs = ["hgt", "transformer"] if args.conv == "both" else [args.conv]
    results = [run_one(conv, data, args) for conv in convs]

    print("\n=== summary ===")
    for r in results:
        print(f"  {r['conv']:>11}: test_auroc={r['test_auroc']:.4f}  test_ap={r['test_ap']:.4f}")


if __name__ == "__main__":
    main()
