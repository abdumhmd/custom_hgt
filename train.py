"""
Train one of the two architectures in `model.py` on one of the datasets in
`datasets.py`, then evaluate the best checkpoint on the test split.

    python train.py --dataset synthetic
    python train.py --dataset tigergraph --conv transformer --no-reify
    python train.py --dataset ieee_fraud --conv hgt
    python train.py --dataset ogbn_mag --conv transformer --max-epochs 3

Two routes get edge attributes (e.g. TigerGraph's Transfer.amount, IEEE
fraud's transaction amount/time) into the model, since HGTConv reads no edge
attributes at all: reify them into vertices (TigerGraph/synthetic only,
default there), or keep them as edges and use a conv that accepts edge
features (--conv transformer).
"""

import argparse

import pytorch_lightning as pl
import torch
import torch_geometric
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from datasets import REGISTRY, class_weights, load_dataset
from eval import evaluate, make_datamodule
from lightning_module import GNNLightningModule

# HGTConv's per-relation Linear defaults to pyg_lib's fused segment_matmul
# kernel, which has no MPS implementation and silently falls back to a single
# CPU core for that op. Force PyG's plain per-type matmul loop so MPS training
# actually runs on MPS. No effect on CUDA/CPU.
torch_geometric.backend.use_segment_matmul = False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(REGISTRY))
    parser.add_argument("--conv", choices=["hgt", "transformer"], default="hgt")
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--sampling", choices=["neighbor", "full_graph"], default=None,
                         help="defaults to the dataset's usual strategy (see datasets.py)")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-neighbors", type=int, default=10)
    parser.add_argument("--num-hops", type=int, default=None,
                         help="NeighborLoader depth; defaults to --num-layers")
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--accelerator", default="cpu",
                         help="'cpu' is the safe default -- torchmetrics' binary "
                              "ROC/AP kernels are unreliable on MPS")
    parser.add_argument("--devices", default="auto")
    # dataset-specific passthroughs (ignored by datasets that don't use them)
    parser.add_argument("--exclude-derived", action="store_true")
    parser.add_argument("--no-reify", dest="reify_transfer", action="store_false")
    parser.add_argument("--split", dest="split_mode", choices=["stratified", "random", "temporal"],
                         default="stratified")
    return parser


def train(args) -> dict:
    pl.seed_everything(args.seed, workers=True)

    ds = load_dataset(
        args.dataset, seed=args.seed, exclude_derived=args.exclude_derived,
        reify_transfer=args.reify_transfer, split_mode=args.split_mode,
    )
    print(f"[{args.dataset}] node feature dims: {ds.node_feat_dims}")
    if ds.edge_feat_dims:
        used = "read by TransformerConv" if args.conv == "transformer" else "IGNORED by HGTConv"
        print(f"[{args.dataset}] edge feature dims: {ds.edge_feat_dims}  ({used})")

    weights = class_weights(ds)
    if weights is not None:
        print(f"[{args.dataset}] class weights: {weights.tolist()}")

    module = GNNLightningModule(
        node_feat_dims=ds.node_feat_dims,
        metadata=ds.data.metadata(),
        target_node_type=ds.target_node_type,
        num_classes=ds.num_classes,
        task=ds.task,
        hidden_channels=args.hidden_channels,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        class_weights=weights,
        conv=args.conv,
        edge_feat_dims=ds.edge_feat_dims,
    )

    dm = make_datamodule(
        ds, sampling=args.sampling, batch_size=args.batch_size,
        num_neighbors=args.num_neighbors, num_hops=args.num_hops or args.num_layers,
    )

    callbacks = [
        EarlyStopping(monitor=module.monitor, mode="max", patience=args.patience),
        ModelCheckpoint(
            monitor=module.monitor, mode="max",
            filename=f"{args.dataset}-{args.conv}-{{epoch}}-{{{module.monitor}:.3f}}",
        ),
    ]

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        callbacks=callbacks,
        log_every_n_steps=5,
        enable_progress_bar=True,
    )
    trainer.fit(module, datamodule=dm)

    metrics = evaluate(module, dm, trainer=trainer, ckpt_path="best")
    print("\n=== test metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    best_path = trainer.checkpoint_callback.best_model_path
    print(f"\nbest checkpoint: {best_path}")
    return {"metrics": metrics, "ckpt_path": best_path}


def main():
    args = build_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
