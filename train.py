"""
Entry point: trains the Party fraud classifier on the Mule_Account_Detection
TigerGraph schema.

Run:
    python train.py                                 # synthetic data
    python train.py --source tigergraph             # live TigerGraph pull
    python train.py --conv transformer --no-reify   # edge features off the edge
    python train.py --devices 4 --strategy ddp      # multi-GPU

Two routes get transaction amounts and timestamps into the model, since
HGTConv reads no edge attributes: reify transfers into vertices (default), or
keep them as edges and use a conv that accepts edge features (--conv
transformer --no-reify).
"""

import argparse

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

import schema
from data import make_synthetic_frames, make_synthetic_graph
from features import build_edge_feat_dims, build_node_feat_dims, leakage_audit
from lightning_module import (
    FraudFullGraphDataModule,
    FraudHeteroDataModule,
    FraudHGTLightningModule,
)


def _run_leakage_audit(args):
    """Warn about attributes that separate the label almost perfectly."""
    if args.source == "tigergraph":
        from tg_loader import fetch_raw

        raw = fetch_raw(exclude_derived=args.exclude_derived, verbose=False)
    else:
        raw = make_synthetic_frames()
    vertex_frames, edge_indices, _ = raw

    findings = leakage_audit(
        vertex_frames, edge_indices, exclude_derived=args.exclude_derived
    )
    if not findings:
        print("leakage audit: no single attribute separates the label (AUC < 0.99)")
        return
    print("leakage audit: these attributes are near-perfect label predictors ON THEIR OWN")
    for node_type, name, auc in findings:
        print(f"  {node_type}.{name}: rank AUC {auc:.3f}")
    print("  a score built on these is measuring the leak, not the model.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["synthetic", "tigergraph"], default="synthetic")
    parser.add_argument("--target", default=schema.TARGET_TYPE)
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument(
        "--conv",
        choices=["hgt", "transformer"],
        default="hgt",
        help=(
            "'hgt': HGTConv, type-aware attention. Reads no edge attributes, so "
            "transfers must be reified (the default) for amount/transfer_time to "
            "reach it. 'transformer': HeteroConv + TransformerConv(edge_dim=...), "
            "which consumes Transfer's edge features directly -- pair it with "
            "--no-reify."
        ),
    )
    parser.add_argument(
        "--no-reify",
        dest="reify",
        action="store_false",
        help=(
            "Keep Transfer as an Account->Account edge instead of turning each "
            "transfer into a Transfer_Transaction vertex. With --conv hgt this "
            "discards amount and transfer_time entirely."
        ),
    )
    parser.add_argument(
        "--no-audit",
        dest="audit",
        action="store_false",
        help="Skip the label-leakage audit that runs before training.",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument(
        "--strategy", default="auto", help="e.g. 'ddp' for multi-GPU/multi-node training"
    )
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument(
        "--split",
        choices=["stratified", "random", "temporal"],
        default="stratified",
        help=(
            "'stratified' gives each split a proportional share of the positives "
            "-- important here, where only 49 of 5000 Parties are fraudulent and "
            "an unstratified draw can leave validation with almost none. "
            "'temporal' needs a populated Party.created_at, which this dataset "
            "does not have (every value is epoch-zero)."
        ),
    )
    parser.add_argument(
        "--exclude-derived",
        action="store_true",
        help=(
            "Drop the precomputed Account graph features (fraud_device, fraud_ip, "
            "mule_cnt, trans_*_mule_ratio, shortest_path_length, com_id, com_size). "
            "Kept by default; if they were computed from known fraud labels, the "
            "gap between the two runs is the size of the leak."
        ),
    )
    parser.add_argument(
        "--sampling",
        choices=["neighbor", "full_graph"],
        default="neighbor",
        help=(
            "'neighbor': PyG NeighborLoader mini-batching with the per-edge-type "
            "fan-out in schema.build_fanout. Needs 'pyg-lib' or 'torch-sparse'. "
            "'full_graph': no sampler. This graph fits either way, but full_graph "
            "gives only one gradient step per epoch."
        ),
    )
    args = parser.parse_args()

    pl.seed_everything(args.seed, workers=True)

    # ---- Data --------------------------------------------------------
    if args.source == "tigergraph":
        from tg_loader import load_from_tigergraph

        data = load_from_tigergraph(
            exclude_derived=args.exclude_derived,
            reify_transfer=args.reify,
            split_mode=args.split,
            seed=args.seed,
        )
    else:
        data = make_synthetic_graph(
            exclude_derived=args.exclude_derived,
            reify_transfer=args.reify,
            split_mode=args.split,
            seed=args.seed,
            verbose=True,
        )

    if args.audit:
        _run_leakage_audit(args)

    if args.sampling == "neighbor":
        dm = FraudHeteroDataModule(
            data,
            target_node_type=args.target,
            num_hops=args.num_layers,  # sample as deep as the model propagates
            batch_size=args.batch_size,
            num_workers=0,  # bump this (and use --strategy ddp) at scale
        )
    else:
        dm = FraudFullGraphDataModule(data, target_node_type=args.target)

    # ---- Class imbalance ----------------------------------------------
    y_train = data[args.target].y[data[args.target].train_mask]
    n_pos = (y_train == 1).sum().item()
    n_neg = (y_train == 0).sum().item()
    # inverse-frequency weighting, clipped so the positive class can't dominate
    pos_weight = min(n_neg / max(n_pos, 1), 20.0)
    class_weights = torch.tensor([1.0, pos_weight])
    print(f"train fraud prevalence: {n_pos}/{n_pos + n_neg} -> pos_weight={pos_weight:.2f}")

    node_feat_dims = build_node_feat_dims(data)
    edge_feat_dims = build_edge_feat_dims(data)
    print(f"node feature dims: {node_feat_dims}")
    if edge_feat_dims:
        used = "read by TransformerConv" if args.conv == "transformer" else "IGNORED by HGTConv"
        print(f"edge feature dims: {edge_feat_dims}  ({used})")

    # ---- Model ----------------------------------------------------------
    module = FraudHGTLightningModule(
        node_feat_dims=node_feat_dims,
        metadata=data.metadata(),
        hidden_channels=args.hidden_channels,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        lr=args.lr,
        target_node_type=args.target,
        class_weights=class_weights,
        conv=args.conv,
        edge_feat_dims=edge_feat_dims,
    )

    callbacks = [
        EarlyStopping(monitor="val_auroc", mode="max", patience=8),
        ModelCheckpoint(monitor="val_auroc", mode="max", filename="best-{epoch}-{val_auroc:.3f}"),
    ]

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy=args.strategy,
        callbacks=callbacks,
        log_every_n_steps=5,
        enable_progress_bar=True,
    )

    trainer.fit(module, datamodule=dm)
    trainer.test(module, datamodule=dm, ckpt_path="best")


if __name__ == "__main__":
    main()
