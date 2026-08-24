"""
Multi-seed evaluation.

With 49 positives in the graph, a 70/15/15 split puts ~7 in test. A single
AUROC computed on 7 positives is dominated by which 7 they happen to be, so a
single number is not evidence of anything. This runs the same configuration
across seeds and reports the spread, which is.

Pulls from TigerGraph once and re-splits per seed rather than re-querying.

    python sweep.py --source tigergraph --seeds 10
    python sweep.py --source tigergraph --conv transformer --no-reify
"""

import argparse
import statistics
import warnings

import pytorch_lightning as pl
import torch

import schema
from data import build_hetero_data, make_synthetic_frames
from features import build_edge_feat_dims, build_node_feat_dims
from lightning_module import (
    FraudFullGraphDataModule,
    FraudHeteroDataModule,
    FraudHGTLightningModule,
)


def run_one(raw, args, seed: int):
    vertex_frames, edge_indices, edge_frames = raw
    pl.seed_everything(seed, workers=True)

    data = build_hetero_data(
        vertex_frames=vertex_frames,
        edge_indices=edge_indices,
        edge_frames=edge_frames,
        exclude_derived=args.exclude_derived,
        reify_transfer=args.reify,
        split_mode=args.split,
        seed=seed,
    )

    if args.sampling == "neighbor":
        dm = FraudHeteroDataModule(
            data, num_hops=args.num_layers, batch_size=args.batch_size
        )
    else:
        dm = FraudFullGraphDataModule(data)

    target = schema.TARGET_TYPE
    y_train = data[target].y[data[target].train_mask]
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    class_weights = torch.tensor([1.0, min(n_neg / max(n_pos, 1), 20.0)])

    module = FraudHGTLightningModule(
        node_feat_dims=build_node_feat_dims(data),
        edge_feat_dims=build_edge_feat_dims(data),
        metadata=data.metadata(),
        hidden_channels=args.hidden_channels,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        lr=args.lr,
        class_weights=class_weights,
        conv=args.conv,
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        logger=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(module, datamodule=dm)
    result = trainer.test(module, datamodule=dm, verbose=False)[0]
    n_test_pos = int(data[target].y[data[target].test_mask].sum())
    return result["test_auroc"], result["test_ap"], n_test_pos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["synthetic", "tigergraph"], default="tigergraph")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--conv", choices=["hgt", "transformer"], default="hgt")
    parser.add_argument("--no-reify", dest="reify", action="store_false")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_epochs", type=int, default=15)
    parser.add_argument("--accelerator", default="cpu")
    parser.add_argument("--split", choices=["stratified", "random", "temporal"], default="stratified")
    parser.add_argument("--exclude-derived", action="store_true")
    parser.add_argument("--sampling", choices=["neighbor", "full_graph"], default="neighbor")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    import logging

    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

    if args.source == "tigergraph":
        from tg_loader import fetch_raw

        raw = fetch_raw(exclude_derived=args.exclude_derived)
    else:
        raw = make_synthetic_frames()

    label = f"{args.conv}{'' if args.reify else '+no-reify'}"
    label += " --exclude-derived" if args.exclude_derived else ""
    print(f"\n=== {label} | {args.seeds} seeds | split={args.split} ===")

    aurocs, aps = [], []
    for seed in range(args.seeds):
        auroc, ap, n_test_pos = run_one(raw, args, seed)
        aurocs.append(auroc)
        aps.append(ap)
        print(f"  seed {seed:>2}  auroc {auroc:.3f}   ap {ap:.3f}   ({n_test_pos} test positives)")

    def summarize(name, values):
        spread = statistics.stdev(values) if len(values) > 1 else 0.0
        print(
            f"  {name:<6} mean {statistics.mean(values):.3f}  sd {spread:.3f}  "
            f"min {min(values):.3f}  max {max(values):.3f}"
        )

    print()
    summarize("AUROC", aurocs)
    summarize("AP", aps)


if __name__ == "__main__":
    main()
