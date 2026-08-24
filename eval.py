"""
Standalone evaluation: load a trained checkpoint and report test metrics on
one of the datasets in `datasets.py`.

    python eval.py --dataset ieee_fraud --ckpt lightning_logs/version_0/checkpoints/best.ckpt

Also exposes `evaluate()` and `make_datamodule()`, which `train.py` and
`hparam_search.py` import directly to run eval at the end of a run in-process
instead of shelling out to this file -- the "separate files, but eval can run
at the end of training or a sweep" split the whole trio is built around.
"""

import argparse

import pytorch_lightning as pl

from datasets import REGISTRY, load_dataset
from lightning_module import FullGraphDataModule, GNNLightningModule, NeighborDataModule


def make_datamodule(ds, sampling=None, batch_size=512, num_neighbors=10, num_hops=2):
    sampling = sampling or ds.sampling
    if sampling == "full_graph":
        return FullGraphDataModule(ds.data, target_node_type=ds.target_node_type)
    return NeighborDataModule(
        ds.data, target_node_type=ds.target_node_type,
        num_neighbors=num_neighbors, num_hops=num_hops, batch_size=batch_size,
    )


def evaluate(module, datamodule, trainer=None, ckpt_path=None, accelerator="cpu") -> dict:
    """Runs trainer.test and returns the flat metrics dict.

    Pass `ckpt_path="best"` (from a just-finished trainer.fit call) to
    restore the best checkpoint first; pass an explicit path to evaluate a
    checkpoint from a previous run; pass None to evaluate the module's
    current in-memory weights as-is.
    """
    trainer = trainer or pl.Trainer(accelerator=accelerator, logger=False, enable_progress_bar=False)
    (metrics,) = trainer.test(module, datamodule=datamodule, ckpt_path=ckpt_path, verbose=False)
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=sorted(REGISTRY))
    p.add_argument("--ckpt", required=True, help="path to a .ckpt saved by train.py or hparam_search.py")
    p.add_argument("--sampling", choices=["neighbor", "full_graph"], default=None,
                    help="defaults to the dataset's usual sampling strategy")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-neighbors", type=int, default=10)
    p.add_argument("--num-hops", type=int, default=2)
    p.add_argument("--accelerator", default="cpu")
    args = p.parse_args()

    ds = load_dataset(args.dataset)
    module = GNNLightningModule.load_from_checkpoint(
        args.ckpt,
        metadata=ds.data.metadata(),
        edge_feat_dims=ds.edge_feat_dims,
    )
    dm = make_datamodule(
        ds, sampling=args.sampling, batch_size=args.batch_size,
        num_neighbors=args.num_neighbors, num_hops=args.num_hops,
    )
    trainer = pl.Trainer(accelerator=args.accelerator, logger=False, enable_progress_bar=True)
    metrics = evaluate(module, dm, trainer=trainer)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
