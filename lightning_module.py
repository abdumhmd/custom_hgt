"""
LightningModule + LightningDataModules for the two architectures in
`model.py`, generalized across datasets (see `datasets.py`): binary fraud
classification (AUROC/AP) on TigerGraph/synthetic/IEEE-fraud, or multiclass
classification (accuracy) on OGBN-MAG. Nothing here is schema- or
dataset-specific -- `target_node_type`, `num_classes`, and `task` all come
from the caller.

Two datamodules, interchangeable from the LightningModule's side -- it only
assumes a batch exposes `x_dict`, `edge_index_dict`, optional
`edge_attr_dict`, and a way to identify seed nodes and labels:

- `FullGraphDataModule`: one gradient step per epoch, no sampler. Fine for
  graphs that fit in memory whole (synthetic, IEEE fraud).
- `NeighborDataModule`: PyG `NeighborLoader` mini-batching, uniform fan-out
  per hop. Needed once the graph doesn't fit whole (OGBN-MAG), or to get more
  gradient steps per epoch out of a small positive class (TigerGraph).
"""

from typing import Dict, Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from torch_geometric.loader import NeighborLoader

from model import build_model


class FullGraphBatch:
    """Wraps the whole graph plus a seed-node index for one split.

    Unlike NeighborLoader batches, node order is left untouched (no
    seed-first permutation needed), so edge_index_dict stays valid as-is.
    The LightningModule detects this type (via `seed_idx`) and indexes
    predictions/labels directly instead of using the `[:batch_size]` slice
    convention used for real NeighborLoader batches.
    """

    def __init__(self, data, target_node_type, mask_name):
        self.x_dict = data.x_dict
        self.edge_index_dict = data.edge_index_dict
        self.edge_attr_dict = {
            et: data[et].edge_attr for et in data.edge_types if "edge_attr" in data[et]
        }
        mask = data[target_node_type][mask_name]
        self.seed_idx = mask.nonzero(as_tuple=True)[0]
        self.seed_y = data[target_node_type].y[self.seed_idx]

    def to(self, device):
        """Lightning moves batches to the accelerator via this hook."""
        self.x_dict = {k: v.to(device) for k, v in self.x_dict.items()}
        self.edge_index_dict = {k: v.to(device) for k, v in self.edge_index_dict.items()}
        self.edge_attr_dict = {k: v.to(device) for k, v in self.edge_attr_dict.items()}
        self.seed_idx = self.seed_idx.to(device)
        self.seed_y = self.seed_y.to(device)
        return self


class GNNLightningModule(pl.LightningModule):
    def __init__(
        self,
        node_feat_dims: Dict[str, int],
        metadata,
        target_node_type: str,
        num_classes: int,
        task: str = "binary",  # "binary" -> AUROC/AP, "multiclass" -> accuracy
        hidden_channels: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        class_weights: Optional[torch.Tensor] = None,
        conv: str = "hgt",
        edge_feat_dims: Optional[Dict] = None,
    ):
        super().__init__()
        if task not in ("binary", "multiclass"):
            raise ValueError(f"task must be 'binary' or 'multiclass', got {task!r}")
        # avoid saving huge metadata objects verbatim in the checkpoint hparams
        self.save_hyperparameters(ignore=["metadata", "class_weights", "edge_feat_dims"])
        self.metadata = metadata
        self.target_node_type = target_node_type
        self.task = task

        self.model = build_model(
            conv=conv,
            node_feat_dims=node_feat_dims,
            edge_feat_dims=edge_feat_dims,
            metadata=metadata,
            hidden_channels=hidden_channels,
            out_channels=num_classes,
            num_heads=num_heads,
            num_layers=num_layers,
            target_node_type=target_node_type,
            dropout=dropout,
        )

        self.register_buffer(
            "class_weights",
            class_weights if class_weights is not None else torch.ones(num_classes),
        )

        if task == "binary":
            # AUROC/AP matter far more than accuracy under class imbalance.
            self.train_metric = torchmetrics.AUROC(task="binary")
            self.val_metric = torchmetrics.AUROC(task="binary")
            self.val_ap = torchmetrics.AveragePrecision(task="binary")
            self.test_metric = torchmetrics.AUROC(task="binary")
            self.test_ap = torchmetrics.AveragePrecision(task="binary")
            self.monitor = "val_auroc"
        else:
            self.train_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
            self.val_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
            self.test_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
            self.monitor = "val_acc"

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        return self.model(x_dict, edge_index_dict, edge_attr_dict)

    def _shared_step(self, batch, stage: str):
        # Note: `hasattr(batch, "edge_attr_dict")` is not usable to tell the two
        # batch types apart. HeteroData.__getattr__ intercepts any `*_dict`
        # access and raises KeyError when no edge has that attribute, and
        # hasattr only swallows AttributeError. Branch on the batch type first.
        if isinstance(batch, FullGraphBatch):
            out = self(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
            # node order is untouched, so index the seed nodes directly
            seed_out = out[batch.seed_idx]
            seed_y = batch.seed_y
        else:
            edge_attr_dict = {
                et: batch[et].edge_attr
                for et in batch.edge_types
                if "edge_attr" in batch[et]
            }
            out = self(batch.x_dict, batch.edge_index_dict, edge_attr_dict)
            # NeighborLoader mini-batch: seed nodes come first, so slice to
            # `batch_size` to compute loss only on the seed nodes, not the
            # neighbours pulled in for message passing.
            size = batch[self.target_node_type].batch_size
            seed_out = out[:size]
            seed_y = batch[self.target_node_type].y[:size]

        loss = F.cross_entropy(seed_out, seed_y, weight=self.class_weights)
        self.log(f"{stage}_loss", loss, prog_bar=True, batch_size=seed_y.size(0))
        return loss, seed_out, seed_y

    def training_step(self, batch, batch_idx):
        loss, out, y = self._shared_step(batch, "train")
        self._update_metric(self.train_metric, out, y)
        self.log(f"train_{self._metric_name}", self.train_metric, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, out, y = self._shared_step(batch, "val")
        self._update_metric(self.val_metric, out, y)
        self.log(f"val_{self._metric_name}", self.val_metric, prog_bar=True, on_step=False, on_epoch=True)
        if self.task == "binary":
            self.val_ap.update(F.softmax(out, dim=-1)[:, 1], y)
            self.log("val_ap", self.val_ap, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, out, y = self._shared_step(batch, "test")
        self._update_metric(self.test_metric, out, y)
        self.log(f"test_{self._metric_name}", self.test_metric, on_step=False, on_epoch=True)
        if self.task == "binary":
            self.test_ap.update(F.softmax(out, dim=-1)[:, 1], y)
            self.log("test_ap", self.test_ap, on_step=False, on_epoch=True)
        return loss

    @property
    def _metric_name(self) -> str:
        return "auroc" if self.task == "binary" else "acc"

    def _update_metric(self, metric, out, y):
        if self.task == "binary":
            metric.update(F.softmax(out, dim=-1)[:, 1], y)
        else:
            metric.update(out.argmax(dim=-1), y)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", patience=5)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "monitor": self.monitor},
        }


class NeighborDataModule(pl.LightningDataModule):
    """PyG `NeighborLoader` mini-batching, uniform fan-out per hop.

    This is the class to swap out if the graph outgrows one machine: `setup()`
    would open a partitioned graph store and the `*_dataloader()` methods
    would return distributed neighbor loaders. The LightningModule does not
    need to change.
    """

    def __init__(
        self,
        data,
        target_node_type: str,
        num_neighbors: int = 10,
        num_hops: int = 2,
        batch_size: int = 512,
        num_workers: int = 0,
    ):
        super().__init__()
        self.data = data
        self.target_node_type = target_node_type
        self.num_neighbors = {et: [num_neighbors] * num_hops for et in data.edge_types}
        self.batch_size = batch_size
        self.num_workers = num_workers

    def _loader(self, mask_name: str, shuffle: bool):
        mask = self.data[self.target_node_type][mask_name]
        input_nodes = (self.target_node_type, mask.nonzero(as_tuple=True)[0])
        return NeighborLoader(
            self.data,
            num_neighbors=self.num_neighbors,
            input_nodes=input_nodes,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )

    def train_dataloader(self):
        return self._loader("train_mask", shuffle=True)

    def val_dataloader(self):
        return self._loader("val_mask", shuffle=False)

    def test_dataloader(self):
        return self._loader("test_mask", shuffle=False)


class FullGraphDataModule(pl.LightningDataModule):
    """Full-batch (no neighbor sampling) datamodule.

    The tradeoff for graphs that fit this way is one gradient step per
    epoch, which trains much more slowly than the sampled path.
    """

    def __init__(self, data, target_node_type: str):
        super().__init__()
        self.data = data
        self.target_node_type = target_node_type

    class _SingleBatchLoader:
        """A minimal iterable that yields exactly one FullGraphBatch per
        epoch. A plain Python list isn't safe here: Lightning interprets a
        list returned from *_dataloader() as *multiple dataloaders*, not
        multiple batches from one dataloader."""

        def __init__(self, batch):
            self.batch = batch

        def __iter__(self):
            yield self.batch

        def __len__(self):
            return 1

    def train_dataloader(self):
        batch = FullGraphBatch(self.data, self.target_node_type, "train_mask")
        return self._SingleBatchLoader(batch)

    def val_dataloader(self):
        batch = FullGraphBatch(self.data, self.target_node_type, "val_mask")
        return self._SingleBatchLoader(batch)

    def test_dataloader(self):
        batch = FullGraphBatch(self.data, self.target_node_type, "test_mask")
        return self._SingleBatchLoader(batch)
