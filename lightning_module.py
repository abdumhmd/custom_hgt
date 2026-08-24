"""
LightningModule + LightningDataModules for the Party fraud classifier.

`Mule_Account_Detection` is small enough (~40k vertices, ~130k edges, or
~130k vertices once transfers are reified) that full-graph training fits
comfortably. `NeighborLoader` remains the default anyway: it is the path that
survives a larger graph, and with only 49 positives the mini-batch loop gives
far more gradient steps per epoch than the single step full-graph training
provides.

The two datamodules are interchangeable from the LightningModule's side --
it only assumes a batch exposes `x_dict`, `edge_index_dict`, optional
`edge_attr`, and a way to identify seed nodes and labels.
"""

from typing import Dict, Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from torch_geometric.loader import NeighborLoader

import schema
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


class FraudHGTLightningModule(pl.LightningModule):
    def __init__(
        self,
        node_feat_dims: Dict[str, int],
        metadata,
        hidden_channels: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        target_node_type: str = schema.TARGET_TYPE,
        class_weights: torch.Tensor = None,
        conv: str = "hgt",
        edge_feat_dims: Optional[Dict] = None,
    ):
        super().__init__()
        # avoid saving huge metadata objects verbatim in the checkpoint hparams
        self.save_hyperparameters(ignore=["metadata", "class_weights", "edge_feat_dims"])
        self.metadata = metadata
        self.target_node_type = target_node_type

        self.model = build_model(
            conv=conv,
            node_feat_dims=node_feat_dims,
            edge_feat_dims=edge_feat_dims,
            metadata=metadata,
            hidden_channels=hidden_channels,
            out_channels=2,
            num_heads=num_heads,
            num_layers=num_layers,
            target_node_type=target_node_type,
        )

        self.register_buffer(
            "class_weights",
            class_weights if class_weights is not None else torch.tensor([1.0, 1.0]),
        )

        # metrics: AUROC / AP matter far more than accuracy for fraud
        # (heavy class imbalance -> accuracy is close to meaningless)
        self.train_auroc = torchmetrics.AUROC(task="binary")
        self.val_auroc = torchmetrics.AUROC(task="binary")
        self.val_ap = torchmetrics.AveragePrecision(task="binary")
        self.test_auroc = torchmetrics.AUROC(task="binary")
        self.test_ap = torchmetrics.AveragePrecision(task="binary")

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        return self.model(x_dict, edge_index_dict, edge_attr_dict)

    def _shared_step(self, batch, stage: str):
        # Only the un-reified Transfer relation carries edge attributes, and
        # only HeteroEdgeGNN reads them; HGT ignores the argument.
        #
        # Note: `hasattr(batch, "edge_attr_dict")` is not usable to tell the two
        # batch types apart. HeteroData.__getattr__ intercepts any `*_dict`
        # access and raises KeyError when no edge has that attribute, and
        # hasattr only swallows AttributeError. Branch on the batch type first.
        if isinstance(batch, FullGraphBatch):
            edge_attr_dict = batch.edge_attr_dict
            out = self(batch.x_dict, batch.edge_index_dict, edge_attr_dict)
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
            # `batch_size` to compute loss only on the seed Parties, not the
            # neighbours pulled in for message passing.
            size = batch[self.target_node_type].batch_size
            seed_out = out[:size]
            seed_y = batch[self.target_node_type].y[:size]

        loss = F.cross_entropy(seed_out, seed_y, weight=self.class_weights)
        probs = F.softmax(seed_out, dim=-1)[:, 1]

        self.log(f"{stage}_loss", loss, prog_bar=True, batch_size=seed_y.size(0))
        return loss, probs, seed_y

    def training_step(self, batch, batch_idx):
        loss, probs, y = self._shared_step(batch, "train")
        self.train_auroc.update(probs, y)
        self.log("train_auroc", self.train_auroc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, probs, y = self._shared_step(batch, "val")
        self.val_auroc.update(probs, y)
        self.val_ap.update(probs, y)
        self.log("val_auroc", self.val_auroc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_ap", self.val_ap, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, probs, y = self._shared_step(batch, "test")
        self.test_auroc.update(probs, y)
        self.test_ap.update(probs, y)
        self.log("test_auroc", self.test_auroc, on_step=False, on_epoch=True)
        self.log("test_ap", self.test_ap, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", patience=5)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "monitor": "val_auroc"},
        }


class FraudHeteroDataModule(pl.LightningDataModule):
    """Wraps a single in-memory HeteroData object with NeighborLoader.

    This is the class to swap out if the graph outgrows one machine: `setup()`
    would open a partitioned graph store and the `*_dataloader()` methods
    would return distributed neighbor loaders. The LightningModule does not
    need to change.
    """

    def __init__(
        self,
        data,
        target_node_type: str = schema.TARGET_TYPE,
        num_neighbors=None,
        num_hops: int = 3,
        batch_size: int = 64,
        num_workers: int = 0,
    ):
        super().__init__()
        self.data = data
        self.target_node_type = target_node_type
        # Per-relation fan-out, keyed off the relations the graph actually has
        # rather than the schema's declared ones -- reification swaps Transfer
        # for Send/Receive, and NeighborLoader raises if any relation present in
        # the data is missing an entry. This graph's hubs are mild (~5 Parties
        # per shared IP or Device), so they are capped modestly rather than
        # hard: at that width the sharing IS the fraud signal.
        self.num_neighbors = (
            num_neighbors
            if num_neighbors is not None
            else schema.build_fanout(num_hops, edge_types=data.edge_types)
        )
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


class FraudFullGraphDataModule(pl.LightningDataModule):
    """Full-batch (no neighbor sampling) datamodule.

    Fits this graph easily, and needs no pyg-lib. The tradeoff is one gradient
    step per epoch, which trains much more slowly than the sampled path.
    """

    def __init__(self, data, target_node_type: str = schema.TARGET_TYPE):
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
