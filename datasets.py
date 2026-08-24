"""
Dataset registry.

Turns a `--dataset` name into a ready-to-train `HeteroData` plus the task
metadata (target node type, class count, task kind, default sampling
strategy) that `train.py` / `hparam_search.py` / `eval.py` need and would
otherwise have to special-case per dataset.

Adding a fifth dataset means adding one loader function here and one
REGISTRY entry -- nothing in `lightning_module.py`, `train.py`,
`hparam_search.py`, or `eval.py` needs to change, since all of those only
ever see a `Dataset`.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict

import torch
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData

import schema
from data import make_synthetic_graph
from features import build_edge_feat_dims, build_node_feat_dims


@dataclass
class Dataset:
    data: HeteroData
    target_node_type: str
    num_classes: int
    task: str  # "binary" (AUROC/AP) or "multiclass" (accuracy)
    sampling: str  # default loader strategy: "neighbor" or "full_graph"
    node_feat_dims: Dict[str, int]
    edge_feat_dims: Dict


def _finalize(data: HeteroData, target_node_type: str, num_classes: int, task: str, sampling: str) -> Dataset:
    return Dataset(
        data=data,
        target_node_type=target_node_type,
        num_classes=num_classes,
        task=task,
        sampling=sampling,
        node_feat_dims=build_node_feat_dims(data),
        edge_feat_dims=build_edge_feat_dims(data),
    )


def _load_synthetic(
    seed: int = 0, exclude_derived: bool = False, reify_transfer: bool = True,
    split_mode: str = "stratified", **_,
) -> Dataset:
    data = make_synthetic_graph(
        exclude_derived=exclude_derived, reify_transfer=reify_transfer,
        split_mode=split_mode, seed=seed, verbose=False,
    )
    return _finalize(data, schema.TARGET_TYPE, num_classes=2, task="binary", sampling="full_graph")


def _load_tigergraph(
    seed: int = 0, exclude_derived: bool = False, reify_transfer: bool = True,
    split_mode: str = "stratified", **_,
) -> Dataset:
    from tg_loader import load_from_tigergraph

    data = load_from_tigergraph(
        exclude_derived=exclude_derived, reify_transfer=reify_transfer,
        split_mode=split_mode, seed=seed,
    )
    return _finalize(data, schema.TARGET_TYPE, num_classes=2, task="binary", sampling="neighbor")


def _load_ieee_fraud(cache: str = ".cache/ieee_fraud_graph.pt", root: str = "data/ieee_fraud", **_) -> Dataset:
    path = Path(cache)
    if not path.exists():
        from build_ieee_fraud_graph import build

        build(root, str(path))
    data = torch.load(path, weights_only=False)
    data = T.ToUndirected()(data)
    return _finalize(data, "transaction", num_classes=2, task="binary", sampling="full_graph")


def _load_ogbn_mag(root: str = "data/ogbn_mag", **_) -> Dataset:
    from torch_geometric.datasets import OGB_MAG

    dataset = OGB_MAG(root=root, preprocess="metapath2vec", transform=T.ToUndirected())
    data = dataset[0]
    num_classes = int(data["paper"].y.max()) + 1
    return _finalize(data, "paper", num_classes=num_classes, task="multiclass", sampling="neighbor")


REGISTRY: Dict[str, Callable[..., Dataset]] = {
    "synthetic": _load_synthetic,
    "tigergraph": _load_tigergraph,
    "ieee_fraud": _load_ieee_fraud,
    "ogbn_mag": _load_ogbn_mag,
}


def load_dataset(name: str, **kwargs) -> Dataset:
    if name not in REGISTRY:
        raise ValueError(f"unknown dataset {name!r}; choices: {sorted(REGISTRY)}")
    return REGISTRY[name](**kwargs)


def class_weights(ds: Dataset):
    """Inverse-frequency class weights on the train split, clipped so the
    minority class can't dominate. Only meaningful for imbalanced binary
    tasks (fraud); multiclass datasets here (OGBN-MAG) are roughly balanced,
    so this returns None and cross_entropy falls back to uniform weights.
    """
    if ds.task != "binary":
        return None
    y_train = ds.data[ds.target_node_type].y[ds.data[ds.target_node_type].train_mask]
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    pos_weight = min(n_neg / max(n_pos, 1), 20.0)
    return torch.tensor([1.0, pos_weight])
