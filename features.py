"""
Attribute -> tensor encoding for the TigerGraph fraud schema.

Takes raw per-vertex attribute columns (however they were sourced -- live
TigerGraph pull or the synthetic generator) and produces the `x` tensors that
HGTConv consumes, driven entirely by the `Kind` annotations in `schema.py`.

Two things here matter beyond mechanical dtype conversion:

  * Normalisation statistics for the target type are fit on the *training*
    split only. Fitting z-scores over all Party rows would let the val/test
    feature distribution influence training -- a small leak, but a free one
    to avoid.

  * `structural_features` gives the 8 attribute-less vertex types a feature
    vector. It is not filler: per-relation degree is the actual fraud signal
    on those nodes. A Phone attached to 40 Parties, or an Address attached to
    30, is exactly what a shared-identity fraud ring looks like.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

import schema
from schema import Kind

# Columns with fewer than this many distinct values get one-hot encoded; above
# it we fall back to a frequency encoding rather than exploding the width.
# Set above 50 so Address_v2.state (47 distinct) is one-hot rather than
# collapsed to a frequency scalar; zipcode (796 distinct) still falls back.
MAX_ONEHOT_CARDINALITY = 64

SECONDS_PER_YEAR = 365.25 * 24 * 3600


def _zscore(values: np.ndarray, fit_mask: Optional[np.ndarray]) -> np.ndarray:
    ref = values[fit_mask] if fit_mask is not None and fit_mask.any() else values
    mean = np.nanmean(ref) if ref.size else 0.0
    std = np.nanstd(ref) if ref.size else 1.0
    if not np.isfinite(std) or std < 1e-8:
        std = 1.0
    if not np.isfinite(mean):
        mean = 0.0
    out = (values - mean) / std
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _to_epoch_seconds(series: pd.Series) -> np.ndarray:
    dt = pd.to_datetime(series, errors="coerce", utc=True)
    return dt.astype("int64").to_numpy(dtype=np.float64) / 1e9


def _encode_datetime(series: pd.Series, fit_mask: Optional[np.ndarray]) -> np.ndarray:
    """Normalized epoch plus cyclical hour-of-day / day-of-week.

    The cyclical terms let the model express "3am transfers" or "weekend
    activity" without an artificial discontinuity at the 23->0 wrap.
    """
    epoch = _to_epoch_seconds(series)
    missing = ~np.isfinite(epoch)
    epoch_filled = np.where(missing, np.nanmedian(epoch[~missing]) if (~missing).any() else 0.0, epoch)

    hour = (epoch_filled / 3600.0) % 24.0
    weekday = (epoch_filled / 86400.0) % 7.0
    return np.column_stack([
        _zscore(epoch_filled, fit_mask),
        np.sin(2 * np.pi * hour / 24.0),
        np.cos(2 * np.pi * hour / 24.0),
        np.sin(2 * np.pi * weekday / 7.0),
        np.cos(2 * np.pi * weekday / 7.0),
        missing.astype(np.float64),
    ])


def _encode_age(series: pd.Series, fit_mask: Optional[np.ndarray]) -> np.ndarray:
    epoch = _to_epoch_seconds(series)
    ref = pd.Timestamp(schema.REFERENCE_DATE, tz="UTC").timestamp()
    age = (ref - epoch) / SECONDS_PER_YEAR
    missing = ~np.isfinite(age)
    age = np.where(missing, np.nan, age)
    # implausible ages are data errors, not signal
    age = np.where((age < 0) | (age > 120), np.nan, age)
    median = np.nanmedian(age) if np.isfinite(age).any() else 0.0
    filled = np.nan_to_num(age, nan=median)
    return np.column_stack([_zscore(filled, fit_mask), (~np.isfinite(age)).astype(np.float64)])


def _encode_categorical(series: pd.Series) -> np.ndarray:
    values = series.astype("string").fillna("__missing__")
    categories = sorted(values.unique().tolist())
    if len(categories) <= MAX_ONEHOT_CARDINALITY:
        index = {c: i for i, c in enumerate(categories)}
        out = np.zeros((len(values), len(categories)), dtype=np.float64)
        out[np.arange(len(values)), values.map(index).to_numpy()] = 1.0
        return out
    # high cardinality: encode how common the value is instead of one-hotting it
    freq = values.map(values.value_counts(normalize=True)).to_numpy(dtype=np.float64)
    return np.log1p(freq * len(values)).reshape(-1, 1)


def _encode_numeric(series: pd.Series, log: bool, fit_mask: Optional[np.ndarray]) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    missing = ~np.isfinite(values)
    median = np.nanmedian(values[~missing]) if (~missing).any() else 0.0
    values = np.where(missing, median, values)
    if log:
        values = np.log1p(np.clip(values, 0.0, None))
    return np.column_stack([_zscore(values, fit_mask), missing.astype(np.float64)])


def encode_vertex_frame(
    node_type: str,
    frame: pd.DataFrame,
    exclude_derived: bool = False,
    fit_mask: Optional[np.ndarray] = None,
    drop_constant: bool = True,
) -> torch.Tensor:
    """Encode one vertex type's attribute frame into a float32 feature matrix.

    `fit_mask` restricts normalisation statistics to a subset of rows (the
    training split, for the target type). Attribute-less types return an
    empty (n, 0) tensor -- `structural_features` fills them in later.
    """
    blocks: List[np.ndarray] = []
    for name, kind in schema.active_attrs(node_type, exclude_derived=exclude_derived):
        if name not in frame.columns:
            raise KeyError(
                f"{node_type}.{name} is declared in schema.VERTEX_ATTRS but missing "
                f"from the loaded data (columns: {sorted(frame.columns)})"
            )
        column = frame[name]
        if kind is Kind.BOOL:
            block = pd.to_numeric(column, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64).reshape(-1, 1)
        elif kind is Kind.NUMERIC:
            block = _encode_numeric(column, log=False, fit_mask=fit_mask)
        elif kind is Kind.LOG_NUMERIC:
            block = _encode_numeric(column, log=True, fit_mask=fit_mask)
        elif kind is Kind.CATEGORICAL:
            block = _encode_categorical(column)
        elif kind is Kind.DATETIME:
            block = _encode_datetime(column, fit_mask=fit_mask)
        elif kind is Kind.AGE:
            block = _encode_age(column, fit_mask=fit_mask)
        else:
            continue
        blocks.append(block)

    n = len(frame)
    if not blocks:
        return torch.zeros((n, 0), dtype=torch.float32)

    matrix = np.column_stack(blocks)
    if drop_constant:
        # Many attributes are declared in the TigerGraph schema but never
        # populated -- Party.gender, Account.account_type, IP.is_blocked and
        # others are constant across every row of the current dataset. A
        # constant column carries no information and only widens the input
        # projection, so it is pruned here rather than hardcoded to DROP in
        # schema.py: the moment the data is populated it starts counting again,
        # with no schema edit.
        keep = matrix.std(axis=0) > 1e-8
        if not keep.any():
            return torch.zeros((n, 0), dtype=torch.float32)
        matrix = matrix[:, keep]

    return torch.from_numpy(matrix).float()


def encode_edge_frame(edge_type, frame: pd.DataFrame) -> torch.Tensor:
    """Encode a relation's edge attributes into a float32 matrix.

    Only relations declared in schema.EDGE_ATTRS produce anything. These are
    consumed exclusively by a conv that accepts edge features
    (model.HeteroEdgeGNN) -- HGTConv ignores them, which is why reification
    exists as the alternative route for the same information.
    """
    declared = schema.EDGE_ATTRS.get(tuple(edge_type), [])
    blocks: List[np.ndarray] = []
    for name, kind in declared:
        if name not in frame.columns:
            continue
        column = frame[name]
        if kind is Kind.DATETIME:
            blocks.append(_encode_datetime(column, fit_mask=None))
        elif kind is Kind.LOG_NUMERIC:
            blocks.append(_encode_numeric(column, log=True, fit_mask=None))
        elif kind is Kind.NUMERIC:
            blocks.append(_encode_numeric(column, log=False, fit_mask=None))
        elif kind is Kind.CATEGORICAL:
            blocks.append(_encode_categorical(column))
        elif kind is Kind.BOOL:
            blocks.append(
                pd.to_numeric(column, errors="coerce").fillna(0.0)
                .to_numpy(dtype=np.float64).reshape(-1, 1)
            )

    if not blocks:
        return torch.zeros((len(frame), 0), dtype=torch.float32)
    matrix = np.column_stack(blocks)
    keep = matrix.std(axis=0) > 1e-8
    matrix = matrix[:, keep] if keep.any() else matrix[:, :0]
    return torch.from_numpy(matrix).float()


def _rank_auc(values: np.ndarray, labels: np.ndarray) -> float:
    """AUC of a single column against a binary label, via rank statistics.

    Returned as distance from chance folded to >= 0.5, so a perfectly
    *inverted* predictor scores as high as a perfectly aligned one -- a sign
    flip is not protection against leakage.
    """
    finite = np.isfinite(values)
    values, labels = values[finite], labels[finite]
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1)
    # average ranks within ties, or a constant column would look informative
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inverse, ranks)
    ranks = (sums / counts)[inverse]

    auc = (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return max(auc, 1.0 - auc)


def leakage_audit(
    vertex_frames: Dict[str, pd.DataFrame],
    edge_indices: Dict,
    threshold: float = 0.99,
    exclude_derived: bool = False,
) -> List[tuple]:
    """Find attributes that separate the label almost perfectly on their own.

    Audits the target type directly, plus any type joined to it 1:1 (here
    Account, via Party_Has_Account) -- a feature one hop from the target is
    just as readable by message passing as one on the target itself.

    A single column reaching AUC ~1.0 is not a good feature, it is the label
    wearing a different name. `Account.shortest_path_length` in this dataset is
    exactly that: 0 for every fraudulent account and 1-2 for every other one,
    because it measures distance to the nearest known fraud node.
    """
    target = schema.TARGET_TYPE
    target_frame = vertex_frames.get(target)
    if target_frame is None or schema.LABEL_ATTR not in target_frame:
        return []
    labels = pd.to_numeric(target_frame[schema.LABEL_ATTR], errors="coerce").fillna(0).to_numpy()

    audited = {target: labels}

    # propagate labels across strictly 1:1 relations with the target
    n_target = len(target_frame)
    for (src, _, dst), edge_index in edge_indices.items():
        other = dst if src == target else src if dst == target else None
        if other is None or other not in vertex_frames:
            continue
        n_other = len(vertex_frames[other])
        if n_other != n_target or edge_index.size(1) != n_target:
            continue
        target_side = edge_index[0] if src == target else edge_index[1]
        other_side = edge_index[1] if src == target else edge_index[0]
        if len(np.unique(other_side.numpy())) != n_other:
            continue
        mapped = np.zeros(n_other)
        mapped[other_side.numpy()] = labels[target_side.numpy()]
        audited[other] = mapped

    findings = []
    for node_type, node_labels in audited.items():
        frame = vertex_frames[node_type]
        for name, kind in schema.active_attrs(node_type, exclude_derived=exclude_derived):
            if name not in frame.columns or kind is Kind.CATEGORICAL:
                continue
            column = pd.to_numeric(frame[name], errors="coerce")
            if column.isna().all() or column.nunique(dropna=True) <= 1:
                continue
            auc = _rank_auc(column.to_numpy(dtype=np.float64), node_labels)
            if auc >= threshold:
                findings.append((node_type, name, auc))
    return sorted(findings, key=lambda f: -f[2])


def constant_attrs(
    node_type: str, frame: pd.DataFrame, exclude_derived: bool = False
) -> List[str]:
    """Declared attributes that are constant in this dataset, for reporting."""
    dead = []
    for name, kind in schema.active_attrs(node_type, exclude_derived=exclude_derived):
        if name not in frame.columns:
            continue
        if frame[name].nunique(dropna=False) <= 1:
            dead.append(name)
    return dead


def structural_features(data: HeteroData) -> None:
    """Append per-relation log-degree features to every node type, in place.

    Runs before ToUndirected so the degree counts come from the 17 declared
    relations rather than being doubled by their generated reverses. Each node
    type gets one column per incident relation (as source, and as destination),
    which for the attribute-less types is their whole representation.
    """
    columns: Dict[str, List[torch.Tensor]] = {t: [] for t in data.node_types}

    for edge_type in data.edge_types:
        src, _, dst = edge_type
        edge_index = data[edge_type].edge_index
        for node_type, row in ((src, 0), (dst, 1)):
            n = data[node_type].num_nodes
            deg = torch.zeros(n, dtype=torch.float32)
            deg.scatter_add_(
                0, edge_index[row], torch.ones(edge_index.size(1), dtype=torch.float32)
            )
            columns[node_type].append(torch.log1p(deg).unsqueeze(1))

    for node_type, cols in columns.items():
        existing = data[node_type].x if "x" in data[node_type] else None
        if existing is None:
            existing = torch.zeros((data[node_type].num_nodes, 0), dtype=torch.float32)
        parts = [existing] + cols
        data[node_type].x = torch.cat(parts, dim=1)

    # A node type with neither attributes nor edges would end up 0-wide and
    # crash the input projection; give it a constant so the graph still builds.
    for node_type in data.node_types:
        if data[node_type].x.size(1) == 0:
            data[node_type].x = torch.ones((data[node_type].num_nodes, 1), dtype=torch.float32)


def build_edge_feat_dims(data: HeteroData) -> Dict:
    """Edge-feature widths, for relations that carry any."""
    return {
        edge_type: int(data[edge_type].edge_attr.size(1))
        for edge_type in data.edge_types
        if "edge_attr" in data[edge_type]
    }


def build_node_feat_dims(data: HeteroData) -> Dict[str, int]:
    """Derive feature widths from the built tensors.

    Replaces the hand-maintained NODE_FEAT_DIMS constant, which could silently
    drift out of sync with the actual data.
    """
    return {t: int(data[t].x.size(1)) for t in data.node_types}


def _cut(index: torch.Tensor, fractions) -> Dict[str, torch.Tensor]:
    n_train = int(fractions[0] * index.numel())
    n_val = int(fractions[1] * index.numel())
    return {
        "train_mask": index[:n_train],
        "val_mask": index[n_train : n_train + n_val],
        "test_mask": index[n_train + n_val :],
    }


def make_splits(
    n: int,
    labelled: torch.Tensor,
    y: Optional[torch.Tensor] = None,
    mode: str = "stratified",
    order_key: Optional[Sequence] = None,
    fractions=(0.7, 0.15),
    generator: Optional[torch.Generator] = None,
) -> Dict[str, torch.Tensor]:
    """Train/val/test masks over the labelled target nodes only.

    `mode="stratified"` (default) splits each class separately so every split
    gets a proportional share of the positives. This matters a great deal on
    this graph: there are only 49 fraudulent Parties out of 5000, and an
    unstratified 70/15/15 draw can easily hand validation two positives, or
    none at all, making the metric meaningless.

    `mode="temporal"` orders by `order_key` and cuts chronologically, so
    validation measures forward-looking generalisation. It needs a populated
    timestamp -- `Party.created_at` is epoch-zero for every row in the current
    dataset, so this mode is unusable there and will raise.
    """
    eligible = labelled.nonzero(as_tuple=True)[0]

    if mode == "temporal":
        if order_key is None:
            raise ValueError("temporal split requires order_key (e.g. Party.created_at)")
        key = _to_epoch_seconds(pd.Series(list(order_key)))[eligible.numpy()]
        if np.nanstd(key) < 1e-8:
            raise ValueError(
                "temporal split requires a populated timestamp, but every value is "
                "identical. Party.created_at is epoch-zero throughout this dataset; "
                "use --split stratified instead."
            )
        ordered = eligible[torch.from_numpy(np.argsort(key, kind="stable"))]
        chunks = _cut(ordered, fractions)

    elif mode == "stratified":
        if y is None:
            raise ValueError("stratified split requires labels")
        chunks = {"train_mask": [], "val_mask": [], "test_mask": []}
        for label in torch.unique(y[eligible]):
            members = eligible[y[eligible] == label]
            members = members[torch.randperm(members.numel(), generator=generator)]
            for name, part in _cut(members, fractions).items():
                chunks[name].append(part)
        chunks = {name: torch.cat(parts) for name, parts in chunks.items()}

    else:  # plain random
        shuffled = eligible[torch.randperm(eligible.numel(), generator=generator)]
        chunks = _cut(shuffled, fractions)

    masks = {}
    for name, idx in chunks.items():
        mask = torch.zeros(n, dtype=torch.bool)
        mask[idx] = True
        masks[name] = mask
    return masks
