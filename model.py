"""
Two heterogeneous models for Party fraud classification.

`HGT` -- PyG's HGTConv (Hu et al. 2020). Type-aware attention with
per-relation parameters, which is the right prior for a schema with 10+ node
types. Its one limitation here is structural: `HGTConv.forward` takes only
`(x_dict, edge_index_dict)`, so it can read no edge attributes at all. On this
schema that would silently discard ~100k transaction amounts and timestamps,
which is why `data.reify_transfers` exists -- it turns Transfer edges into
vertices so those attributes arrive as node features instead.

`HeteroEdgeGNN` -- the alternative answer to the same problem. `HeteroConv`
routes a per-relation conv, and `TransformerConv` accepts `edge_dim`, so
transformer-style attention consumes `Transfer.amount` / `transfer_time`
directly off the edge with no reification. It gives up HGT's specific
type-aware parameterisation in exchange.

Both classify only `Party` embeddings.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, HGTConv, Linear, TransformerConv


class _Base(nn.Module):
    """Shared input projection and classification head."""

    def __init__(self, node_feat_dims, metadata, hidden_channels, out_channels,
                 target_node_type, dropout):
        super().__init__()
        self.target_node_type = target_node_type
        self.dropout = dropout

        # Every node type in the metadata must have a feature width, or message
        # passing raises deep inside the conv. Fail here with a readable message.
        missing = set(metadata[0]) - set(node_feat_dims)
        if missing:
            raise ValueError(
                f"no feature dim for node type(s) {sorted(missing)}. Every type in "
                "the graph metadata needs an `x` -- attribute-less types get one "
                "from features.structural_features()."
            )

        self.lin_dict = nn.ModuleDict()
        for ntype, dim in node_feat_dims.items():
            self.lin_dict[ntype] = Linear(dim, hidden_channels)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
        )

    def _project(self, x_dict):
        return {ntype: self.lin_dict[ntype](x).relu() for ntype, x in x_dict.items()}


class HGT(_Base):
    def __init__(
        self,
        node_feat_dims: Dict[str, int],
        metadata,
        hidden_channels: int = 64,
        out_channels: int = 2,
        num_heads: int = 4,
        num_layers: int = 4,
        target_node_type: str = "Party",
        dropout: float = 0.2,
        **_,
    ):
        super().__init__(node_feat_dims, metadata, hidden_channels, out_channels,
                         target_node_type, dropout)
        self.convs = nn.ModuleList([
            HGTConv(hidden_channels, hidden_channels, metadata, heads=num_heads)
            for _ in range(num_layers)
        ])

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        x_dict = self._project(x_dict)
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {
                ntype: F.dropout(x.relu(), p=self.dropout, training=self.training)
                for ntype, x in x_dict.items()
                if x is not None
            }
        return self.classifier(x_dict[self.target_node_type])


class HeteroEdgeGNN(_Base):
    """Per-relation TransformerConv that consumes edge features.

    `edge_feat_dims` maps a relation to its edge-feature width; relations
    absent from it get `edge_dim=None` and run without edge attributes.
    HeteroConv treats `edge_index_dict` as an edge-level argument, so every
    relation is still processed whether or not it has edge features.
    """

    def __init__(
        self,
        node_feat_dims: Dict[str, int],
        metadata,
        edge_feat_dims: Optional[Dict] = None,
        hidden_channels: int = 64,
        out_channels: int = 2,
        num_heads: int = 4,
        num_layers: int = 3,
        target_node_type: str = "Party",
        dropout: float = 0.2,
        **_,
    ):
        super().__init__(node_feat_dims, metadata, hidden_channels, out_channels,
                         target_node_type, dropout)
        edge_feat_dims = edge_feat_dims or {}
        if hidden_channels % num_heads:
            raise ValueError("hidden_channels must be divisible by num_heads")
        head_dim = hidden_channels // num_heads

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(HeteroConv(
                {
                    edge_type: TransformerConv(
                        hidden_channels,
                        head_dim,
                        heads=num_heads,
                        dropout=dropout,
                        edge_dim=edge_feat_dims.get(edge_type),
                    )
                    for edge_type in metadata[1]
                },
                aggr="sum",
            ))

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        x_dict = self._project(x_dict)
        for conv in self.convs:
            out = conv(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict or {})
            # HeteroConv only returns types that received messages; carry the
            # rest forward unchanged so they survive to the next layer.
            x_dict = {
                ntype: F.dropout(out[ntype].relu(), p=self.dropout, training=self.training)
                if ntype in out else x
                for ntype, x in x_dict.items()
            }
        return self.classifier(x_dict[self.target_node_type])


def build_model(conv: str = "hgt", **kwargs) -> nn.Module:
    if conv == "hgt":
        return HGT(**kwargs)
    if conv == "transformer":
        return HeteroEdgeGNN(**kwargs)
    raise ValueError(f"unknown conv: {conv}")
