"""
Single source of truth for the `Mule_Account_Detection` TigerGraph schema.

Transcribed from the live instance (TigerGraph 4.2.4 Community), not from
documentation: 10 global vertex types and 10 edge relations. `preflight.py`
re-checks this against the running graph and reports any drift.

Two things differ from the older schema this repo originally targeted:

  * geography is **denormalized** onto `Address_v2` (zipcode / city / state /
    population as attributes) rather than being City / State / Zipcode
    vertices joined by Located_In and Assigned_To.

  * transactions are a single directed **`Transfer` edge** (Account->Account)
    carrying `amount` / `transfer_time`, not `Transfer_Transaction` vertices.
    Since HGTConv reads no edge attributes, `reified_*` below optionally turns
    those edges back into vertices so the amounts and timestamps can reach the
    model. See REIFY_TRANSFER.
"""

from enum import Enum
from typing import Dict, List, Set, Tuple

EdgeTriple = Tuple[str, str, str]


class Kind(str, Enum):
    """How an attribute is turned into model features."""

    NUMERIC = "numeric"          # z-scored
    LOG_NUMERIC = "log_numeric"  # log1p then z-scored (heavy-tailed)
    CATEGORICAL = "categorical"  # one-hot, or frequency-encoded if wide
    DATETIME = "datetime"        # normalized epoch + cyclical hour/weekday
    AGE = "age"                  # years elapsed at REFERENCE_DATE
    BOOL = "bool"                # cast to 0.0/1.0
    DROP = "drop"                # never used as a feature


TARGET_TYPE = "Party"
LABEL_ATTR = "is_fraud"

# Fixed anchor so `dob` -> age is reproducible. On real data with a defined
# prediction window this should become that window's start date.
REFERENCE_DATE = "2026-01-01"

# The reified transaction vertex, created at load time from Transfer edges.
TRANSFER_NODE = "Transfer_Transaction"


# --- Vertex types and their attributes -------------------------------------
# Primary keys differ per type (phone_number, email, dob, address_key, name,
# id), so loaders must never assume `.id`. They are identity, not signal, and
# are excluded here.
#
# Many attributes are declared in the schema but unpopulated in the current
# dataset -- Party.gender, Party.created_at, Account.account_type, IP.is_blocked
# and others are constant across all rows. They are declared with their true
# Kind rather than hardcoded to DROP, and the zero-variance pruning in
# features.encode_vertex_frame removes them automatically. That way they start
# contributing the moment the data is populated, with no schema edit.
VERTEX_ATTRS: Dict[str, List[Tuple[str, Kind]]] = {
    # -- attribute-less types; features come from structural degree only
    "Phone": [],
    "Email": [],
    "Full_Name": [],
    "DOB": [],
    # -- typed vertices
    "IP": [
        ("is_blocked", Kind.BOOL),
    ],
    "Device": [
        ("is_blocked", Kind.BOOL),
    ],
    "ID": [
        ("id_type", Kind.CATEGORICAL),
    ],
    "Address_v2": [
        ("address_line1", Kind.DROP),  # identity-like, one per address
        ("zipcode", Kind.CATEGORICAL),
        ("city", Kind.CATEGORICAL),
        ("state", Kind.CATEGORICAL),
        ("population", Kind.LOG_NUMERIC),
    ],
    "Account": [
        ("create_Time", Kind.DATETIME),
        ("is_fraud", Kind.DROP),  # see ALWAYS_DROP
        ("account_type", Kind.CATEGORICAL),
        ("account_level", Kind.CATEGORICAL),
        ("com_size", Kind.LOG_NUMERIC),
        ("pagerank", Kind.NUMERIC),
        ("shortest_path_length", Kind.NUMERIC),
        ("ip_collision", Kind.NUMERIC),
        ("fraud_ip", Kind.NUMERIC),
        ("device_collision", Kind.NUMERIC),
        ("fraud_device", Kind.NUMERIC),
        ("trans_in_mule_ratio", Kind.NUMERIC),
        ("trans_out_mule_ratio", Kind.NUMERIC),
        ("mule_cnt", Kind.LOG_NUMERIC),
        ("com_id", Kind.DROP),  # a community identifier; meaningless as a scalar
    ],
    "Party": [
        ("is_fraud", Kind.DROP),  # the label
        ("gender", Kind.CATEGORICAL),
        ("dob", Kind.AGE),
        ("party_type", Kind.CATEGORICAL),
        ("name", Kind.DROP),  # unique per party; identity, not signal
        ("created_at", Kind.DATETIME),
    ],
}

# Attributes of the reified transaction vertex, lifted off the Transfer edge.
TRANSFER_NODE_ATTRS: List[Tuple[str, Kind]] = [
    ("amount", Kind.LOG_NUMERIC),
    ("transfer_time", Kind.DATETIME),
]

NODE_TYPES: List[str] = list(VERTEX_ATTRS)

FEATURELESS_TYPES: Set[str] = {t for t, attrs in VERTEX_ATTRS.items() if not attrs}


# --- Leakage control --------------------------------------------------------
# Dropped unconditionally. `Party.is_fraud` is the label. `Account.is_fraud` is
# worse than it looks here: the graph holds exactly 5000 Parties, 5000 Accounts
# and 5000 Party_Has_Account edges, with 49 fraud on each side -- the mapping is
# 1:1, so Account.is_fraud is the label copied one hop away.
#
# `Account.shortest_path_length` is in the same category, established by
# measurement rather than suspicion: it is 0 for all 49 fraudulent accounts and
# 1 or 2 for all 4951 others -- rank AUC 1.000 on its own. It measures distance
# to the nearest known fraud node, and a fraud node is distance 0 from itself,
# so it is the label under another name. Left in, it produces a perfect score
# that means nothing. `features.leakage_audit` re-derives this from the data
# and will flag any future attribute that behaves the same way.
ALWAYS_DROP: Set[Tuple[str, str]] = {
    ("Party", "is_fraud"),
    ("Account", "is_fraud"),
    ("Account", "shortest_path_length"),
}

# Precomputed Account features that look label-derived. Kept by default;
# `--exclude-derived` drops them to measure how much of the score they carry.
DERIVED_ATTRS: Set[Tuple[str, str]] = {
    ("Account", "fraud_device"),
    ("Account", "fraud_ip"),
    ("Account", "mule_cnt"),
    ("Account", "trans_in_mule_ratio"),
    ("Account", "trans_out_mule_ratio"),
    ("Account", "shortest_path_length"),
    ("Account", "com_id"),
    ("Account", "com_size"),
}

# This schema has no Connected_Component vertex, so unlike the earlier one
# there is no structurally-leaking relation to exclude. Kept as empty sets so
# the exclusion machinery stays in place if such a relation is added later.
DERIVED_EDGES: Set[EdgeTriple] = set()
DERIVED_NODE_TYPES: Set[str] = set()


# --- Edge types -------------------------------------------------------------
# All `Has_*` relations are undirected in TigerGraph; `Transfer` is directed.
# TigerGraph's auto-generated `reverse_Transfer` is deliberately absent:
# build_hetero_data applies ToUndirected, which generates reverses for every
# relation, and pulling TigerGraph's own would double-count it.
BASE_EDGE_TRIPLES: List[EdgeTriple] = [
    ("Party", "Party_Has_Account", "Account"),
    ("Party", "Has_ID", "ID"),
    ("Party", "Has_IP", "IP"),
    ("Party", "Has_Device", "Device"),
    ("Party", "Has_Phone", "Phone"),
    ("Party", "Has_Email", "Email"),
    ("Party", "Has_DOB", "DOB"),
    ("Party", "Has_Full_Name", "Full_Name"),
    ("Address_v2", "Has_Address_v2", "Party"),
    ("Account", "Transfer", "Account"),
]

TRANSFER_EDGE: EdgeTriple = ("Account", "Transfer", "Account")

# Replacements when transfers are reified into vertices.
REIFIED_TRANSFER_EDGES: List[EdgeTriple] = [
    ("Account", "Send_Transfer", TRANSFER_NODE),
    (TRANSFER_NODE, "Receive_Transfer", "Account"),
]

# Edge attributes present in TigerGraph. Only consumable by a conv that accepts
# edge features (see model.HeteroEdgeGNN); HGTConv ignores them entirely, which
# is the whole reason reification exists.
EDGE_ATTRS: Dict[EdgeTriple, List[Tuple[str, Kind]]] = {
    TRANSFER_EDGE: [
        ("amount", Kind.LOG_NUMERIC),
        ("transfer_time", Kind.DATETIME),
    ],
}


# --- Neighbor sampling fan-out ---------------------------------------------
DEFAULT_FANOUT: List[int] = [15, 10, 5, 5]

# This graph has no severe hubs -- unlike the schema originally targeted, there
# is no State vertex joining every address, and DOB is nearly unique (4505 DOB
# vertices for 5000 Parties). The widest relation is Transfer at ~20 per
# Account. Identity nodes are only mildly shared: 999 IPs and 997 Devices
# across 5000 Parties, so ~5 Parties per node in the reverse direction. Those
# are capped modestly rather than hard, since at this width the sharing IS the
# fraud signal and over-capping would discard it.
HUB_RELATIONS: List[EdgeTriple] = [
    ("Party", "Has_IP", "IP"),
    ("Party", "Has_Device", "Device"),
    ("Party", "Has_DOB", "DOB"),
]
HUB_FANOUT_WIDTH: List[int] = [8, 5, 3, 3]


def node_types(reify_transfer: bool = False, exclude_derived: bool = False) -> List[str]:
    """Vertex types that make it into the graph."""
    out = [t for t in NODE_TYPES if not (exclude_derived and t in DERIVED_NODE_TYPES)]
    if reify_transfer:
        out = out + [TRANSFER_NODE]
    return out


def edge_triples(reify_transfer: bool = False, exclude_derived: bool = False) -> List[EdgeTriple]:
    """Relations that make it into the graph.

    Reification swaps the single Account->Account Transfer relation for the
    Send/Receive pair through the transaction vertex.
    """
    dropped_nodes = DERIVED_NODE_TYPES if exclude_derived else set()
    out: List[EdgeTriple] = []
    for triple in BASE_EDGE_TRIPLES:
        if exclude_derived and triple in DERIVED_EDGES:
            continue
        src, _, dst = triple
        if src in dropped_nodes or dst in dropped_nodes:
            continue
        if reify_transfer and triple == TRANSFER_EDGE:
            out.extend(REIFIED_TRANSFER_EDGES)
            continue
        out.append(triple)
    return out


def attrs_for(node_type: str) -> List[Tuple[str, Kind]]:
    """Declared attributes, including the reified transaction vertex."""
    if node_type == TRANSFER_NODE:
        return list(TRANSFER_NODE_ATTRS)
    return VERTEX_ATTRS[node_type]


def active_attrs(node_type: str, exclude_derived: bool = False) -> List[Tuple[str, Kind]]:
    """Attributes of `node_type` that become model features."""
    out = []
    for name, kind in attrs_for(node_type):
        if kind is Kind.DROP:
            continue
        if (node_type, name) in ALWAYS_DROP:
            continue
        if exclude_derived and (node_type, name) in DERIVED_ATTRS:
            continue
        out.append((name, kind))
    return out


def is_hub(edge_type: EdgeTriple) -> bool:
    """Whether a relation is hub-like, in either direction.

    ToUndirected names generated reverses `rev_<rel>`, so a reverse relation is
    resolved back to its forward triple before the lookup.
    """
    src, rel, dst = edge_type
    if rel.startswith("rev_"):
        src, rel, dst = dst, rel[4:], src
    return (src, rel, dst) in set(HUB_RELATIONS)


def build_fanout(
    num_hops: int,
    edge_types=None,
    default: List[int] = None,
    hub: List[int] = None,
    reify_transfer: bool = False,
    exclude_derived: bool = False,
) -> Dict[EdgeTriple, List[int]]:
    """Per-edge-type `num_neighbors` dict for NeighborLoader.

    Pass `edge_types` (normally `data.edge_types`) to key the fan-out off the
    graph that actually exists. Deriving it from the schema flags instead is
    fragile: reification swaps Transfer for Send/Receive, and NeighborLoader
    raises if any relation in the graph is missing a fan-out entry.

    Without `edge_types` it falls back to the declared relations plus their
    `rev_*` counterparts.
    """
    default = list(default or DEFAULT_FANOUT)
    hub = list(hub or HUB_FANOUT_WIDTH)
    default = (default + [default[-1]] * num_hops)[:num_hops]
    hub = (hub + [hub[-1]] * num_hops)[:num_hops]

    if edge_types is None:
        edge_types = []
        for src, rel, dst in edge_triples(reify_transfer, exclude_derived):
            edge_types.append((src, rel, dst))
            edge_types.append((dst, f"rev_{rel}", src))

    return {
        tuple(edge_type): (hub if is_hub(tuple(edge_type)) else default)
        for edge_type in edge_types
    }
