"""
Graph assembly for the `Mule_Account_Detection` schema, plus a synthetic
generator matching it for offline development.

`build_hetero_data` is the one path both sources go through -- the synthetic
generator here and the live pull in `tg_loader.py` -- so encoding, splits,
reification and reverse edges are identical either way.

On reification: TigerGraph stores transactions as a directed `Transfer` edge
carrying `amount` and `transfer_time`. HGTConv reads no edge attributes, so
under `reify_transfer=True` each Transfer edge becomes a Transfer_Transaction
vertex with Send_Transfer / Receive_Transfer edges either side, lifting those
attributes into node features where the model can see them. The alternative --
keeping the edge and using a conv that accepts edge features -- is
`model.HeteroEdgeGNN`, selected by `--conv transformer`.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected

import features
import schema

EdgeTriple = Tuple[str, str, str]


# --------------------------------------------------------------------------
# Reification
# --------------------------------------------------------------------------
def reify_transfers(
    edge_indices: Dict[EdgeTriple, torch.Tensor],
    edge_frames: Dict[EdgeTriple, pd.DataFrame],
) -> Tuple[Dict[EdgeTriple, torch.Tensor], pd.DataFrame]:
    """Turn Account->Transfer->Account edges into transaction vertices.

    Returns the rewritten edge dict and the attribute frame for the new
    vertices. Each transfer becomes its own vertex, so parallel transfers
    between the same pair of accounts stay distinct rather than collapsing.
    """
    transfer = edge_indices.get(schema.TRANSFER_EDGE)
    if transfer is None:
        return dict(edge_indices), pd.DataFrame()

    n_transfers = transfer.size(1)
    transaction_ids = torch.arange(n_transfers, dtype=torch.long)

    rewritten = {k: v for k, v in edge_indices.items() if k != schema.TRANSFER_EDGE}
    rewritten[("Account", "Send_Transfer", schema.TRANSFER_NODE)] = torch.stack(
        [transfer[0], transaction_ids]
    )
    rewritten[(schema.TRANSFER_NODE, "Receive_Transfer", "Account")] = torch.stack(
        [transaction_ids, transfer[1]]
    )

    frame = edge_frames.get(schema.TRANSFER_EDGE)
    if frame is None or len(frame) != n_transfers:
        # no attributes available; the vertices still carry structural degree
        frame = pd.DataFrame(index=range(n_transfers))
    return rewritten, frame.reset_index(drop=True)


# --------------------------------------------------------------------------
# Shared assembly
# --------------------------------------------------------------------------
def build_hetero_data(
    vertex_frames: Dict[str, pd.DataFrame],
    edge_indices: Dict[EdgeTriple, torch.Tensor],
    edge_frames: Optional[Dict[EdgeTriple, pd.DataFrame]] = None,
    exclude_derived: bool = False,
    reify_transfer: bool = True,
    split_mode: str = "stratified",
    seed: int = 0,
    verbose: bool = False,
) -> HeteroData:
    """Assemble encoded `HeteroData` from raw per-type frames and edge lists.

    Order matters:
      1. reify transfers, so the new vertices exist before anything counts them
      2. splits, so feature normalisation is fit on training rows only
      3. attribute encoding
      4. structural degree features, over the declared relations
      5. ToUndirected last, so it does not double the degree counts
    """
    edge_frames = dict(edge_frames or {})
    vertex_frames = dict(vertex_frames)

    if reify_transfer:
        edge_indices, transfer_frame = reify_transfers(edge_indices, edge_frames)
        vertex_frames[schema.TRANSFER_NODE] = transfer_frame

    node_types = schema.node_types(reify_transfer, exclude_derived)
    edge_triples = schema.edge_triples(reify_transfer, exclude_derived)

    missing = set(node_types) - set(vertex_frames)
    if missing:
        raise ValueError(f"missing vertex frames for: {sorted(missing)}")

    data = HeteroData()
    target = schema.TARGET_TYPE
    target_frame = vertex_frames[target]
    n_target = len(target_frame)

    # ---- Labels and splits ------------------------------------------------
    raw_label = pd.to_numeric(target_frame[schema.LABEL_ATTR], errors="coerce")
    labelled = torch.from_numpy(raw_label.notna().to_numpy().copy())
    y = torch.from_numpy(raw_label.fillna(0).to_numpy().copy()).long()

    generator = torch.Generator().manual_seed(seed)
    masks = features.make_splits(
        n=n_target,
        labelled=labelled,
        y=y,
        mode=split_mode,
        order_key=target_frame["created_at"] if split_mode == "temporal" else None,
        generator=generator,
    )
    train_mask_np = masks["train_mask"].numpy()

    # ---- Node attributes --------------------------------------------------
    for node_type in node_types:
        frame = vertex_frames[node_type]
        data[node_type].num_nodes = len(frame)
        data[node_type].x = features.encode_vertex_frame(
            node_type,
            frame,
            exclude_derived=exclude_derived,
            # only the target type has a training split to fit statistics on
            fit_mask=train_mask_np if node_type == target else None,
        )
        if verbose:
            dead = features.constant_attrs(node_type, frame, exclude_derived)
            if dead:
                print(f"  {node_type}: dropped constant attrs {dead}")

    data[target].y = y
    data[target].labelled = labelled
    for name, mask in masks.items():
        data[target][name] = mask

    # ---- Edges ------------------------------------------------------------
    for triple in edge_triples:
        edge_index = edge_indices.get(triple)
        if edge_index is None:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        data[triple].edge_index = edge_index.long()

        # Edge features survive only on the un-reified Transfer relation, and
        # are read only by a conv that accepts them (--conv transformer).
        frame = edge_frames.get(triple)
        if frame is not None and len(frame) == edge_index.size(1):
            attr = features.encode_edge_frame(triple, frame)
            if attr.size(1):
                data[triple].edge_attr = attr

    # ---- Structural features, then reverse relations ----------------------
    features.structural_features(data)
    data = ToUndirected(merge=False)(data)

    return data


# --------------------------------------------------------------------------
# Synthetic generator
# --------------------------------------------------------------------------
_STATES = [f"state_{i}" for i in range(47)]
_GENDERS = ["M", "F", "X"]
_PARTY_TYPES = ["individual", "business", "trust"]
_ACCOUNT_LEVELS = ["basic", "silver", "gold", "platinum"]
_ACCOUNT_TYPES = ["checking", "savings", "credit", "business"]
_ID_TYPES = ["passport", "drivers_license", "ssn", "national_id"]


def _random_dates(rng, n, start="2023-01-01", end="2024-12-31"):
    lo = pd.Timestamp(start).value // 10**9
    hi = pd.Timestamp(end).value // 10**9
    return pd.to_datetime(rng.integers(lo, hi, size=n), unit="s").strftime("%Y-%m-%d %H:%M:%S")


def make_synthetic_frames(
    n_parties: int = 5000,
    n_rings: int = 12,
    ring_size_range: Tuple[int, int] = (3, 6),
    seed: int = 0,
) -> Tuple[Dict[str, pd.DataFrame], Dict[EdgeTriple, torch.Tensor], Dict[EdgeTriple, pd.DataFrame]]:
    """Raw synthetic frames and edge lists on the live schema's shape.

    Deliberately mirrors the real graph's proportions: 1:1 Party-to-Account,
    ~1% fraud prevalence, ~1000 IPs and Devices shared across 5000 Parties,
    and ~20 transfers per Account.
    """
    rng = np.random.default_rng(seed)

    ring_of_party = np.full(n_parties, -1, dtype=np.int64)
    cursor = 0
    for ring in range(n_rings):
        size = rng.integers(*ring_size_range)
        if cursor + size >= n_parties:
            break
        ring_of_party[cursor : cursor + size] = ring
        cursor += size
    is_fraud = (ring_of_party >= 0).astype(np.int64)

    n_ips, n_devices = 999, 997
    n_dobs = 4505
    n_addresses = 4995

    def identity_assignment(n_pool: int, shared_per_ring: int) -> np.ndarray:
        assignment = np.empty(n_parties, dtype=np.int64)
        honest_slots = max(n_pool - n_rings * shared_per_ring, 1)
        for party in range(n_parties):
            ring = ring_of_party[party]
            if ring < 0:
                assignment[party] = rng.integers(0, honest_slots)
            else:
                base = honest_slots + ring * shared_per_ring
                assignment[party] = min(base + rng.integers(0, shared_per_ring), n_pool - 1)
        return assignment

    party_ip = identity_assignment(n_ips, 1)
    party_device = identity_assignment(n_devices, 1)
    party_address = identity_assignment(n_addresses, 1)
    parties = np.arange(n_parties)

    # 1:1 Party to Account, matching the live graph
    n_accounts = n_parties
    account_is_fraud = is_fraud

    # transfers: ring accounts transact mostly with each other
    src_list = rng.integers(0, n_accounts, size=n_accounts * 18).tolist()
    dst_list = rng.integers(0, n_accounts, size=n_accounts * 18).tolist()
    for ring in range(n_rings):
        members = np.where(ring_of_party == ring)[0]
        if len(members) < 2:
            continue
        n_internal = len(members) * 40
        src_list.extend(rng.choice(members, size=n_internal).tolist())
        dst_list.extend(rng.choice(members, size=n_internal).tolist())

    transfer_src = np.array(src_list, dtype=np.int64)
    transfer_dst = np.array(dst_list, dtype=np.int64)
    keep = transfer_src != transfer_dst
    transfer_src, transfer_dst = transfer_src[keep], transfer_dst[keep]
    n_transfers = len(transfer_src)

    def noisy(signal: np.ndarray, scale: float) -> np.ndarray:
        return np.clip(signal * scale + rng.normal(0, scale * 0.9, size=len(signal)), 0, None)

    frames: Dict[str, pd.DataFrame] = {
        "Party": pd.DataFrame({
            "is_fraud": is_fraud,
            "gender": rng.choice(_GENDERS, size=n_parties),
            "dob": _random_dates(rng, n_parties, "1950-01-01", "2005-12-31"),
            "party_type": rng.choice(_PARTY_TYPES, size=n_parties, p=[0.8, 0.15, 0.05]),
            "name": [f"party_{i}" for i in range(n_parties)],
            "created_at": _random_dates(rng, n_parties),
        }),
        "Account": pd.DataFrame({
            "create_Time": _random_dates(rng, n_accounts),
            "is_fraud": account_is_fraud,
            "account_type": rng.choice(_ACCOUNT_TYPES, size=n_accounts),
            "account_level": rng.choice(_ACCOUNT_LEVELS, size=n_accounts),
            "com_size": rng.integers(1, 4, size=n_accounts),
            "pagerank": rng.gamma(2.0, 0.5, size=n_accounts),
            "shortest_path_length": np.where(
                account_is_fraud == 1, rng.integers(1, 3, n_accounts), rng.integers(2, 7, n_accounts)
            ),
            "ip_collision": noisy(account_is_fraud, 3.0).astype(np.int64),
            "fraud_ip": noisy(account_is_fraud, 2.0).astype(np.int64),
            "device_collision": noisy(account_is_fraud, 3.0).astype(np.int64),
            "fraud_device": noisy(account_is_fraud, 2.0).astype(np.int64),
            "trans_in_mule_ratio": np.clip(noisy(account_is_fraud, 0.3), 0, 1),
            "trans_out_mule_ratio": np.clip(noisy(account_is_fraud, 0.3), 0, 1),
            "mule_cnt": noisy(account_is_fraud, 4.0).astype(np.int64),
            "com_id": rng.integers(0, 10**7, size=n_accounts),
        }),
        "Address_v2": pd.DataFrame({
            "address_line1": [f"addr_{i}" for i in range(n_addresses)],
            "zipcode": rng.integers(10000, 99999, size=n_addresses).astype(str),
            "city": [""] * n_addresses,  # unpopulated in the live data too
            "state": rng.choice(_STATES, size=n_addresses),
            "population": np.zeros(n_addresses, dtype=np.int64),
        }),
        "ID": pd.DataFrame({"id_type": rng.choice(_ID_TYPES, size=n_parties)}),
        "IP": pd.DataFrame({"is_blocked": rng.random(n_ips) < 0.03}),
        "Device": pd.DataFrame({"is_blocked": rng.random(n_devices) < 0.02}),
        "Phone": pd.DataFrame(index=range(n_parties)),
        "Email": pd.DataFrame(index=range(n_parties)),
        "Full_Name": pd.DataFrame(index=range(n_parties)),
        "DOB": pd.DataFrame(index=range(n_dobs)),
    }

    def stack(src, dst) -> torch.Tensor:
        return torch.stack([
            torch.as_tensor(np.asarray(src), dtype=torch.long),
            torch.as_tensor(np.asarray(dst), dtype=torch.long),
        ])

    edge_indices: Dict[EdgeTriple, torch.Tensor] = {
        ("Party", "Party_Has_Account", "Account"): stack(parties, np.arange(n_accounts)),
        ("Party", "Has_ID", "ID"): stack(parties, parties),
        ("Party", "Has_IP", "IP"): stack(parties, party_ip),
        ("Party", "Has_Device", "Device"): stack(parties, party_device),
        ("Party", "Has_Phone", "Phone"): stack(parties, parties),
        ("Party", "Has_Email", "Email"): stack(parties, parties),
        ("Party", "Has_DOB", "DOB"): stack(parties, rng.integers(0, n_dobs, size=n_parties)),
        ("Party", "Has_Full_Name", "Full_Name"): stack(parties, parties),
        ("Address_v2", "Has_Address_v2", "Party"): stack(party_address, parties),
        schema.TRANSFER_EDGE: stack(transfer_src, transfer_dst),
    }
    edge_frames: Dict[EdgeTriple, pd.DataFrame] = {
        schema.TRANSFER_EDGE: pd.DataFrame({
            "amount": rng.lognormal(11.0, 1.5, size=n_transfers),
            "transfer_time": _random_dates(rng, n_transfers),
        }),
    }
    return frames, edge_indices, edge_frames


def make_synthetic_graph(
    exclude_derived: bool = False,
    reify_transfer: bool = True,
    split_mode: str = "stratified",
    seed: int = 0,
    verbose: bool = False,
    **kwargs,
) -> HeteroData:
    """Generate a schema-accurate synthetic graph, ready for training."""
    frames, edge_indices, edge_frames = make_synthetic_frames(seed=seed, **kwargs)
    return build_hetero_data(
        vertex_frames=frames,
        edge_indices=edge_indices,
        edge_frames=edge_frames,
        exclude_derived=exclude_derived,
        reify_transfer=reify_transfer,
        split_mode=split_mode,
        seed=seed,
        verbose=verbose,
    )


if __name__ == "__main__":
    data = make_synthetic_graph(verbose=True)
    print(data)
    print("\nnode feature dims:", features.build_node_feat_dims(data))
    print("fraud prevalence:", data[schema.TARGET_TYPE].y.float().mean().item())
