"""
Build a heterogeneous, edge-attributed fraud graph from the IEEE-CIS Fraud
Detection dataset (Kaggle) -- data/ieee_fraud/{train_transaction,train_identity}.csv.

Unlike OGBN-MAG, this graph is heterogeneous *and* has real numeric edge
attributes, which is the gap both prior candidates (OGBN-MAG, the HF
travel-fraud-graphs dataset) had. `TransactionAmt` / `TransactionDT` mirror
this repo's own `Transfer.amount` / `transfer_time` -- just attached to
Card/Addr/EmailDomain/Device relations instead of Account/Account.

Node types
----------
- transaction (590,540): the classification target. `y = isFraud`.
  Real intrinsic features (amount, time-of-day/week, ProductCD, C1-14, D1-15,
  M1-9) -- this is the info HGT gets, since HGTConv cannot read edge_attr at
  all (same limitation `data.reify_transfers` works around in the main repo).
- card, addr, email_domain, device: featureless identity nodes (composite
  keys). Get structural (log-degree) features, same rationale as
  `features.structural_features` for Phone/Email/DOB in the main schema.

Edge types (all carry edge_attr = [log1p(amount), time_of_transaction_norm])
-----------------------------------------------------------------------
- (card, made, transaction)
- (transaction, to_addr, addr)
- (transaction, from_email, email_domain)   -- P_emaildomain (purchaser)
- (transaction, to_email, email_domain)     -- R_emaildomain (recipient)
- (transaction, via_device, device)         -- only for the ~24% of rows
  with a matching train_identity.csv row

This is the edge-attribute path HGT structurally cannot use and
HeteroEdgeGNN (TransformerConv(edge_dim=2)) can -- the actual reason
HeteroEdgeGNN exists in `model.py`.

An addr/email/device edge is simply omitted when the source column is NaN,
rather than routing every missing value through one giant "unknown" node --
that would create a hub connecting ~11-16% of all transactions with no
information content (the same identity-mirror trap flagged in this repo's
own README for TigerGraph's Has_ID/Has_Phone/etc).

Split: chronological on TransactionDT (70/15/15) -- this dataset actually
supports the temporal split the README says the live TigerGraph data cannot.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData


def _factorize(series: pd.Series) -> tuple[torch.Tensor, int]:
    codes, _ = pd.factorize(series.astype("string").fillna("__NA__"))
    return torch.from_numpy(codes.astype(np.int64)), int(codes.max()) + 1


def _log_degree(num_nodes: int, edge_index: torch.Tensor) -> torch.Tensor:
    deg = torch.zeros(num_nodes, dtype=torch.float)
    if edge_index.numel() > 0:
        deg.scatter_add_(0, edge_index, torch.ones(edge_index.size(0)))
    return torch.log1p(deg)


def build(root: str, cache_path: str, val_frac: float = 0.15, test_frac: float = 0.15) -> HeteroData:
    root = Path(root)
    tx = pd.read_csv(root / "train_transaction.csv")
    idn = pd.read_csv(root / "train_identity.csv")
    tx = tx.merge(idn, on="TransactionID", how="left")
    tx = tx.sort_values("TransactionDT").reset_index(drop=True)
    n = len(tx)

    data = HeteroData()

    # ---- transaction node features ----
    amt = np.log1p(tx["TransactionAmt"].to_numpy(dtype=np.float64))
    dt = tx["TransactionDT"].to_numpy(dtype=np.float64)
    hour = (dt / 3600) % 24
    dow = (dt / 86400) % 7
    hour_sin, hour_cos = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    dow_sin, dow_cos = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)

    product_oh = pd.get_dummies(tx["ProductCD"], prefix="product").to_numpy(dtype=np.float32)

    c_cols = [f"C{i}" for i in range(1, 15)]
    c_feats = np.log1p(tx[c_cols].fillna(0).clip(lower=0).to_numpy(dtype=np.float64))

    d_cols = [f"D{i}" for i in range(1, 16)]
    d_feats = tx[d_cols].fillna(-1).to_numpy(dtype=np.float64)
    d_feats = np.clip(d_feats, -1, 1000) / 1000.0

    m_cols = [f"M{i}" for i in range(1, 10)]
    m_feats = np.stack(
        [pd.factorize(tx[c].astype("string").fillna("__NA__"))[0].astype(np.float64) for c in m_cols],
        axis=1,
    )

    tx_x = np.concatenate(
        [
            amt[:, None], hour_sin[:, None], hour_cos[:, None], dow_sin[:, None], dow_cos[:, None],
            product_oh, c_feats, d_feats, m_feats,
        ],
        axis=1,
    ).astype(np.float32)
    data["transaction"].x = torch.from_numpy(tx_x)
    data["transaction"].y = torch.from_numpy(tx["isFraud"].to_numpy(dtype=np.int64).copy())

    # ---- chronological split ----
    n_train = int(n * (1 - val_frac - test_frac))
    n_val = int(n * val_frac)
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    data["transaction"].train_mask = train_mask
    data["transaction"].val_mask = val_mask
    data["transaction"].test_mask = test_mask

    edge_attr_base = np.stack([amt, (dt - dt.min()) / (dt.max() - dt.min())], axis=1).astype(np.float32)

    # card node (composite key)
    card_key = tx[["card1", "card2", "card3", "card5", "card6"]].astype("string").fillna("NA").agg("-".join, axis=1)
    card_codes, num_card = _factorize(card_key)
    data["card"].num_nodes = num_card

    # addr node (addr1 only; drop edge when missing)
    has_addr = tx["addr1"].notna().to_numpy()
    addr_codes_all, num_addr = _factorize(tx["addr1"])
    data["addr"].num_nodes = num_addr

    # email_domain node, shared vocabulary across P_ and R_ email columns
    email_all = pd.concat([tx["P_emaildomain"], tx["R_emaildomain"]], axis=0)
    _, num_email = _factorize(email_all)
    email_codes_full, _ = pd.factorize(email_all.astype("string").fillna("__NA__"))
    p_email_codes = email_codes_full[:n]
    r_email_codes = email_codes_full[n:]
    has_p_email = tx["P_emaildomain"].notna().to_numpy()
    has_r_email = tx["R_emaildomain"].notna().to_numpy()
    data["email_domain"].num_nodes = num_email

    # device node (composite key), only present for identity-matched rows
    has_device = tx["DeviceType"].notna().to_numpy() | tx["DeviceInfo"].notna().to_numpy()
    device_key = tx.loc[has_device, ["DeviceType", "DeviceInfo"]].astype("string").fillna("NA").agg("-".join, axis=1)
    device_codes_present, num_device = _factorize(device_key)
    data["device"].num_nodes = num_device

    def make_edges(mask, src_codes, dst_codes):
        idx = np.nonzero(mask)[0]
        return (
            torch.from_numpy(idx.astype(np.int64)),
            torch.tensor(src_codes[idx], dtype=torch.long) if not torch.is_tensor(src_codes) else src_codes[idx],
            torch.tensor(dst_codes[idx], dtype=torch.long) if not torch.is_tensor(dst_codes) else dst_codes[idx],
        )

    all_mask = np.ones(n, dtype=bool)

    tx_ids, src, dst = make_edges(all_mask, torch.arange(n), card_codes)
    data["card", "made", "transaction"].edge_index = torch.stack([dst, src])
    data["card", "made", "transaction"].edge_attr = torch.from_numpy(edge_attr_base[tx_ids.numpy()])

    tx_ids, src, dst = make_edges(has_addr, torch.arange(n), addr_codes_all)
    data["transaction", "to_addr", "addr"].edge_index = torch.stack([src, dst])
    data["transaction", "to_addr", "addr"].edge_attr = torch.from_numpy(edge_attr_base[tx_ids.numpy()])

    tx_ids, src, dst = make_edges(has_p_email, torch.arange(n), torch.from_numpy(p_email_codes.astype(np.int64)))
    data["transaction", "from_email", "email_domain"].edge_index = torch.stack([src, dst])
    data["transaction", "from_email", "email_domain"].edge_attr = torch.from_numpy(edge_attr_base[tx_ids.numpy()])

    tx_ids, src, dst = make_edges(has_r_email, torch.arange(n), torch.from_numpy(r_email_codes.astype(np.int64)))
    data["transaction", "to_email", "email_domain"].edge_index = torch.stack([src, dst])
    data["transaction", "to_email", "email_domain"].edge_attr = torch.from_numpy(edge_attr_base[tx_ids.numpy()])

    device_idx = np.nonzero(has_device)[0]
    data["transaction", "via_device", "device"].edge_index = torch.stack(
        [torch.from_numpy(device_idx.astype(np.int64)), device_codes_present]
    )
    data["transaction", "via_device", "device"].edge_attr = torch.from_numpy(edge_attr_base[device_idx])

    # ---- structural features for featureless node types ----
    for ntype in ["card", "addr", "email_domain", "device"]:
        num_nodes = data[ntype].num_nodes
        cols = []
        for (src_t, rel, dst_t), store in data.edge_index_dict.items():
            if src_t == ntype:
                cols.append(_log_degree(num_nodes, store[0]))
            if dst_t == ntype:
                cols.append(_log_degree(num_nodes, store[1]))
        data[ntype].x = torch.stack(cols, dim=1) if cols else torch.zeros(num_nodes, 1)

    torch.save(data, cache_path)
    return data


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/ieee_fraud")
    p.add_argument("--cache", default=".cache/ieee_fraud_graph.pt")
    args = p.parse_args()

    Path(args.cache).parent.mkdir(parents=True, exist_ok=True)
    data = build(args.root, args.cache)
    print(data)
    print(f"saved to {args.cache}")


if __name__ == "__main__":
    main()
