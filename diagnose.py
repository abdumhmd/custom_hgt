"""
Does this graph contain learnable fraud signal at all?

Run before training, not after. A GNN can only exploit structure that exists:
fraudulent parties clustering on shared identity nodes, transacting with each
other, or differing in volume or degree. If none of that is present, no
architecture will beat chance, and a chance-level result is the correct answer
rather than a bug to debug.

Each check compares an observed count against what pure chance would produce at
the same prevalence. Ratios near 1.0 mean no signal.

    python diagnose.py --source tigergraph
"""

import argparse
from typing import Dict, List

import numpy as np
import pandas as pd

import schema


def _shared_group_stats(pairs: np.ndarray, fraud_idx: set, prevalence: float):
    """Co-occurrence of fraud parties on a shared identity vertex."""
    frame = pd.DataFrame({"party": pairs[0], "other": pairs[1]})
    groups = frame.groupby("other")["party"].apply(list)
    multi = groups[groups.apply(len) > 1]

    observed = 0
    total_pairs = 0
    for members in multi:
        total_pairs += len(members) * (len(members) - 1) // 2
        flagged = [m for m in members if m in fraud_idx]
        observed += len(flagged) * (len(flagged) - 1) // 2

    expected = total_pairs * prevalence * prevalence
    return len(multi), observed, expected


def report(vertex_frames: Dict, edge_indices: Dict) -> List[str]:
    target = schema.TARGET_TYPE
    fraud = pd.to_numeric(vertex_frames[target][schema.LABEL_ATTR], errors="coerce").fillna(0)
    fraud = fraud.to_numpy()
    n_party = len(fraud)
    fraud_idx = set(np.where(fraud == 1)[0].tolist())
    prevalence = len(fraud_idx) / max(n_party, 1)

    print(f"{n_party:,} {target} vertices, {len(fraud_idx)} fraudulent "
          f"({prevalence:.2%} prevalence)\n")

    warnings: List[str] = []

    # ---- shared identity nodes -------------------------------------------
    print("== do fraudulent parties share identity vertices with each other? ==")
    print(f"{'relation':<20} {'shared':>8} {'obs':>6} {'exp':>7} {'ratio':>7}")
    for triple, edge_index in edge_indices.items():
        src, rel, dst = triple
        if target not in (src, dst) or triple == schema.TRANSFER_EDGE:
            continue
        if "Account" in (src, dst):
            continue
        pairs = edge_index.numpy()
        if src != target:
            pairs = pairs[::-1]

        n_shared, observed, expected = _shared_group_stats(pairs, fraud_idx, prevalence)
        if n_shared == 0:
            print(f"{rel:<20} {0:>8} {'-':>6} {'-':>7} {'1:1':>7}")
            warnings.append(
                f"{rel} is strictly 1:1 with {target} -- it carries no information, "
                "only an identity mirror"
            )
            continue
        ratio = observed / expected if expected > 0 else float("nan")
        print(f"{rel:<20} {n_shared:>8} {observed:>6} {expected:>7.1f} {ratio:>7.2f}")

    # ---- transaction structure -------------------------------------------
    print("\n== do fraudulent accounts transact with each other? ==")
    account_edge = (target, "Party_Has_Account", "Account")
    transfer = edge_indices.get(schema.TRANSFER_EDGE)
    if account_edge in edge_indices and transfer is not None:
        link = edge_indices[account_edge].numpy()
        n_accounts = len(vertex_frames["Account"])
        account_fraud = np.zeros(n_accounts)
        account_fraud[link[1]] = fraud[link[0]]

        t = transfer.numpy()
        src_f, dst_f = account_fraud[t[0]], account_fraud[t[1]]
        n_edges, rate = t.shape[1], account_fraud.mean()

        observed = int(((src_f == 1) & (dst_f == 1)).sum())
        expected = n_edges * rate * rate
        ratio = observed / expected if expected > 0 else float("nan")
        print(f"  fraud -> fraud transfers: {observed} observed, "
              f"{expected:.1f} expected, ratio {ratio:.2f}")
        if ratio < 2.0:
            warnings.append(
                f"fraudulent accounts transact with each other at {ratio:.2f}x chance "
                "-- there is no ring structure in the transaction graph"
            )

        for name, side in (("out", t[0]), ("in", t[1])):
            deg = np.bincount(side, minlength=n_accounts)
            f_deg, c_deg = deg[account_fraud == 1].mean(), deg[account_fraud == 0].mean()
            print(f"  {name}-degree: fraud {f_deg:.2f}, clean {c_deg:.2f}")

    print()
    if warnings:
        print("== findings ==")
        for w in warnings:
            print(f"  ! {w}")
        print(
            "\nA GNN cannot recover a signal the graph does not contain. If these "
            "checks are flat, a chance-level AUROC is the correct result -- verify "
            "the graph-feature pipeline actually ran before blaming the model."
        )
    else:
        print("structure looks exploitable: fraud clusters above chance somewhere")
    return warnings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["synthetic", "tigergraph"], default="tigergraph")
    args = parser.parse_args()

    if args.source == "tigergraph":
        from tg_loader import fetch_raw

        vertex_frames, edge_indices, _ = fetch_raw(verbose=False)
    else:
        from data import make_synthetic_frames

        vertex_frames, edge_indices, _ = make_synthetic_frames()

    report(vertex_frames, edge_indices)


if __name__ == "__main__":
    main()
