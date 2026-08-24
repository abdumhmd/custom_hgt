"""
Conformance tests for the Mule_Account_Detection pipeline.

Run directly (`python test_schema.py`) or under pytest. These guard the things
that fail silently: a node type losing its features, a label leaking into an
input, splits overlapping or losing a class, reification dropping transactions,
or the NeighborLoader seed convention drifting out of sync with the loss.

They use the synthetic generator, so they need no database. `preflight.py`
checks the same schema against the live instance.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import features
import schema
from data import build_hetero_data, make_synthetic_frames, make_synthetic_graph

EXPECTED_NODE_TYPES = 10
EXPECTED_EDGE_TRIPLES = 10

SMALL = dict(n_parties=400, n_rings=6)


def test_schema_constants():
    assert len(schema.NODE_TYPES) == EXPECTED_NODE_TYPES
    assert len(schema.BASE_EDGE_TRIPLES) == EXPECTED_EDGE_TRIPLES
    assert len(set(schema.BASE_EDGE_TRIPLES)) == EXPECTED_EDGE_TRIPLES
    assert len(schema.FEATURELESS_TYPES) == 4

    for src, rel, dst in schema.BASE_EDGE_TRIPLES:
        assert src in schema.VERTEX_ATTRS, f"{src} in {rel} is not a declared vertex type"
        assert dst in schema.VERTEX_ATTRS, f"{dst} in {rel} is not a declared vertex type"

    # TigerGraph's auto-reverse must never be declared; ToUndirected makes its own
    assert not any(rel.startswith("reverse_") for _, rel, _ in schema.BASE_EDGE_TRIPLES)


def test_metadata_matches_schema():
    data = make_synthetic_graph(reify_transfer=False, **SMALL)
    node_types, edge_types = data.metadata()

    assert set(node_types) == set(schema.NODE_TYPES)
    assert len(edge_types) == 2 * EXPECTED_EDGE_TRIPLES
    for triple in schema.BASE_EDGE_TRIPLES:
        src, rel, dst = triple
        assert triple in edge_types
        assert (dst, f"rev_{rel}", src) in edge_types


def test_reification_preserves_transfers():
    """Every Transfer edge must become exactly one transaction vertex."""
    frames, edge_indices, edge_frames = make_synthetic_frames(**SMALL)
    n_transfers = edge_indices[schema.TRANSFER_EDGE].size(1)

    data = build_hetero_data(frames, edge_indices, edge_frames, reify_transfer=True)

    assert data[schema.TRANSFER_NODE].num_nodes == n_transfers
    send = ("Account", "Send_Transfer", schema.TRANSFER_NODE)
    receive = (schema.TRANSFER_NODE, "Receive_Transfer", "Account")
    assert data[send].edge_index.size(1) == n_transfers
    assert data[receive].edge_index.size(1) == n_transfers
    # the original relation is gone, replaced by the pair
    assert schema.TRANSFER_EDGE not in data.edge_types

    # each transaction vertex is used exactly once on each side
    assert torch.equal(
        torch.sort(data[send].edge_index[1])[0], torch.arange(n_transfers)
    )
    assert torch.equal(
        torch.sort(data[receive].edge_index[0])[0], torch.arange(n_transfers)
    )

    # and the endpoints still line up with the original edge
    original = edge_indices[schema.TRANSFER_EDGE]
    assert torch.equal(data[send].edge_index[0], original[0])
    assert torch.equal(data[receive].edge_index[1], original[1])


def test_reification_carries_edge_attributes_to_nodes():
    """amount/transfer_time are invisible to HGTConv on an edge; reification
    is the whole reason they reach the model at all."""
    frames, edge_indices, edge_frames = make_synthetic_frames(**SMALL)

    reified = build_hetero_data(frames, edge_indices, edge_frames, reify_transfer=True)
    assert reified[schema.TRANSFER_NODE].x.size(1) > 1  # more than just degree

    plain = build_hetero_data(frames, edge_indices, edge_frames, reify_transfer=False)
    assert "edge_attr" in plain[schema.TRANSFER_EDGE]
    assert plain[schema.TRANSFER_EDGE].edge_attr.size(1) > 0


def test_every_node_type_has_features():
    for reify in (True, False):
        data = make_synthetic_graph(reify_transfer=reify, **SMALL)
        dims = features.build_node_feat_dims(data)
        for node_type in data.node_types:
            assert "x" in data[node_type], f"{node_type} has no x"
            assert data[node_type].x.size(1) > 0, f"{node_type} has zero-width features"
            assert data[node_type].x.size(0) == data[node_type].num_nodes
            assert dims[node_type] == data[node_type].x.size(1)
            assert torch.isfinite(data[node_type].x).all()


def test_constant_attributes_are_pruned():
    """Address_v2.city and .population are constant in the live data and here."""
    frames, edge_indices, _ = make_synthetic_frames(**SMALL)
    dead = features.constant_attrs("Address_v2", frames["Address_v2"])
    assert "city" in dead and "population" in dead

    encoded = features.encode_vertex_frame("Address_v2", frames["Address_v2"])
    kept = features.encode_vertex_frame(
        "Address_v2", frames["Address_v2"], drop_constant=False
    )
    assert encoded.size(1) < kept.size(1)


def test_label_never_enters_features():
    """Perturbing is_fraud must not change any encoded feature."""
    frames, _, _ = make_synthetic_frames(**SMALL)

    for node_type in ("Party", "Account"):
        baseline = features.encode_vertex_frame(node_type, frames[node_type])
        flipped = frames[node_type].copy()
        flipped["is_fraud"] = 1 - flipped["is_fraud"].to_numpy()
        perturbed = features.encode_vertex_frame(node_type, flipped)
        assert torch.equal(baseline, perturbed), (
            f"{node_type}.is_fraud leaked into the encoded features"
        )

    for node_type, attr in schema.ALWAYS_DROP:
        assert attr not in [a for a, _ in schema.active_attrs(node_type)]


def test_shortest_path_length_is_always_dropped():
    """It is 0 for every fraud account and 1-2 for every other one in the live
    data -- the label under another name, so it must never be a feature."""
    assert ("Account", "shortest_path_length") in schema.ALWAYS_DROP
    assert "shortest_path_length" not in [a for a, _ in schema.active_attrs("Account")]


def test_leakage_audit_catches_a_perfect_predictor():
    """Plant a label copy and confirm the audit reports it."""
    frames, edge_indices, _ = make_synthetic_frames(**SMALL)

    clean = features.leakage_audit(frames, edge_indices)
    assert not any(name == "pagerank" for _, name, _ in clean)

    planted = {k: v.copy() for k, v in frames.items()}
    planted["Party"]["pagerank_like"] = planted["Party"]["is_fraud"].to_numpy() * 1.0
    original = schema.VERTEX_ATTRS["Party"]
    schema.VERTEX_ATTRS["Party"] = original + [("pagerank_like", schema.Kind.NUMERIC)]
    try:
        found = features.leakage_audit(planted, edge_indices)
    finally:
        schema.VERTEX_ATTRS["Party"] = original

    assert any(name == "pagerank_like" and auc > 0.99 for _, name, auc in found)


def test_stratified_split_balances_positives():
    """With ~1% prevalence an unstratified split can starve validation."""
    data = make_synthetic_graph(split_mode="stratified", **SMALL)
    target = data[schema.TARGET_TYPE]

    rates = []
    for name in ("train_mask", "val_mask", "test_mask"):
        mask = target[name]
        assert mask.sum() > 0
        assert target.y[mask].unique().numel() == 2, f"{name} is single-class"
        rates.append(target.y[mask].float().mean().item())

    # every split within a factor of two of the overall prevalence
    overall = target.y[target.labelled].float().mean().item()
    for rate in rates:
        assert 0.5 * overall <= rate <= 2.0 * overall


def test_splits_are_disjoint():
    data = make_synthetic_graph(**SMALL)
    target = data[schema.TARGET_TYPE]
    train, val, test = target.train_mask, target.val_mask, target.test_mask
    assert not (train & val).any()
    assert not (train & test).any()
    assert not (val & test).any()
    assert (train | val | test).sum() == target.labelled.sum()


def test_temporal_split_rejects_dead_timestamps():
    """Party.created_at is epoch-zero throughout the live data, so a temporal
    split there is meaningless and must fail loudly rather than silently."""
    frames, edge_indices, edge_frames = make_synthetic_frames(**SMALL)
    frames["Party"]["created_at"] = "1970-01-01 00:00:00"

    try:
        build_hetero_data(frames, edge_indices, edge_frames, split_mode="temporal")
    except ValueError as exc:
        assert "epoch-zero" in str(exc) or "identical" in str(exc)
    else:
        raise AssertionError("expected a ValueError for a constant timestamp")


def test_neighbor_loader_seed_alignment():
    """The loss slices out[:batch_size] assuming seeds come first."""
    from torch_geometric.loader import NeighborLoader

    data = make_synthetic_graph(**SMALL)
    target = schema.TARGET_TYPE
    idx = data[target].train_mask.nonzero(as_tuple=True)[0]

    loader = NeighborLoader(
        data,
        num_neighbors=schema.build_fanout(2, edge_types=data.edge_types),
        input_nodes=(target, idx),
        batch_size=16,
        shuffle=False,
    )
    batch = next(iter(loader))
    size = batch[target].batch_size
    assert torch.equal(batch[target].n_id[:size], idx[:size])
    assert torch.equal(batch[target].y[:size], data[target].y[idx[:size]])


def test_fanout_covers_every_relation():
    """A missing entry makes NeighborLoader raise at sample time."""
    for reify in (True, False):
        data = make_synthetic_graph(reify_transfer=reify, **SMALL)
        fanout = schema.build_fanout(4, edge_types=data.edge_types)
        assert set(fanout) == set(data.edge_types)

    # hubs are capped in both directions
    for src, rel, dst in schema.HUB_RELATIONS:
        assert schema.is_hub((src, rel, dst))
        assert schema.is_hub((dst, f"rev_{rel}", src))
    assert not schema.is_hub(("Party", "Party_Has_Account", "Account"))


def test_file_export_round_trip():
    from tg_loader import load_from_files

    frames, edge_indices, edge_frames = make_synthetic_frames(**SMALL)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "vertices").mkdir()
        (root / "edges").mkdir()

        for node_type, frame in frames.items():
            out = frame.copy()
            out.insert(0, "v_id", [f"{node_type}_{i}" for i in range(len(frame))])
            out.to_csv(root / "vertices" / f"{node_type}.csv", index=False)

        for triple, edge_index in edge_indices.items():
            src, rel, dst = triple
            columns = {
                "from": [f"{src}_{i}" for i in edge_index[0].tolist()],
                "to": [f"{dst}_{i}" for i in edge_index[1].tolist()],
            }
            if triple in edge_frames:
                for name in edge_frames[triple].columns:
                    columns[name] = edge_frames[triple][name].to_numpy()
            pd.DataFrame(columns).to_csv(root / "edges" / f"{src}__{rel}__{dst}.csv", index=False)

        loaded = load_from_files(str(root))

    direct = build_hetero_data(frames, edge_indices, edge_frames)
    for node_type in direct.node_types:
        assert loaded[node_type].num_nodes == direct[node_type].num_nodes
        assert loaded[node_type].x.shape == direct[node_type].x.shape
    for triple in direct.edge_types:
        assert torch.equal(loaded[triple].edge_index, direct[triple].edge_index), triple


def test_both_models_run():
    from model import build_model

    for conv, reify in (("hgt", True), ("transformer", False)):
        data = make_synthetic_graph(reify_transfer=reify, **SMALL)
        model = build_model(
            conv=conv,
            node_feat_dims=features.build_node_feat_dims(data),
            edge_feat_dims=features.build_edge_feat_dims(data),
            metadata=data.metadata(),
            hidden_channels=16,
            num_heads=2,
            num_layers=2,
        )
        edge_attr_dict = {
            et: data[et].edge_attr for et in data.edge_types if "edge_attr" in data[et]
        }
        out = model(data.x_dict, data.edge_index_dict, edge_attr_dict)
        assert out.shape == (data[schema.TARGET_TYPE].num_nodes, 2)
        assert torch.isfinite(out).all()


def test_transformer_actually_consumes_edge_features():
    """Changing edge features must change the transformer's output."""
    from model import build_model

    data = make_synthetic_graph(reify_transfer=False, **SMALL)
    edge_feat_dims = features.build_edge_feat_dims(data)
    assert edge_feat_dims, "no edge features to test with"

    model = build_model(
        conv="transformer",
        node_feat_dims=features.build_node_feat_dims(data),
        edge_feat_dims=edge_feat_dims,
        metadata=data.metadata(),
        hidden_channels=16,
        num_heads=2,
        num_layers=2,
    ).eval()

    base = {et: data[et].edge_attr for et in data.edge_types if "edge_attr" in data[et]}
    perturbed = {et: attr + 5.0 for et, attr in base.items()}
    with torch.no_grad():
        a = model(data.x_dict, data.edge_index_dict, base)
        b = model(data.x_dict, data.edge_index_dict, perturbed)
    assert not torch.allclose(a, b), "edge features are not reaching the model"


def test_model_rejects_missing_feature_dim():
    from model import build_model

    data = make_synthetic_graph(**SMALL)
    dims = features.build_node_feat_dims(data)
    dims.pop("Phone")
    try:
        build_model(conv="hgt", node_feat_dims=dims, metadata=data.metadata(),
                    hidden_channels=16, num_heads=2)
    except ValueError as exc:
        assert "Phone" in str(exc)
    else:
        raise AssertionError("expected a ValueError for the missing feature dim")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"PASS  {test.__name__}")
        except Exception as exc:
            failures += 1
            print(f"FAIL  {test.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    raise SystemExit(1 if failures else 0)
