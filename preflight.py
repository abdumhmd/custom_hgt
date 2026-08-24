"""
Connection and schema conformance check against a live TigerGraph instance.

Run this before any training run on real data. It is read-only and cheap: it
pulls the schema and per-type counts, never vertex payloads. It answers the
questions that would otherwise surface as a confusing crash forty minutes into
a full pull:

  * do the credentials and endpoints actually work
  * does the live schema match schema.py -- every vertex type, every attribute
    name, every edge endpoint pair
  * how big is each type, i.e. will a full in-memory pull fit
  * is Party.is_fraud actually populated, and at what prevalence

Usage:
    set -a && source .env && set +a && python preflight.py

Secrets are read from the environment and never printed.
"""

import os
import sys
from typing import Dict, List

import schema

CHECK = "  ok  "
WARN = " warn "
FAIL = " FAIL "


def _fmt(count) -> str:
    if not isinstance(count, (int, float)):
        return str(count)
    return f"{int(count):,}"


def check_connection():
    from tg_loader import connect

    print("== connection ==")
    for var in ("TG_HOST", "TG_GRAPH"):
        if not os.environ.get(var):
            print(f"[{FAIL}] {var} is not set")
            sys.exit(1)
    # host is not a secret; password/secret are never printed
    print(f"        host  {os.environ['TG_HOST']}")
    print(f"        graph {os.environ['TG_GRAPH']}")
    auth = "TG_SECRET" if os.environ.get("TG_SECRET") else "TG_PASSWORD"
    print(f"        auth  via {auth}")

    conn = connect()
    version = conn.getVer()
    print(f"[{CHECK}] connected, TigerGraph {version}")
    return conn


def _schema_edge_pairs(edge_def: Dict) -> List[tuple]:
    """Endpoint pairs for an edge type.

    TigerGraph reports a single-pair edge via FromVertexTypeName/ToVertexTypeName
    and a multi-pair edge (like Assigned_To and Located_In here) via EdgePairs.
    """
    pairs = edge_def.get("EdgePairs")
    if pairs:
        return [(p["From"], p["To"]) for p in pairs]
    return [(edge_def.get("FromVertexTypeName"), edge_def.get("ToVertexTypeName"))]


def check_vertex_types(conn) -> bool:
    print("\n== vertex types ==")
    live = set(conn.getVertexTypes())
    expected = set(schema.NODE_TYPES)
    ok = True

    for missing in sorted(expected - live):
        print(f"[{FAIL}] {missing}: declared in schema.py, absent from the graph")
        ok = False
    for extra in sorted(live - expected):
        print(f"[{WARN}] {extra}: in the graph, not declared in schema.py (ignored)")

    for node_type in sorted(expected & live):
        live_attrs = {a["AttributeName"] for a in conn.getVertexType(node_type)["Attributes"]}
        declared = {name for name, _ in schema.VERTEX_ATTRS[node_type]}

        missing = declared - live_attrs
        extra = live_attrs - declared
        if missing:
            print(f"[{FAIL}] {node_type}: declared attrs missing from graph: {sorted(missing)}")
            ok = False
        if extra:
            print(f"[{WARN}] {node_type}: graph attrs not in schema.py: {sorted(extra)}")
        if not missing and not extra:
            print(f"[{CHECK}] {node_type}: {len(declared)} attrs match")
    return ok


def check_edge_types(conn) -> bool:
    print("\n== edge types ==")
    live_names = set(conn.getEdgeTypes())
    ok = True

    live_triples = set()
    for name in live_names:
        for src, dst in _schema_edge_pairs(conn.getEdgeType(name)):
            live_triples.add((src, name, dst))

    expected = set(schema.BASE_EDGE_TRIPLES)
    for triple in sorted(expected - live_triples):
        print(f"[{FAIL}] {triple[0]} -{triple[1]}-> {triple[2]}: not found in the graph")
        ok = False

    for triple in sorted(live_triples - expected):
        src, rel, dst = triple
        # TigerGraph's auto-generated reverse edges are expected to be unused:
        # build_hetero_data generates its own reverses via ToUndirected.
        if rel.startswith("reverse_"):
            print(f"[{CHECK}] {src} -{rel}-> {dst}: auto-reverse, intentionally not loaded")
        else:
            print(f"[{WARN}] {src} -{rel}-> {dst}: in graph, not declared in schema.py")

    for triple in sorted(expected & live_triples):
        print(f"[{CHECK}] {triple[0]} -{triple[1]}-> {triple[2]}")

    # any edge name covering more than one endpoint pair must be pulled per
    # triple, not per name, or the relations merge
    by_name = {}
    for src, rel, dst in live_triples:
        by_name.setdefault(rel, []).append((src, dst))
    for rel, pairs in sorted(by_name.items()):
        if len(pairs) > 1:
            print(f"[{WARN}] {rel}: covers {len(pairs)} endpoint pairs {sorted(pairs)}")
    return ok


def report_sizes(conn):
    print("\n== size ==")
    try:
        vertex_counts = conn.getVertexCount("*")
    except Exception as exc:
        print(f"[{WARN}] could not read vertex counts: {exc}")
        return

    total = 0
    for node_type in schema.NODE_TYPES:
        count = vertex_counts.get(node_type, "?")
        if isinstance(count, int):
            total += count
        print(f"        {node_type:<22} {_fmt(count):>14}")
    print(f"        {'TOTAL VERTICES':<22} {_fmt(total):>14}")

    try:
        edge_counts = conn.getEdgeCount()
        edge_total = sum(v for v in edge_counts.values() if isinstance(v, int))
        print(f"        {'TOTAL EDGES':<22} {_fmt(edge_total):>14}")
    except Exception as exc:
        print(f"[{WARN}] could not read edge counts: {exc}")
        edge_total = 0

    if total > 5_000_000 or edge_total > 20_000_000:
        print(
            f"\n[{WARN}] this is too large for the current full-pull loader, which "
            "materialises every vertex and edge in memory. Pull a subgraph first, "
            "or move to torch_geometric.distributed / GraphStorm."
        )


def report_label(conn):
    print("\n== label ==")
    target, attr = schema.TARGET_TYPE, schema.LABEL_ATTR
    try:
        total = conn.getVertexCount(target)
        positives = conn.getVertexCount(target, where=f"{attr}>0")
    except Exception as exc:
        print(f"[{WARN}] could not read {target}.{attr}: {exc}")
        return

    print(f"        {target} total      {_fmt(total)}")
    print(f"        {attr}!=0          {_fmt(positives)}")
    if isinstance(total, int) and isinstance(positives, int) and total:
        print(f"        prevalence         {positives / total:.4%}")
        if positives == 0:
            print(f"[{FAIL}] no positive labels -- nothing to train against")
        elif positives < 200:
            n_val = int(0.15 * positives)
            print(
                f"[{WARN}] only {positives} positives. A 70/15/15 split leaves ~{n_val} "
                "in validation and the same in test, so those metrics will be very "
                "noisy. Use --split stratified (the default) at minimum."
            )


def main():
    conn = check_connection()
    vertices_ok = check_vertex_types(conn)
    edges_ok = check_edge_types(conn)
    report_sizes(conn)
    report_label(conn)

    print()
    if vertices_ok and edges_ok:
        print("schema matches schema.py -- safe to try: python train.py --source tigergraph")
        return 0
    print("schema does NOT match schema.py -- fix the mismatches above before training")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
