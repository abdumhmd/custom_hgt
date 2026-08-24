"""
Load `Mule_Account_Detection` out of TigerGraph into a model-ready `HeteroData`.

Feeds the same `data.build_hetero_data` assembly the synthetic generator uses,
so encoding, splits, reification and reverse edges are identical either way.

Credentials come from the environment, never from code:

    TG_HOST         e.g. http://10.0.0.76  or  https://mycluster.i.tgcloud.io
    TG_GRAPH        graph name
    TG_USERNAME     default "tigergraph"
    TG_PASSWORD     default "tigergraph"  (or TG_SECRET for token auth)
    TG_RESTPP_PORT  default 14240; use 9000 if REST++ is exposed separately
    TG_GS_PORT      default 14240

Three things this has to get right, all verified against a live 4.2.4 instance:

  * **Primary keys differ per vertex type** -- phone_number, email, dob,
    address_key, name, id. A query referencing `s.id` fails on most types, so
    edge pulls use `ListAccum<VERTEX>`, which prints external ids regardless of
    what the key attribute is called.

  * **`Transfer` carries `amount` and `transfer_time` on the edge.** They are
    pulled alongside the endpoints so reification can lift them onto the
    transaction vertices, or `--conv transformer` can read them off the edge.

  * **TigerGraph's `reverse_Transfer` is not pulled.** build_hetero_data applies
    ToUndirected, which generates a reverse for every relation; pulling
    TigerGraph's own on top would double-count it.
"""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import torch

import schema
from data import build_hetero_data

EdgeTriple = Tuple[str, str, str]

CACHE_DIR = Path(".cache")


# --------------------------------------------------------------------------
# Connection
# --------------------------------------------------------------------------
def connect():
    """Build a pyTigerGraph connection from environment variables."""
    try:
        from pyTigerGraph import TigerGraphConnection
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "the tigergraph source needs pyTigerGraph: pip install pyTigerGraph"
        ) from exc

    host = os.environ.get("TG_HOST")
    graph = os.environ.get("TG_GRAPH")
    if not host or not graph:
        raise EnvironmentError("set TG_HOST and TG_GRAPH to connect to TigerGraph")

    # Port layout varies by deployment. A local Community Edition instance
    # proxies both GSQL and REST++ through 14240; older self-hosted installs
    # expose REST++ separately on 9000; TGCloud uses 443 for both.
    conn = TigerGraphConnection(
        host=host,
        graphname=graph,
        username=os.environ.get("TG_USERNAME", "tigergraph"),
        password=os.environ.get("TG_PASSWORD", "tigergraph"),
        restppPort=os.environ.get("TG_RESTPP_PORT", "14240"),
        gsPort=os.environ.get("TG_GS_PORT", "14240"),
    )
    # Token auth is only needed when REST++ authentication is enabled, which it
    # is not by default on a local Community Edition instance.
    secret = os.environ.get("TG_SECRET")
    if secret:
        conn.getToken(secret)
    return conn


# --------------------------------------------------------------------------
# Pulls
# --------------------------------------------------------------------------
def fetch_vertex_frame(conn, node_type: str) -> pd.DataFrame:
    """Pull one vertex type as a DataFrame indexed by external vertex id.

    Attribute-less types come back with just their primary key column, which is
    dropped -- identity is carried by the index, not by a feature.
    """
    frame = conn.getVertexDataFrame(node_type)
    if frame is None or len(frame) == 0:
        return pd.DataFrame(index=pd.Index([], name="v_id"))
    frame = frame.set_index("v_id")
    frame.index = frame.index.astype(str)
    return frame


def fetch_edge_triple(conn, triple: EdgeTriple) -> pd.DataFrame:
    """Pull one (src, rel, dst) relation, with its edge attributes.

    Uses an interpreted GSQL query rather than the per-vertex REST endpoints: it
    pulls the whole relation in one round trip (all 99,796 Transfer edges in
    ~1s on the local instance) and pins both endpoint types, which keeps
    relations that share an edge name distinct.
    """
    src, rel, dst = triple
    directed = ">" if triple == schema.TRANSFER_EDGE else ""

    attrs = schema.EDGE_ATTRS.get(triple, [])
    accum_decls, accum_ops, prints = [], [], []
    for name, _ in attrs:
        # DATETIME accumulates as STRING to survive the JSON round trip intact
        accum_decls.append(f"ListAccum<STRING> @@{name};")
        accum_ops.append(f"@@{name} += to_string(e.{name})")
        prints.append(f"@@{name} AS {name}")

    query = f"""
    INTERPRET QUERY () FOR GRAPH {conn.graphname} {{
      ListAccum<VERTEX> @@src, @@dst;
      {" ".join(accum_decls)}
      start = {{{src}.*}};
      result = SELECT s FROM start:s -({rel}:e)-{directed} {dst}:t
               ACCUM @@src += s, @@dst += t{"".join("," + op for op in accum_ops)};
      PRINT @@src AS src, @@dst AS dst{"".join(", " + p for p in prints)};
    }}
    """
    response = conn.runInterpretedQuery(query)
    payload = response[0] if isinstance(response, list) else response

    columns = {
        "from": [str(v) for v in payload.get("src", [])],
        "to": [str(v) for v in payload.get("dst", [])],
    }
    for name, _ in attrs:
        columns[name] = payload.get(name, [])
    return pd.DataFrame(columns)


def fetch_raw(exclude_derived: bool = False, verbose: bool = True, use_cache: bool = True):
    """Pull raw frames and edge lists, before any encoding or splitting.

    Split out from `load_from_tigergraph` so a seed sweep can pull once and
    rebuild only the splits, rather than re-hitting the database per seed.
    """
    cache_path = CACHE_DIR / f"raw_{os.environ.get('TG_GRAPH', 'graph')}.pt"
    if use_cache and cache_path.exists():
        if verbose:
            print(f"loading cached raw pull from {cache_path}")
        return torch.load(cache_path, weights_only=False)

    conn = connect()
    print(f"connected to {conn.graphname} (TigerGraph {conn.getVer()})")

    vertex_frames: Dict[str, pd.DataFrame] = {}
    id_maps: Dict[str, Dict[str, int]] = {}
    # base types only -- Transfer_Transaction is created by reification, not pulled
    for node_type in schema.node_types(reify_transfer=False, exclude_derived=exclude_derived):
        frame = fetch_vertex_frame(conn, node_type)
        # contiguous 0..n-1 indices; PyG edge_index cannot use TigerGraph ids
        id_maps[node_type] = {vid: i for i, vid in enumerate(frame.index)}
        vertex_frames[node_type] = frame.reset_index(drop=True)
        if verbose:
            print(f"  {node_type}: {len(frame):,} vertices")

    edge_indices: Dict[EdgeTriple, torch.Tensor] = {}
    edge_frames: Dict[EdgeTriple, pd.DataFrame] = {}
    for triple in schema.edge_triples(reify_transfer=False, exclude_derived=exclude_derived):
        src, rel, dst = triple
        frame = fetch_edge_triple(conn, triple)
        edge_index, keep = _map_edges(frame, id_maps[src], id_maps[dst])
        edge_indices[triple] = edge_index
        if schema.EDGE_ATTRS.get(triple):
            edge_frames[triple] = frame.loc[keep, [n for n, _ in schema.EDGE_ATTRS[triple]]].reset_index(drop=True)
        if verbose:
            print(f"  {src} -{rel}-> {dst}: {edge_index.size(1):,} edges")

    raw = (vertex_frames, edge_indices, edge_frames)
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(raw, cache_path)
        if verbose:
            print(f"cached raw pull to {cache_path}")
    return raw


def load_from_tigergraph(
    exclude_derived: bool = False,
    reify_transfer: bool = True,
    split_mode: str = "stratified",
    seed: int = 0,
    use_cache: bool = True,
    verbose: bool = True,
):
    """Pull the whole graph and assemble it into `HeteroData`."""
    vertex_frames, edge_indices, edge_frames = fetch_raw(
        exclude_derived=exclude_derived, verbose=verbose, use_cache=use_cache
    )
    return build_hetero_data(
        vertex_frames=vertex_frames,
        edge_indices=edge_indices,
        edge_frames=edge_frames,
        exclude_derived=exclude_derived,
        reify_transfer=reify_transfer,
        split_mode=split_mode,
        seed=seed,
        verbose=verbose,
    )


# --------------------------------------------------------------------------
# File exports
# --------------------------------------------------------------------------
def load_from_files(
    directory: str,
    exclude_derived: bool = False,
    reify_transfer: bool = True,
    split_mode: str = "stratified",
    seed: int = 0,
):
    """Assemble from on-disk exports.

    Expected layout, csv or parquet:

        <dir>/vertices/<NodeType>.csv         first column the vertex id
        <dir>/edges/<Src>__<Rel>__<Dst>.csv   columns: from, to, [attrs...]

    The `Src__Rel__Dst` filename convention keeps relations that share an edge
    name from colliding.
    """
    root = Path(directory)

    def read(path_stem: Path) -> Optional[pd.DataFrame]:
        for suffix, reader in ((".parquet", pd.read_parquet), (".csv", pd.read_csv)):
            candidate = path_stem.with_suffix(suffix)
            if candidate.exists():
                return reader(candidate)
        return None

    vertex_frames: Dict[str, pd.DataFrame] = {}
    id_maps: Dict[str, Dict[str, int]] = {}
    for node_type in schema.node_types(reify_transfer=False, exclude_derived=exclude_derived):
        frame = read(root / "vertices" / node_type)
        if frame is None:
            raise FileNotFoundError(f"no export for vertex type {node_type} under {root}/vertices")
        ids = frame.iloc[:, 0].astype(str)
        id_maps[node_type] = {vid: i for i, vid in enumerate(ids)}
        vertex_frames[node_type] = frame.iloc[:, 1:].reset_index(drop=True)

    edge_indices: Dict[EdgeTriple, torch.Tensor] = {}
    edge_frames: Dict[EdgeTriple, pd.DataFrame] = {}
    for triple in schema.edge_triples(reify_transfer=False, exclude_derived=exclude_derived):
        src, rel, dst = triple
        frame = read(root / "edges" / f"{src}__{rel}__{dst}")
        if frame is None:
            print(f"  warning: no export for {src}-{rel}->{dst}, treating as empty")
            edge_indices[triple] = torch.empty((2, 0), dtype=torch.long)
            continue
        edge_index, keep = _map_edges(frame, id_maps[src], id_maps[dst])
        edge_indices[triple] = edge_index
        declared = [n for n, _ in schema.EDGE_ATTRS.get(triple, []) if n in frame.columns]
        if declared:
            edge_frames[triple] = frame.loc[keep, declared].reset_index(drop=True)

    return build_hetero_data(
        vertex_frames=vertex_frames,
        edge_indices=edge_indices,
        edge_frames=edge_frames,
        exclude_derived=exclude_derived,
        reify_transfer=reify_transfer,
        split_mode=split_mode,
        seed=seed,
    )


def _map_edges(
    frame: pd.DataFrame, src_map: Dict[str, int], dst_map: Dict[str, int]
) -> Tuple[torch.Tensor, pd.Series]:
    """Translate a from/to id frame into a contiguous PyG edge_index.

    Returns the edge_index and the boolean keep-mask, so any edge attribute
    columns can be filtered to exactly the rows that survived. Edges pointing at
    vertices that were not exported are dropped rather than silently mapped to
    index 0, which would fabricate connections.
    """
    if len(frame) == 0:
        return torch.empty((2, 0), dtype=torch.long), pd.Series(dtype=bool)

    src = frame["from"].astype(str).map(src_map)
    dst = frame["to"].astype(str).map(dst_map)

    valid = src.notna() & dst.notna()
    dropped = int((~valid).sum())
    if dropped:
        print(f"  warning: dropped {dropped} edges with unknown endpoints")

    edge_index = torch.stack([
        torch.tensor(src[valid].to_numpy(dtype="int64")),
        torch.tensor(dst[valid].to_numpy(dtype="int64")),
    ])
    return edge_index, valid
