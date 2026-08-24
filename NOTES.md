# Deep-dive notes: the TigerGraph / synthetic fraud graph

This is the investigation log behind `--dataset tigergraph` / `--dataset synthetic`
(the `Mule_Account_Detection` schema). Read the main `README.md` first — this is
detail, not orientation.

## Result on the live graph: no learnable signal

Measured, 5 seeds, stratified split, leak removed:

| | mean | sd | range |
|---|---|---|---|
| test AUROC | **0.514** | 0.066 | 0.410 – 0.586 |
| test AP | **0.014** | 0.003 | 0.009 – 0.017 |

That is chance (AP base rate ≈ 0.011). It is not a pipeline bug — two positive
controls rule that out: the same code scores AUROC 1.000 on this graph with
`shortest_path_length` left in, and ~1.0 on synthetic data with planted rings.

`python diagnose.py --source tigergraph` shows why. The 49 fraudulent Parties
are statistically indistinguishable from the other 4951 on every dimension a
GNN could use:

| Check | Observed | Expected by chance | Ratio |
|---|---|---|---|
| Fraud pairs sharing an IP | 2 | 1.2 | 1.7 |
| Fraud pairs sharing a Device | 0 | 1.2 | 0.0 |
| Fraud pairs sharing a DOB | 0 | 0.1 | 0.0 |
| Fraud → fraud transfers | 7 | 9.6 | 0.7 |
| Mean transfer amount | 499,107 | 498,938 (clean) | 1.0 |
| Account in/out degree | 19.8 / 20.0 | 20.0 / 20.0 (clean) | 1.0 |

The same diagnostic on synthetic data with real rings returns 65x, 318x and
180x. There is no mule-ring structure here to find.

Compounding it, **`Has_ID`, `Has_Phone`, `Has_Email` and `Has_Full_Name` are
strictly 1:1** with Party — 5000 vertices each, zero shared groups. Four of the
ten relations are identity mirrors carrying no information at all.

**The most likely cause is that the graph-feature pipeline never ran.**
`mule_cnt`, `fraud_ip`, `fraud_device`, `ip_collision`, `device_collision` and
`trans_in/out_mule_ratio` are all identically zero, while
`shortest_path_length` *is* populated — so some queries ran and others did not.
Re-run the mule-detection GSQL queries, or load a dataset where rings actually
exist, before drawing conclusions about model quality.

## The dataset also leaks, and it is not subtle

`Account.shortest_path_length` is **0 for all 49 fraudulent accounts and 1 or 2
for all 4951 others**. Rank AUC 1.000 on its own. It measures distance to the
nearest known fraud node, and a fraud node is distance 0 from itself — so it is
`is_fraud` wearing a different name, sitting one hop from the target across the
1:1 `Party_Has_Account` relation.

Left in, it produces a perfect score that means nothing. It is in
`schema.ALWAYS_DROP` alongside `Party.is_fraud` and `Account.is_fraud`.
`features.leakage_audit` re-derives this from the data on every run and flags
any attribute that behaves the same way:

```
leakage audit: these attributes are near-perfect label predictors ON THEIR OWN
  Account.shortest_path_length: rank AUC 1.000
```

Separately, **10 of Account's 15 attributes are constant** in this dataset —
including every remaining leakage suspect (`fraud_ip`, `fraud_device`,
`mule_cnt`, `trans_in/out_mule_ratio`, `ip_collision`, `device_collision`, all
zero). They are pruned automatically by variance, so `--exclude-derived` has
little left to remove here. It stays wired for when the data is populated.

| Vertex | Constant in this dataset |
|---|---|
| Party | `gender`, `party_type`, `created_at` (epoch-zero) |
| Account | `create_Time`, `account_type`, `account_level`, `ip_collision`, `fraud_ip`, `device_collision`, `fraud_device`, `trans_in_mule_ratio`, `trans_out_mule_ratio`, `mule_cnt` |
| Address_v2 | `city`, `population` |
| ID / IP / Device | `id_type`, `is_blocked` |

What is left to learn from: `Party.dob`, `Address_v2.state`/`zipcode`,
`Account.pagerank`/`com_size`, transaction amounts and times, and the graph
structure itself.

## Two ways to use transaction data

TigerGraph stores transactions as a directed `Transfer` edge (Account→Account)
carrying `amount` and `transfer_time`. **`HGTConv.forward` takes only
`(x_dict, edge_index_dict)`** — it accepts no edge attributes at all, so used
naively it discards all ~100k transaction amounts and timestamps.

Two routes, both implemented:

- **Reify** (default). Each Transfer edge becomes a `Transfer_Transaction`
  vertex with `Send_Transfer` / `Receive_Transfer` either side, lifting the
  attributes into node features. HGT stays HGT, parallel transfers between the
  same pair stay distinct. Costs ~100k extra vertices and needs `num_layers>=4`,
  since one counterparty step becomes Party→Account→Transfer→Account→Party.
- **`--conv transformer --no-reify`.** `HeteroConv` + `TransformerConv(edge_dim=…)`
  per relation reads the edge features directly. Shallower, but gives up HGT's
  type-aware parameterisation.

`test_transformer_actually_consumes_edge_features` perturbs the edge attributes
and asserts the output changes, so the second route cannot silently degrade
into ignoring them.

## Evaluation: read the spread, not the number

With 49 positives, a 70/15/15 split leaves **~7 in test**. A single AUROC on 7
positives is dominated by which 7 they are. `--split stratified` (the default)
at least guarantees each split gets a proportional share, but it cannot
manufacture statistical power. Treat any single-run number as anecdote.

**Multi-seed evaluation isn't currently wired up.** The original `sweep.py`
re-split across seeds and reported the spread for exactly this reason, but it
was retired when `train.py` was rebuilt around the multi-dataset `datasets.py`
registry (see git history / ask if you want it rebuilt as a flag on `train.py`).
In the meantime, `train.py --dataset synthetic --seed N` lets you do this by
hand across a few seeds.

## Schema handling worth knowing about

**Four vertex types carry no attributes** — Phone, Email, Full_Name, DOB.
`HGTConv` raises if a type it propagates over has no `x`, so
`features.structural_features` gives every type a per-relation log-degree
vector. For those four that *is* the signal, and it stays inductive, unlike a
per-node embedding table.

**Primary keys differ per vertex type** — `phone_number`, `email`, `dob`,
`address_key`, `name`, `id`. A GSQL query referencing `s.id` fails on most of
them, so edge pulls use `ListAccum<VERTEX>`, which yields external ids whatever
the key is called.

**Geography is denormalized** onto `Address_v2` (`zipcode`, `city`, `state`,
`population` as attributes) rather than City/State/Zipcode vertices. `state`
one-hots to 47 columns; `zipcode` (796 distinct) falls back to frequency
encoding above `MAX_ONEHOT_CARDINALITY`.

**TigerGraph's `reverse_Transfer` is not pulled.** `build_hetero_data` applies
`ToUndirected`, which generates a reverse for every relation; pulling
TigerGraph's own on top would double-count it.

**Fan-out is keyed off `data.edge_types`, not the schema.** Reification swaps
`Transfer` for `Send`/`Receive`, and `NeighborLoader` raises if any relation in
the graph lacks an entry. This graph's hubs are mild (~5 Parties per shared IP
or Device), so they are capped modestly — at that width the sharing *is* the
fraud signal.

**`--split temporal` does not work here** and raises rather than silently
producing a meaningless split: every `Party.created_at` is epoch-zero. The only
real timestamps in the graph are on `Transfer` edges.

## If this moves to a larger graph

`--devices` / `--num_nodes` / `--strategy ddp` solve compute parallelism, not
data parallelism for the graph itself. Past what fits on one machine, either
`torch_geometric.distributed` (METIS partitioning + `DistNeighborLoader`,
replacing only the datamodule) or AWS GraphStorm. `schema.py`, `features.py`,
`model.py` and the training logic carry over; the loader/partitioning layer
changes.

## Still worth deciding

- **Fix the data before touching the model.** Confirm the mule-detection GSQL
  queries have run — six Account attributes that should carry ring signal are
  identically zero. Until `diagnose.py` shows co-occurrence above chance, no
  architecture change will move the number.
- **Get more labels.** 49 positives leaves ~7 in test. Even with real signal,
  that cannot support a confident estimate. Semi-supervised or
  anomaly-detection framing may fit better than supervised classification.
- **Populate the dead attributes.** A third of the schema is declared but
  empty. The pruning is variance-based, so they start counting automatically.
- **Confirm how `shortest_path_length`, `com_id` and `com_size` were computed**,
  and over what time window, before ever re-enabling them.
- **Alert-volume metrics** — precision@k / recall@k at the volume the fraud team
  can action, rather than AUROC.
