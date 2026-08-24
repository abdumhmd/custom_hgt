# custom_hgt

Two heterogeneous GNN architectures, trained and compared across four graph
datasets.

## The two architectures (`model.py`)

- **`hgt`** — PyG's `HGTConv`. Type-aware attention per relation. Cannot read
  edge attributes (e.g. a transaction's amount/time).
- **`transformer`** (`HeteroEdgeGNN`) — `HeteroConv` + `TransformerConv`.
  Reads numeric edge attributes directly, at the cost of HGT's type-specific
  parameterization.

Select with `--conv hgt` or `--conv transformer` on any script below.

## The four datasets (`datasets.py`)

| `--dataset` | What it is | Setup |
|---|---|---|
| `synthetic` | Generated fraud graph on the TigerGraph schema | none |
| `tigergraph` | Live pull from a TigerGraph `Mule_Account_Detection` instance | needs a running instance |
| `ieee_fraud` | Kaggle IEEE-CIS transactions, built into a heterogeneous graph with real edge attributes | needs a Kaggle download |
| `ogbn_mag` | OGBN-MAG citation graph, 349-way paper classification | auto-downloads |

## Install

```bash
pip install torch torch_geometric pytorch_lightning torchmetrics pandas pyTigerGraph optuna ogb kaggle
pip install pyg-lib -f https://data.pyg.org/whl/torch-<ver>+cpu.html   # for NeighborLoader
```

## Run

```bash
python test_schema.py                                       # 18 tests, no data needed

python train.py --dataset synthetic                         # train + eval
python train.py --dataset ieee_fraud --conv transformer      # edge-feature route
python hparam_search.py --dataset ieee_fraud --trials 30     # Optuna search + final retrain
python eval.py --dataset ieee_fraud --ckpt <path-to-.ckpt>   # standalone eval
```

Run any script with `--help` for the full flag list (hidden size, heads, layers,
sampling strategy, etc). `train.py` and `hparam_search.py` both evaluate the
best checkpoint on the test split automatically when they finish.

## Getting the data

- **`synthetic`, `ogbn_mag`** — nothing to do; both fetch/generate on first use.
- **`tigergraph`** — set connection env vars before running:
  ```bash
  TG_HOST=http://10.0.0.76        # no port; ports are separate vars
  TG_GRAPH=Mule_Account_Detection
  TG_USERNAME=tigergraph
  TG_PASSWORD=tigergraph
  ```
  Run `python preflight.py` first to sanity-check the connection and schema.
- **`ieee_fraud`** — gated Kaggle competition, one-time manual pull:
  ```bash
  # 1. accept the rules: https://www.kaggle.com/competitions/ieee-fraud-detection/rules
  # 2. create a token: https://www.kaggle.com/settings -> API -> Create New Token -> ~/.kaggle/kaggle.json
  mkdir -p data/ieee_fraud
  kaggle competitions download -c ieee-fraud-detection -f train_transaction.csv -p data/ieee_fraud
  kaggle competitions download -c ieee-fraud-detection -f train_identity.csv -p data/ieee_fraud
  cd data/ieee_fraud && unzip -o train_transaction.csv.zip && unzip -o train_identity.csv.zip && cd -
  ```
  `train.py --dataset ieee_fraud` builds and caches the graph automatically
  the first time it's needed.

## Files

| File | Purpose |
|---|---|
| `model.py` | The two architectures. |
| `datasets.py` | Dataset registry — add a dataset here, nothing downstream changes. |
| `lightning_module.py` | Training/eval logic (loss, metrics, data loading) shared by every dataset. |
| `train.py` | Train + evaluate one (dataset, architecture) pair. |
| `hparam_search.py` | Optuna hyperparameter search, then a final retrain + eval. |
| `eval.py` | Standalone: evaluate a saved checkpoint. |
| `schema.py`, `features.py`, `data.py`, `tg_loader.py` | TigerGraph/synthetic schema, feature encoding, and data loading. |
| `build_ieee_fraud_graph.py` | Builds the IEEE-CIS heterogeneous graph from the raw CSVs. |
| `diagnose.py` | Checks whether a graph has any learnable fraud signal, before blaming the model. |
| `preflight.py` | Read-only connection/schema check against a live TigerGraph instance. |
| `test_schema.py` | 18 conformance tests, no data needed. |

## More detail

[`NOTES.md`](NOTES.md) has the deep-dive findings behind the TigerGraph/synthetic
side specifically: why the live graph currently has no learnable fraud signal,
the label-leakage story, and various schema quirks worth knowing before you
change anything there.
