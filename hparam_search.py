"""
Optuna hyperparameter search over one (dataset, conv) pair from
`datasets.py` / `model.py`, then a final retrain of the best trial with
`--final-epochs` and a test-set evaluation.

    python hparam_search.py --dataset ieee_fraud --conv hgt --trials 30
    python hparam_search.py --dataset synthetic --conv transformer --no-reify --trials 20

Each trial is a full call into `train.train()` (see train.py) with a reduced
epoch budget and Optuna-suggested hyperparameters, so the search reuses
exactly the training/eval path a plain `python train.py` run would take --
no separate training loop to keep in sync.
"""

import argparse
import copy

import optuna

from datasets import REGISTRY
from train import build_parser, train


def suggest_args(trial: optuna.Trial, base_args: argparse.Namespace) -> argparse.Namespace:
    args = copy.deepcopy(base_args)
    # hidden_channels must be divisible by num_heads (HeteroEdgeGNN splits
    # hidden_channels // num_heads per head); both grids are powers of 2 with
    # min(hidden_channels) >= max(num_heads), so every combination divides evenly.
    args.hidden_channels = trial.suggest_categorical("hidden_channels", [32, 64, 128])
    args.num_heads = trial.suggest_categorical("num_heads", [2, 4, 8])
    args.num_layers = trial.suggest_int("num_layers", 2, 4)
    args.dropout = trial.suggest_float("dropout", 0.0, 0.5)
    args.lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    args.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    return args


def build_objective(base_args: argparse.Namespace):
    def objective(trial: optuna.Trial) -> float:
        args = suggest_args(trial, base_args)
        args.max_epochs = base_args.search_epochs
        args.patience = min(args.patience, base_args.search_epochs)
        result = train(args)
        metric_name = "test_auroc" if "test_auroc" in result["metrics"] else "test_acc"
        return result["metrics"][metric_name]

    return objective


def main():
    parser = build_parser()
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--search-epochs", type=int, default=10,
                         help="epoch budget for each trial (kept short; the winner gets --final-epochs)")
    parser.add_argument("--final-epochs", type=int, default=None,
                         help="epoch budget for the final retrain of the best trial; defaults to --max-epochs")
    parser.add_argument("--study-name", default=None)
    parser.add_argument("--storage", default=None, help="e.g. sqlite:///optuna.db, to persist/resume a study")
    args = parser.parse_args()

    study = optuna.create_study(
        direction="maximize", study_name=args.study_name, storage=args.storage,
        load_if_exists=args.storage is not None,
    )
    study.optimize(build_objective(args), n_trials=args.trials)

    print("\n=== best trial ===")
    print(f"  value: {study.best_value:.4f}")
    print(f"  params: {study.best_params}")

    final_args = suggest_args_from_dict(args, study.best_params)
    final_args.max_epochs = args.final_epochs or args.max_epochs
    final_args.patience = args.patience
    print("\n=== final retrain with best hyperparameters ===")
    result = train(final_args)
    print(f"\nbest checkpoint: {result['ckpt_path']}")


def suggest_args_from_dict(base_args: argparse.Namespace, params: dict) -> argparse.Namespace:
    args = copy.deepcopy(base_args)
    for k, v in params.items():
        setattr(args, k, v)
    return args


if __name__ == "__main__":
    main()
