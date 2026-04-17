"""Hyperparameter sweep for tuned Elliptic training pipeline."""

import os
import sys

import pandas as pd

# Allow running as `python src/hparam_sweep.py`.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.bitcoin_train_tuned import run_fix_sequence


def main():
    configs = [
        # hidden, dropout, lr, gamma, threshold
        (128, 0.3, 0.0010, 2.0, 0.35),
        (128, 0.4, 0.0010, 2.0, 0.30),
        (256, 0.3, 0.0010, 2.0, 0.35),
        (128, 0.3, 0.0005, 2.0, 0.35),
        (128, 0.3, 0.0010, 3.0, 0.35),
        (256, 0.4, 0.0010, 2.5, 0.30),
    ]

    rows = []
    for hidden, dropout, lr, gamma, threshold in configs:
        out = run_fix_sequence(
            hidden=hidden,
            dropout=dropout,
            lr=lr,
            gamma=gamma,
            threshold=threshold,
            epochs=200,
            patience=30,
        )

        # Use last row (Fix 4 threshold tuned) for ranking.
        final = out["stage_metrics"].iloc[-1].to_dict()
        final.update(
            {
                "hidden": hidden,
                "dropout": dropout,
                "lr": lr,
                "gamma": gamma,
                "threshold_init": threshold,
                "threshold_best": out["best_threshold"],
            }
        )
        rows.append(final)

    df = pd.DataFrame(rows).sort_values("f1", ascending=False)
    df.to_csv("outputs/results/hparam_sweep.csv", index=False)
    print(df[["hidden", "dropout", "lr", "gamma", "threshold_best", "f1", "roc_auc", "pr_auc"]].to_string(index=False))


if __name__ == "__main__":
    main()
