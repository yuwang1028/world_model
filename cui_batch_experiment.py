"""
Batch-level aggregation over ``cui_task_level_full.csv`` (from ``cui_task_experiment``).

Each batch: mean CUI, mean peer/central correctness, delta = peer_acc - central_acc.
Then linear regression on continuous delta_perf and logistic on label_peer_better.
"""

from __future__ import annotations

import shutil

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold


def build_batch_level_df(
    csv_path: str = "cui_task_level_full.csv",
    batch_size: int = 10,
    shuffle_within_dataset: bool = False,
    random_state: int = 42,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required_cols = ["dataset", "CUI", "peer_correct", "central_correct"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(
                f"Missing required column: {c}. "
                "Re-run cui_task_experiment.py to regenerate cui_task_level_full.csv."
            )

    all_batches: list[dict] = []

    for ds, sub in df.groupby("dataset"):
        sub = sub.copy()

        if shuffle_within_dataset:
            sub = sub.sample(frac=1, random_state=random_state).reset_index(
                drop=True
            )
        else:
            if "task_id" in sub.columns:
                sub = sub.sort_values("task_id").reset_index(drop=True)
            elif "task_idx" in sub.columns:
                sub = sub.sort_values("task_idx").reset_index(drop=True)
            else:
                sub = sub.reset_index(drop=True)

        n = len(sub)
        n_batches = int(np.ceil(n / batch_size))

        for b in range(n_batches):
            chunk = sub.iloc[b * batch_size : (b + 1) * batch_size]
            if len(chunk) == 0:
                continue

            batch_cui = chunk["CUI"].mean()
            peer_acc = chunk["peer_correct"].mean()
            central_acc = chunk["central_correct"].mean()
            delta_perf = peer_acc - central_acc

            all_batches.append(
                {
                    "dataset": ds,
                    "batch_id": f"{ds}_batch_{b}",
                    "batch_size": len(chunk),
                    "batch_CUI": batch_cui,
                    "peer_acc": peer_acc,
                    "central_acc": central_acc,
                    "delta_perf": delta_perf,
                    "label_peer_better": int(delta_perf > 0),
                }
            )

    return pd.DataFrame(all_batches)


def batch_level_regression(
    batch_df: pd.DataFrame, *, verbose: bool = True
) -> tuple[LinearRegression, dict]:
    X = batch_df[["batch_CUI"]].values
    y = batch_df["delta_perf"].values

    reg = LinearRegression()
    reg.fit(X, y)

    r2 = reg.score(X, y)
    coef = float(reg.coef_[0])
    intercept = float(reg.intercept_)

    stats = {
        "n_batches": len(batch_df),
        "lin_coef": coef,
        "lin_intercept": intercept,
        "lin_r2": float(r2),
    }

    if verbose:
        print("=== Batch-level Linear Regression ===")
        print(f"Samples: {len(batch_df)}")
        print(f"Coefficient: {coef:.6f}")
        print(f"Intercept: {intercept:.6f}")
        print(f"R²: {r2:.4f}")

    return reg, stats


def batch_level_logistic(
    batch_df: pd.DataFrame, *, verbose: bool = True
) -> tuple[LogisticRegression | None, dict]:
    X = batch_df[["batch_CUI"]].values
    y = batch_df["label_peer_better"].values

    if len(np.unique(y)) < 2:
        if verbose:
            print("Only one class present in labels; cannot run logistic regression.")
        return None, {"log_acc_in": np.nan, "log_auc_in": np.nan}

    clf = LogisticRegression(max_iter=5000, solver="lbfgs")
    clf.fit(X, y)

    probs = clf.predict_proba(X)[:, 1]
    preds = clf.predict(X)

    acc = float(accuracy_score(y, preds))
    auc = float(roc_auc_score(y, probs))
    log_coef = float(clf.coef_[0][0])
    log_intercept = float(clf.intercept_[0])

    stats = {
        "n_batches": len(batch_df),
        "log_coef": log_coef,
        "log_intercept": log_intercept,
        "log_acc_in": acc,
        "log_auc_in": auc,
    }

    if verbose:
        print("\n=== Batch-level Logistic Regression ===")
        print(f"Samples: {len(batch_df)}")
        print(f"Coefficient: {log_coef:.6f}")
        print(f"Intercept: {log_intercept:.6f}")
        print(f"In-sample Accuracy: {acc:.4f}")
        print(f"In-sample AUC:      {auc:.4f}")

    return clf, stats


def batch_level_group_cv(
    batch_df: pd.DataFrame, n_splits: int = 5, *, verbose: bool = True
) -> dict:
    X = batch_df[["batch_CUI"]].values
    y = batch_df["label_peer_better"].values
    groups = batch_df["dataset"].values

    out: dict = {"cv_acc_mean": np.nan, "cv_auc_mean": np.nan}

    if len(np.unique(y)) < 2:
        if verbose:
            print(
                "Only one class present in labels; cannot run CV logistic regression."
            )
        return out

    n_groups = batch_df["dataset"].nunique()
    gkf = GroupKFold(n_splits=min(n_splits, n_groups))

    accs: list[float] = []
    aucs: list[float] = []

    for train_idx, test_idx in gkf.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if len(np.unique(y_train)) < 2:
            continue

        clf = LogisticRegression(max_iter=5000, solver="lbfgs")
        clf.fit(X_train, y_train)

        probs = clf.predict_proba(X_test)[:, 1]
        preds = clf.predict(X_test)

        accs.append(accuracy_score(y_test, preds))

        if len(np.unique(y_test)) > 1:
            aucs.append(roc_auc_score(y_test, probs))

    if accs:
        out["cv_acc_mean"] = float(np.mean(accs))
    if aucs:
        out["cv_auc_mean"] = float(np.mean(aucs))

    if verbose:
        print("\n=== Batch-level GroupKFold CV (by dataset) ===")
        if accs:
            print(f"CV Accuracy: {out['cv_acc_mean']:.4f}")
        else:
            print("CV Accuracy: N/A")

        if aucs:
            print(f"CV AUC:      {out['cv_auc_mean']:.4f}")
        else:
            print("CV AUC:      N/A")

    return out


def run_one_batch_size(
    batch_size: int,
    *,
    csv_path: str = "cui_task_level_full.csv",
    shuffle_within_dataset: bool = False,
    random_state: int = 42,
    n_splits: int = 5,
    verbose: bool = True,
) -> dict:
    """Build batches, save CSV, run linear + logistic + group CV; return flat metrics."""
    batch_df = build_batch_level_df(
        csv_path=csv_path,
        batch_size=batch_size,
        shuffle_within_dataset=shuffle_within_dataset,
        random_state=random_state,
    )

    out_csv = f"cui_batch_level_bs{batch_size}.csv"
    batch_df.to_csv(out_csv, index=False)
    if verbose:
        print(f"\n{'='*60}")
        print(f"batch_size (target) = {batch_size}")
        print(f"Saved: {out_csv}  (n_batches={len(batch_df)})")
        print(batch_df.head(3).to_string())
        print()

    _, lin_stats = batch_level_regression(batch_df, verbose=verbose)
    _, log_stats = batch_level_logistic(batch_df, verbose=verbose)
    cv_stats = batch_level_group_cv(batch_df, n_splits=n_splits, verbose=verbose)

    row = {
        "batch_size_target": batch_size,
        **lin_stats,
        **{k: v for k, v in log_stats.items() if k != "n_batches"},
        **cv_stats,
    }
    return row


def main():
    csv_path = "cui_task_level_full.csv"
    batch_sizes = (5, 10, 20)

    summary_rows: list[dict] = []
    for bs in batch_sizes:
        summary_rows.append(
            run_one_batch_size(
                bs,
                csv_path=csv_path,
                shuffle_within_dataset=False,
                random_state=42,
                n_splits=5,
                verbose=True,
            )
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = "cui_batch_level_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n{'='*60}")
    print(f"Summary (all batch sizes): saved {summary_path}")
    print(summary_df.to_string(index=False))

    # Backward compatibility: mirror size=10 as cui_batch_level.csv
    default_bs = 10
    if default_bs in batch_sizes:
        shutil.copy2(
            f"cui_batch_level_bs{default_bs}.csv", "cui_batch_level.csv"
        )
        print(
            f"\nAlso copied cui_batch_level_bs{default_bs}.csv -> cui_batch_level.csv"
        )


if __name__ == "__main__":
    main()
