"""
Task-level CUI vs (Peer correct − Central correct), then logistic regression.

Contrasts with ``cui_experiment.py`` (one row per dataset, dataset-mean components).
Here each row is one aligned (peer, central) task; CUI components are per-task,
then min–max normalized across *all tasks* in the pool (same 4-term formula).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict

import cui_experiment as ce


def compute_task_level_rows(
    peer_tasks: list,
    central_tasks: list,
    dataset_name: str,
) -> list[dict]:
    rows = []
    if len(peer_tasks) != len(central_tasks):
        raise ValueError(
            f"{dataset_name}: peer len {len(peer_tasks)} != central len {len(central_tasks)}"
        )

    for i, (p, c) in enumerate(zip(peer_tasks, central_tasks)):
        peer_correct = ce.task_correct(p)
        central_correct = ce.task_correct(c)
        delta = int(peer_correct) - int(central_correct)

        votes, confs = ce.extract_review_votes_and_conf(p)
        H = ce.belief_entropy(votes)
        Vconf = ce.confidence_variance(confs)
        risk_scores = [0.0 for _ in p.get("rounds", [])]
        Vrisk = ce.risk_variance(risk_scores)

        answers = ce.extract_answers(p)
        Hans = ce.answer_entropy(answers)

        pc = int(peer_correct)
        cc = int(central_correct)
        rows.append(
            {
                "dataset": dataset_name,
                "task_idx": i,
                "task_id": i,
                "peer_correct": pc,
                "central_correct": cc,
                "delta_perf": delta,
                "H_rev": H,
                "V_conf": Vconf,
                "V_risk": Vrisk,
                "H_ans": Hans,
            }
        )

    return rows


def build_task_dataframe(base: str = "data") -> pd.DataFrame:
    all_rows: list[dict] = []
    config = ce.get_record_config(base)

    for dataset, paths in config.items():
        peer_path = paths["peer"]
        central_path = paths["central"]
        if not (os.path.exists(peer_path) and os.path.exists(central_path)):
            continue
        peer_tasks = ce.load_logs(peer_path)
        central_tasks = ce.load_logs(central_path)
        rows = compute_task_level_rows(peer_tasks, central_tasks, dataset)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    if df.empty:
        return df

    df["H_norm"] = ce.normalize(df["H_rev"].values)
    df["Vconf_norm"] = ce.normalize(df["V_conf"].values)
    df["Vrisk_norm"] = ce.normalize(df["V_risk"].values)
    df["Hans_norm"] = ce.normalize(df["H_ans"].values)
    df["CUI"] = (
        0.25 * df["H_norm"]
        + 0.25 * df["Vconf_norm"]
        + 0.25 * df["Vrisk_norm"]
        + 0.25 * df["Hans_norm"]
    )
    return df


def run_task_logistic(df: pd.DataFrame) -> dict | None:
    """
    Filter delta_perf != 0, label = 1 iff Peer correct and Central wrong.
    Fit logistic regression on CUI; report full-sample metrics and GroupKFold CV
    (by dataset) when possible.
    """
    d = df[df["delta_perf"] != 0].copy()
    if d.shape[0] < 2:
        print("Task logistic: need >= 2 rows with non-zero delta_perf.")
        return None

    d["label"] = (d["delta_perf"] > 0).astype(int)
    X = d[["CUI"]].values.astype(float)
    y = d["label"].values.astype(int)
    groups = d["dataset"].values

    if np.unique(y).size < 2:
        print("Task logistic: only one class in label after filtering.")
        return None

    model = LogisticRegression(max_iter=5000, solver="lbfgs")
    model.fit(X, y)
    prob_in = model.predict_proba(X)[:, 1]
    pred_in = model.predict(X)
    acc_in = float(accuracy_score(y, pred_in))
    auc_in = float(roc_auc_score(y, prob_in))
    coef = float(model.coef_.ravel()[0])
    intercept = float(model.intercept_[0])

    print("\n=== Task-level logistic (CUI → P(Peer wins | disagreement)) ===")
    print(f"Rows (|delta_perf|>0): {len(d)}  |  label=1: Peer right & Central wrong")
    print(f"Coefficient (CUI): {coef:+.6f}")
    print(f"Intercept: {intercept:+.6f}")
    print(f"In-sample Accuracy: {acc_in:.4f}")
    print(f"In-sample AUC:      {auc_in:.4f}")

    n_groups = len(np.unique(groups))
    if n_groups >= 2:
        n_splits = min(5, n_groups)
        gkf = GroupKFold(n_splits=n_splits)
        try:
            prob_cv = cross_val_predict(
                LogisticRegression(max_iter=5000, solver="lbfgs"),
                X,
                y,
                cv=gkf,
                groups=groups,
                method="predict_proba",
            )[:, 1]
            pred_cv = (prob_cv >= 0.5).astype(int)
            acc_cv = float(accuracy_score(y, pred_cv))
            auc_cv = float(roc_auc_score(y, prob_cv))
            print(
                f"GroupKFold CV (n_splits={n_splits}, by dataset) Accuracy: {acc_cv:.4f}"
            )
            print(f"GroupKFold CV AUC:      {auc_cv:.4f}")
        except ValueError as e:
            print(f"GroupKFold CV skipped: {e}")
    else:
        print("GroupKFold CV skipped: need tasks from >=2 datasets.")

    return {
        "n_rows": len(d),
        "coef": coef,
        "intercept": intercept,
        "acc_in_sample": acc_in,
        "auc_in_sample": auc_in,
        "model": model,
        "df_fit": d,
    }


def main():
    df = build_task_dataframe()
    if df.empty:
        print("No task rows; check record paths.")
        return

    out_path = "cui_task_level_full.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved all task rows with CUI to {out_path} (n={len(df)})")

    res = run_task_logistic(df)
    if res is not None:
        fit_path = "cui_task_level_logistic_fit.csv"
        res["df_fit"].assign(
            prob_positive=res["model"].predict_proba(
                res["df_fit"][["CUI"]].values.astype(float)
            )[:, 1]
        ).to_csv(fit_path, index=False)
        print(f"Saved rows used for logistic to {fit_path}")


if __name__ == "__main__":
    main()
