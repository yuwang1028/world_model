import os
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression

from cui_experiment import (
    load_logs,
    task_correct,
    extract_review_votes_and_conf,
    belief_entropy,
    confidence_variance,
    risk_variance,
    normalize,
)


def task_level_components(task):
    """
    Compute per-task CUI components using the same definitions
    as in cui_experiment.py (belief entropy, confidence variance, risk variance).
    """
    votes, confs = extract_review_votes_and_conf(task)
    h = belief_entropy(votes)
    v_conf = confidence_variance(confs)

    # We do not have explicit per-round risk scores in logs; follow cui_experiment
    # and use a constant sequence over rounds so variance = 0.
    risk_scores = [0.0 for _ in task.get("rounds", [])]
    v_risk = risk_variance(risk_scores)
    return h, v_conf, v_risk


def build_task_level_dataframe():
    """
    Build a task-level dataframe from the same 5 datasets/config
    used in cui_experiment.run_experiment(), but at per-task granularity.
    """
    base = "data"

    config = {
        "gsm": {
            "peer": os.path.join(base, "gsm/records/peer_2026-02-25.jsonl"),
            "central": os.path.join(base, "gsm/records/centralized_2026-02-25.jsonl"),
        },
        "svamp": {
            "peer": os.path.join(base, "svamp/records/peer_2026-02-26.jsonl"),
            "central": os.path.join(
                base, "svamp/records/centralized_2026-02-26.jsonl"
            ),
        },
        "ciar": {
            "peer": os.path.join(base, "ciar/records/peer_2026-02-25.jsonl"),
            "central": os.path.join(
                base, "ciar/records/centralized_2026-02-25.jsonl"
            ),
        },
        "stg": {
            "peer": os.path.join(base, "stg/records/peer_2026-03-02.jsonl"),
            "central": os.path.join(
                base, "stg/records/centralized_2026-03-02.jsonl"
            ),
        },
        "gsm_hard": {
            "peer": os.path.join(base, "gsm_hard/records/peer_2026-03-02.jsonl"),
            "central": os.path.join(
                base, "gsm_hard/records/centralized_2026-03-02.jsonl"
            ),
        },
    }

    rows = []

    for ds, paths in config.items():
        peer_tasks = load_logs(paths["peer"])
        central_tasks = load_logs(paths["central"])

        # Use shared "id" field as task identifier
        peer_map = {t["id"]: t for t in peer_tasks}
        central_map = {t["id"]: t for t in central_tasks}

        common_ids = sorted(set(peer_map.keys()) & set(central_map.keys()))

        for tid in common_ids:
            p = peer_map[tid]
            c = central_map[tid]

            ph, pvconf, pvrisk = task_level_components(p)
            ch, cvconf, cvrisk = task_level_components(c)

            # average reviewer confidence for peer and central
            _, peer_confs = extract_review_votes_and_conf(p)
            _, central_confs = extract_review_votes_and_conf(c)

            peer_score = float(np.mean(peer_confs)) if peer_confs else 0.0
            central_score = float(np.mean(central_confs)) if central_confs else 0.0
            y = peer_score - central_score

            rows.append(
                {
                    "dataset": ds,
                    "task_id": tid,
                    "peer_H": ph,
                    "peer_Vconf": pvconf,
                    "peer_Vrisk": pvrisk,
                    "central_H": ch,
                    "central_Vconf": cvconf,
                    "central_Vrisk": cvrisk,
                    "y": y,
                }
            )

    df = pd.DataFrame(rows)
    return df


def add_task_cui(df):
    """
    Add normalized components and task-level CUI using the
    same weighting scheme as cui_experiment.py (0.5/0.3/0.2).
    """
    for col in [
        "peer_H",
        "peer_Vconf",
        "peer_Vrisk",
        "central_H",
        "central_Vconf",
        "central_Vrisk",
    ]:
        df[col + "_norm"] = normalize(df[col].values)

    df["peer_CUI"] = (
        0.5 * df["peer_H_norm"]
        + 0.3 * df["peer_Vconf_norm"]
        + 0.2 * df["peer_Vrisk_norm"]
    )

    df["central_CUI"] = (
        0.5 * df["central_H_norm"]
        + 0.3 * df["central_Vconf_norm"]
        + 0.2 * df["central_Vrisk_norm"]
    )

    df["task_CUI"] = (df["peer_CUI"] + df["central_CUI"]) / 2.0
    return df


def run_linear_regression(df):
    clean_df = df.dropna(subset=["y"]).copy()

    X = clean_df[["task_CUI"]].values
    y = clean_df["y"].values

    model = LinearRegression()
    model.fit(X, y)

    coef = model.coef_[0]
    intercept = model.intercept_
    r2 = model.score(X, y)

    print("Samples:", len(clean_df))
    print("Coefficient:", coef)
    print("Intercept:", intercept)
    print("R^2:", r2)

    return model, clean_df


if __name__ == "__main__":
    df = build_task_level_dataframe()
    df = add_task_cui(df)
    model, clean_df = run_linear_regression(df)
    clean_df.to_csv("task_level_cui_results.csv", index=False)

