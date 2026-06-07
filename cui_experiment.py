import os
import json
import re
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import pearsonr, spearmanr, t as student_t


##############################
# Utility Functions
##############################


def load_logs(path):
    """
    Load jsonl logs
    """
    tasks = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            tasks.append(data)
    return tasks


def task_correct(t):
    """
    Get correctness flag from our MARS logs.
    Prefer multi_score, then score, otherwise fall back to 0/1 heuristic.
    """
    if "multi_score" in t:
        return t["multi_score"]
    if "score" in t:
        return t["score"]
    # Fallback: accepted & has gt_answer → treat as correct
    return 1 if t.get("final_status") == "accepted" else 0


def accuracy(tasks):
    """
    Compute accuracy
    """
    if not tasks:
        return 0.0
    return float(np.mean([task_correct(t) for t in tasks]))


##############################
# CUI Components
##############################


def extract_review_votes_and_conf(task):
    """
    From one task record, extract:
      - votes: list of {0,1} per reviewer (1 = right, 0 = not-right)
      - conf: list of reliability scores in [0,1]

    This is derived from our peer-mode logs structure.
    """
    votes = []
    confs = []

    rounds = task.get("rounds", [])
    if not rounds:
        return votes, confs

    # Use first round (our experiments typically have 1 round)
    first_round = rounds[0]
    reviews = first_round.get("reviews", [])

    for rev in reviews:
        # In our logs each review is a dict like {"reviewer_1": "text ..."}
        if isinstance(rev, dict):
            text = " ".join(str(v) for v in rev.values())
        else:
            text = str(rev)

        lower = text.lower()

        # Decision: treat contains "decision: right" as vote = 1
        if "decision" in lower and "right" in lower:
            votes.append(1)
        else:
            votes.append(0)

        # Confidence: parse "Reliability Score: x / 1.0"
        m = re.search(r"reliability score[:\s]+([\d.]+)", text, re.IGNORECASE)
        if m:
            try:
                conf_val = float(m.group(1))
            except ValueError:
                conf_val = 0.5
        else:
            conf_val = 0.5
        confs.append(conf_val)

    return votes, confs


def belief_entropy(votes):
    """
    Reviewer belief entropy
    """
    if not votes:
        return 0.0

    p = float(np.mean(votes))

    if p == 0 or p == 1:
        return 0.0

    return float(-(p * np.log(p) + (1 - p) * np.log(1 - p)))


def confidence_variance(conf):
    """
    Reviewer confidence variance
    """
    if not conf:
        return 0.0
    return float(np.var(conf))


def risk_variance(risk_scores):
    """
    Risk instability across rounds.
    NOTE: our current logs do not store explicit per-round WM risk,
    so we approximate with a constant zero series → variance = 0.
    This keeps the structure compatible with the original CUI definition.
    """
    if not risk_scores:
        return 0.0
    return float(np.var(risk_scores))


def answer_entropy(answers):
    """
    Answer entropy Hans based on a list of answer strings.
    """
    if not answers:
        return 0.0

    counter = Counter(answers)
    counts = np.array(list(counter.values()), dtype=float)
    probs = counts / counts.sum()

    return float(-np.sum(probs * np.log(probs + 1e-12)))


def extract_answers(task):
    """
    Extract a multiset of answers for one task:
      - model final_answer
      - each reviewer Final Answer
      - meta_review Final Answer (if present)
    This approximates the answer-level uncertainty across agents.
    """
    answers = []

    # Author / model final answer
    fa = task.get("final_answer")
    if isinstance(fa, str) and fa.strip():
        answers.append(fa.strip())

    rounds = task.get("rounds", [])
    if rounds:
        first_round = rounds[0]
        reviews = first_round.get("reviews", [])
        for rev in reviews:
            if isinstance(rev, dict):
                text = " ".join(str(v) for v in rev.values())
            else:
                text = str(rev)

            # Try to extract a 'Final Answer:' line
            m = re.search(r"Final Answer:\s*(.+)", text, re.IGNORECASE)
            if m:
                ans = m.group(1).strip()
                if ans:
                    answers.append(ans)

        meta = first_round.get("meta_review") or task.get("meta_review")
        if isinstance(meta, str):
            m = re.search(r"Final Answer:\s*(.+)", meta, re.IGNORECASE)
            if m:
                ans = m.group(1).strip()
                if ans:
                    answers.append(ans)

    return answers


##############################
# Dataset-level CUI
##############################


def compute_dataset_metrics(tasks):
    H_rev = []
    V_conf = []
    V_risk = []
    H_ans = []

    for t in tasks:
        votes, confs = extract_review_votes_and_conf(t)

        H_rev.append(belief_entropy(votes))
        V_conf.append(confidence_variance(confs))

        # Placeholder: we don't have per-round risk scores in saved logs
        # Use a constant sequence so variance = 0
        risk_scores = [0.0 for _ in t.get("rounds", [])]
        V_risk.append(risk_variance(risk_scores))

        # Answer entropy across model/review answers
        answers = extract_answers(t)
        H_ans.append(answer_entropy(answers))

    if not H_rev:
        return 0.0, 0.0, 0.0, 0.0

    return (
        float(np.mean(H_rev)),
        float(np.mean(V_conf)),
        float(np.mean(V_risk)),
        float(np.mean(H_ans)),
    )


##############################
# Normalize helper
##############################


def normalize(values):
    values = np.array(values, dtype=float)
    if values.size == 0:
        return values

    vmin = values.min()
    vmax = values.max()
    return (values - vmin) / (vmax - vmin + 1e-8)


##############################
# Main Experiment
##############################


def get_record_config(base: str = "data") -> dict:
    """Peer / centralized jsonl paths per dataset name (shared with task-level scripts)."""
    return {
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
            "peer": os.path.join(
                base, "gsm_hard/records/peer_2026-03-02.jsonl"
            ),
            "central": os.path.join(
                base, "gsm_hard/records/centralized_2026-03-02.jsonl"
            ),
        },
        "mmlu_moral": {
            "peer": os.path.join(
                base, "mmlu/records/moral_scenarios_peer_2026-03-30.jsonl"
            ),
            "central": os.path.join(
                base, "mmlu/records/moral_scenarios_centralized_2026-03-30.jsonl"
            ),
        },
        "mmlu_law": {
            "peer": os.path.join(
                base, "mmlu/records/professional_law_peer_2026-03-30.jsonl"
            ),
            "central": os.path.join(
                base, "mmlu/records/professional_law_centralized_2026-03-30.jsonl"
            ),
        },
        "mmlu_elementary_math": {
            "peer": os.path.join(
                base,
                "mmlu/records/elementary_mathematics_peer_2026-03-30.jsonl",
            ),
            "central": os.path.join(
                base,
                "mmlu/records/elementary_mathematics_centralized_2026-03-30.jsonl",
            ),
        },
    }


def run_experiment():
    """
    Aggregate peer vs centralized logs per dataset; see ``get_record_config`` for paths.
    Rows with missing peer/central files are skipped.
    """

    config = get_record_config()

    results = []

    for dataset, paths in config.items():
        peer_path = paths["peer"]
        central_path = paths["central"]

        if not (os.path.exists(peer_path) and os.path.exists(central_path)):
            continue

        peer_tasks = load_logs(peer_path)
        central_tasks = load_logs(central_path)

        peer_acc = accuracy(peer_tasks)
        central_acc = accuracy(central_tasks)
        delta_perf = peer_acc - central_acc

        H, V1, V2, Hans = compute_dataset_metrics(peer_tasks)

        results.append(
            {
                "dataset": dataset,
                "peer_acc": peer_acc,
                "central_acc": central_acc,
                "delta_perf": delta_perf,
                "H_rev": H,
                "V_conf": V1,
                "V_risk": V2,
                "H_ans": Hans,
            }
        )

    df = pd.DataFrame(results)

    if df.empty:
        return df

    # Normalize components
    df["H_norm"] = normalize(df["H_rev"])
    df["Vconf_norm"] = normalize(df["V_conf"])
    df["Vrisk_norm"] = normalize(df["V_risk"])
    df["Hans_norm"] = normalize(df["H_ans"])

    # Final CUI with four equally weighted components:
    # reviewer belief entropy, confidence variance, risk variance, answer entropy.
    df["CUI"] = (
        0.25 * df["H_norm"]
        + 0.25 * df["Vconf_norm"]
        + 0.25 * df["Vrisk_norm"]
        + 0.25 * df["Hans_norm"]
    )

    return df


##############################
# Correlation statistics (small-n aware)
##############################


def permutation_pearson_p(x, y, n_perm=20000, seed=0):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)
    r_obs, _ = pearsonr(x, y)
    if len(x) < 3 or not np.isfinite(r_obs):
        return float("nan"), r_obs
    count = 0
    for _ in range(n_perm):
        yp = rng.permutation(y)
        rp, _ = pearsonr(x, yp)
        if not np.isfinite(rp):
            continue
        if abs(rp) >= abs(r_obs):
            count += 1
    p_perm = (count + 1) / (n_perm + 1)
    return p_perm, r_obs


def bootstrap_pearson_ci(x, y, n_boot=20000, alpha=0.05, seed=0):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    rng = np.random.default_rng(seed)
    rs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if np.std(x[idx]) < 1e-12 or np.std(y[idx]) < 1e-12:
            continue
        r, _ = pearsonr(x[idx], y[idx])
        if np.isfinite(r):
            rs.append(r)
    if len(rs) < 50:
        return float("nan"), float("nan")
    lo, hi = np.quantile(rs, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def pearson_one_sided_p_from_r(r, n):
    if n < 3 or not np.isfinite(r):
        return None
    df = n - 2
    if abs(r) >= 1 - 1e-15:
        return 0.0
    t_stat = r * np.sqrt(df / max(1e-15, 1.0 - r**2))
    if r >= 0:
        return float(student_t.sf(t_stat, df))
    return float(student_t.cdf(t_stat, df))


def print_correlation_summary(cui, delta, label="CUI vs Δperf"):
    x = np.asarray(cui, dtype=float)
    y = np.asarray(delta, dtype=float)
    n = len(x)
    r, p_pearson = pearsonr(x, y)
    rho, p_spear = spearmanr(x, y)
    p_perm, _ = permutation_pearson_p(x, y)
    lo, hi = bootstrap_pearson_ci(x, y)
    p_1 = pearson_one_sided_p_from_r(r, n)

    print(f"\n--- {label} (n={n}) ---")
    print(f"Pearson  r = {r:.4f},  two-sided p = {p_pearson:.4g}")
    print(f"Spearman ρ = {rho:.4f},  two-sided p = {p_spear:.4g}")
    print(f"Permutation (two-sided) p ≈ {p_perm:.4g}")
    if np.isfinite(lo):
        print(f"Bootstrap 95% CI for r: [{lo:.4f}, {hi:.4f}]")
    if p_1 is not None:
        print(f"Pearson one-sided p (sign-matched H1) = {p_1:.4g}")
    if n < 10:
        print(
            f"Note: with n={n}, correlation tests have very low power; "
            f"|r| must be large (often >0.8–0.9) for p<0.05 at two-sided α=0.05."
        )


##############################
# Plot
##############################


def plot_results(df):
    cui = df["CUI"].values
    delta = df["delta_perf"].values
    names = df["dataset"].values

    colors = []
    for d in delta:
        if d > 0:
            colors.append("#d62728")  # peer better
        elif d < 0:
            colors.append("#1f77b4")  # central better
        else:
            colors.append("#7f7f7f")  # tie

    plt.figure(figsize=(6.5, 5))
    plt.scatter(
        cui,
        delta,
        c=colors,
        s=80,
        edgecolor="black",
        linewidth=0.5,
        zorder=3,
    )

    for i, txt in enumerate(names):
        plt.annotate(txt, (cui[i] + 0.005, delta[i] + 0.005), fontsize=9)

    print_correlation_summary(cui, delta)

    plt.axhline(0, linestyle="--", color="gray", linewidth=1)

    tau = 0.30
    plt.axvline(tau, linestyle="--", color="black", linewidth=1)

    plt.fill_betweenx([-0.15, 0.2], 0, tau, color="#1f77b4", alpha=0.08)
    plt.fill_betweenx([-0.15, 0.2], tau, 0.7, color="#d62728", alpha=0.08)

    # Region labels: keep left of tau / right of tau (avoid legend overlap with upper left)
    plt.text(0.12, 0.12, "Centralized\npreferred", fontsize=10, color="#1f77b4")
    plt.text(0.42, 0.12, "Peer\npreferred", fontsize=10, color="#d62728")

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Peer better",
            markerfacecolor="#d62728",
            markersize=8,
            markeredgecolor="black",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Central better",
            markerfacecolor="#1f77b4",
            markersize=8,
            markeredgecolor="black",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Tie",
            markerfacecolor="#7f7f7f",
            markersize=8,
            markeredgecolor="black",
        ),
    ]
    plt.legend(
        handles=legend_elements,
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
    )

    plt.xlabel("Coordination Uncertainty Index (CUI)", fontsize=11)
    plt.ylabel("Performance Gap (Peer - Centralized)", fontsize=11)
    plt.title("CUI Determines Coordination Regime", fontsize=12)

    plt.xlim(0.05, 0.7)
    plt.ylim(-0.12, 0.2)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("cui_vs_performance.png", dpi=300)
    plt.show()


##############################
# Run
##############################


if __name__ == "__main__":
    df = run_experiment()

    print("\n=== Experiment Results ===\n")
    if df.empty:
        print("No results found. Please check that the record files exist.")
    else:
        print(f"Datasets in table: {len(df)}\n")
        print(
            df[
                [
                    "dataset",
                    "peer_acc",
                    "central_acc",
                    "delta_perf",
                    "CUI",
                ]
            ]
        )

        plot_results(df)

        df.to_csv("cui_results.csv", index=False)

        print("\nSaved results to cui_results.csv")

