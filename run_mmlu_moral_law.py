"""
MMLU: moral_scenarios and professional_law — three-mode (baseline / peer / centralized) test.

CSV paths (via utils.load_data):
  data/mmlu/test/moral_scenarios_test.csv
  data/mmlu/test/professional_law_test.csv
"""

from __future__ import annotations

import time
from typing import Any

from evaluation import eval_mars


def run_mmlu_subject_three_modes(
    subject: str,
    label: str | None = None,
    n_problems: int = 50,
    n_reviewers: int = 2,
    model=None,
    verbosity: int = 0,
) -> dict[str, Any]:
    """
    Run baseline, peer, and centralized on one MMLU subject.

    Parameters
    ----------
    subject
        Filename stem under data/mmlu/test/, e.g. "moral_scenarios", "professional_law".
    label
        Human-readable label for logs; defaults to subject.
    n_problems, n_reviewers
        Passed to eval_mars (subject-specific question_ids are sampled when mmlu_subject is set).

    Returns
    -------
    dict with keys: subject, label, n_problems, baseline, peer, centralized,
    delta_peer_minus_central (peer_score - centralized_score), delta_peer_minus_baseline.
    """
    if label is None:
        label = subject

    task = "mmlu"
    print("=" * 80)
    print(f"Running Three Mode Comparison - MMLU ({label}: {subject})")
    print("=" * 80)
    print(f"Task: {task}")
    print(f"Subject: {subject}")
    print(f"Number of Problems: {n_problems}")
    print(f"Number of Reviewers: {n_reviewers}")
    print("=" * 80)
    print()

    results_modes: dict[str, dict[str, float | int]] = {}

    # [1/3] Baseline
    print("[1/3] Running Baseline (Original MARS)...")
    print("-" * 80)
    t0 = time.time()
    b_score, _, _, b_tokens, _ = eval_mars(
        task=task,
        model=model,
        n_problems=n_problems,
        n_reviewers=n_reviewers,
        selected=True,
        mode="baseline",
        verbosity=verbosity,
        mmlu_subject=subject,
    )
    results_modes["baseline"] = {
        "score": int(b_score),
        "tokens_avg": float(b_tokens),
        "time_s": time.time() - t0,
    }
    print(
        f"✅ Baseline: {b_score}/{n_problems} ({b_score / n_problems * 100:.1f}%)"
    )
    print()

    # [2/3] Peer
    print("[2/3] Running Peer Mode (Improved World Model)...")
    print("-" * 80)
    t0 = time.time()
    p_score, _, _, p_tokens, _ = eval_mars(
        task=task,
        model=model,
        n_problems=n_problems,
        n_reviewers=n_reviewers,
        selected=True,
        mode="peer",
        verbosity=verbosity,
        mmlu_subject=subject,
    )
    results_modes["peer"] = {
        "score": int(p_score),
        "tokens_avg": float(p_tokens),
        "time_s": time.time() - t0,
    }
    print(f"✅ Peer: {p_score}/{n_problems} ({p_score / n_problems * 100:.1f}%)")
    print()

    # [3/3] Centralized
    print("[3/3] Running Centralized Mode (Improved World Model)...")
    print("-" * 80)
    t0 = time.time()
    c_score, _, _, c_tokens, _ = eval_mars(
        task=task,
        model=model,
        n_problems=n_problems,
        n_reviewers=n_reviewers,
        selected=True,
        mode="centralized",
        verbosity=verbosity,
        mmlu_subject=subject,
    )
    results_modes["centralized"] = {
        "score": int(c_score),
        "tokens_avg": float(c_tokens),
        "time_s": time.time() - t0,
    }
    print(
        f"✅ Centralized: {c_score}/{n_problems} ({c_score / n_problems * 100:.1f}%)"
    )
    print()

    # Summary table
    print("=" * 80)
    print(f"Results Comparison - {label} ({subject})")
    print("=" * 80)
    print(
        f"{'Mode':<20} {'Accuracy':<25} {'Total Tokens (≈)':<22} {'Avg Tokens':<14} {'Time (s)':<12}"
    )
    print("-" * 80)
    b_tot = int(results_modes["baseline"]["tokens_avg"] * n_problems)
    p_tot = int(results_modes["peer"]["tokens_avg"] * n_problems)
    c_tot = int(results_modes["centralized"]["tokens_avg"] * n_problems)
    b_acc = f"{b_score}/{n_problems} ({b_score / n_problems * 100:.1f}%)"
    p_acc = f"{p_score}/{n_problems} ({p_score / n_problems * 100:.1f}%)"
    c_acc = f"{c_score}/{n_problems} ({c_score / n_problems * 100:.1f}%)"
    pdiff = p_score - b_score
    p_acc_s = p_acc + (f" ({pdiff:+d} vs baseline)" if pdiff != 0 else "")
    cdiff = c_score - b_score
    c_acc_s = c_acc + (f" ({cdiff:+d} vs baseline)" if cdiff != 0 else "")
    print(
        f"{'Baseline':<20} {b_acc:<25} {b_tot:<22,} {results_modes['baseline']['tokens_avg']:<14.0f} {results_modes['baseline']['time_s']:<12.2f}"
    )
    print(
        f"{'Peer':<20} {p_acc_s:<25} {p_tot:<22,} {results_modes['peer']['tokens_avg']:<14.0f} {results_modes['peer']['time_s']:<12.2f}"
    )
    print(
        f"{'Centralized':<20} {c_acc_s:<25} {c_tot:<22,} {results_modes['centralized']['tokens_avg']:<14.0f} {results_modes['centralized']['time_s']:<12.2f}"
    )
    print("=" * 80)
    print()

    delta_pc = int(p_score) - int(c_score)
    delta_pb = int(p_score) - int(b_score)
    print(
        f"Peer - Centralized (count): {delta_pc:+d}  |  "
        f"Peer - Baseline (count): {delta_pb:+d}"
    )
    print()

    return {
        "subject": subject,
        "label": label,
        "n_problems": n_problems,
        "baseline": results_modes["baseline"],
        "peer": results_modes["peer"],
        "centralized": results_modes["centralized"],
        "delta_peer_minus_central": delta_pc,
        "delta_peer_minus_baseline": delta_pb,
        "accuracy_baseline": b_score / n_problems,
        "accuracy_peer": p_score / n_problems,
        "accuracy_centralized": c_score / n_problems,
        "delta_perf_peer_minus_central": (p_score - c_score) / n_problems,
    }


def test_mmlu_moral_scenarios_and_professional_law(
    n_problems: int = 50,
    n_reviewers: int = 2,
    model=None,
    verbosity: int = 0,
) -> dict[str, dict[str, Any]]:
    """
    Run three-mode comparison on MMLU **moral_scenarios** and **professional_law**.

    Logs: ``data/mmlu/records/{subject}_{mode}_<date>.jsonl`` (per subject so two runs on the
    same day do not overwrite). Question IDs: ``data/mmlu/<subject>_question_ids.txt``.

    Returns
    -------
    Nested dict: ``out["moral_scenarios"]`` and ``out["professional_law"]``, each the return
    value of :func:`run_mmlu_subject_three_modes`.
    """
    out: dict[str, dict[str, Any]] = {}
    out["moral_scenarios"] = run_mmlu_subject_three_modes(
        "moral_scenarios",
        label="moral_scenarios",
        n_problems=n_problems,
        n_reviewers=n_reviewers,
        model=model,
        verbosity=verbosity,
    )
    out["professional_law"] = run_mmlu_subject_three_modes(
        "professional_law",
        label="professional_law",
        n_problems=n_problems,
        n_reviewers=n_reviewers,
        model=model,
        verbosity=verbosity,
    )
    return out


if __name__ == "__main__":
    test_mmlu_moral_scenarios_and_professional_law(
        n_problems=50,
        n_reviewers=2,
        verbosity=0,
    )
