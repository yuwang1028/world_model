"""
Run MMLU dataset with three modes: Baseline / Peer / Centralized (v3).
Includes accuracy, avg tokens, and time.
"""

import time

from evaluation import eval_mars


def main():
    print("=" * 80)
    print("Running Three Mode Comparison - MMLU (50 problems, centralized v3)")
    print("=" * 80)

    task = "mmlu"
    n_problems = 50
    n_reviewers = 2

    print(f"Task: {task}")
    print(f"Number of Problems: {n_problems}")
    print(f"Number of Reviewers: {n_reviewers}")
    print("Using Improved World Model (Centralized v3 - task-adaptive)")
    print("=" * 80)
    print()

    model = None  # use default from config.yml
    results = {}

    # 1) Baseline
    print("[1/3] Running Baseline (Original MARS)...")
    print("-" * 80)
    t0 = time.time()
    b_score, b_score_vote, _, b_tokens, _ = eval_mars(
        task=task,
        model=model,
        n_problems=n_problems,
        n_reviewers=n_reviewers,
        selected=False,
        mode="baseline",
        verbosity=0,
    )
    b_time = time.time() - t0
    results["baseline"] = {"score": b_score, "tokens": b_tokens, "time": b_time}
    print(f"✅ Baseline: {b_score}/{n_problems} ({b_score/n_problems*100:.1f}%)")
    print(f"   Avg Tokens: {b_tokens:.0f}, Total Time: {b_time:.1f}s")
    print()

    # 2) Peer
    print("[2/3] Running Peer Mode (World Model, peer)...")
    print("-" * 80)
    t0 = time.time()
    p_score, p_score_vote, _, p_tokens, _ = eval_mars(
        task=task,
        model=model,
        n_problems=n_problems,
        n_reviewers=n_reviewers,
        selected=True,  # Use same questions as baseline
        mode="peer",
        verbosity=0,
    )
    p_time = time.time() - t0
    results["peer"] = {"score": p_score, "tokens": p_tokens, "time": p_time}
    print(f"✅ Peer: {p_score}/{n_problems} ({p_score/n_problems*100:.1f}%)")
    print(f"   Avg Tokens: {p_tokens:.0f}, Total Time: {p_time:.1f}s")
    print()

    # 3) Centralized v3
    print("[3/3] Running Centralized Mode (World Model, v3 - task-adaptive)...")
    print("-" * 80)
    t0 = time.time()
    c_score, c_score_vote, _, c_tokens, _ = eval_mars(
        task=task,
        model=model,
        n_problems=n_problems,
        n_reviewers=n_reviewers,
        selected=True,  # Use same questions as baseline
        mode="centralized",
        verbosity=0,
    )
    c_time = time.time() - t0
    results["centralized"] = {"score": c_score, "tokens": c_tokens, "time": c_time}
    print(f"✅ Centralized (v3): {c_score}/{n_problems} ({c_score/n_problems*100:.1f}%)")
    print(f"   Avg Tokens: {c_tokens:.0f}, Total Time: {c_time:.1f}s")
    print()

    # Summary table
    print("=" * 80)
    print("MMLU (50 problems) - Summary")
    print("=" * 80)
    print(f"{'Mode':<15} {'Acc':<18} {'AvgTokens':<12} {'Time(s)':<10} {'Time/Prob(s)':<12}")
    print("-" * 80)
    for mode in ["baseline", "peer", "centralized"]:
        s = results[mode]
        acc = s["score"] / n_problems
        avg_toks = s["tokens"]
        t = s["time"]
        print(
            f"{mode:<15} "
            f"{s['score']}/{n_problems} ({acc*100:5.1f}%)   "
            f"{avg_toks:10.0f}   "
            f"{t:8.1f}   "
            f"{t/n_problems:10.2f}"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
