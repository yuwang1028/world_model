"""
Run three-mode comparison experiment on GSM_HARD dataset (50 problems).
Uses current Centralized logic (v2/v3 as configured in PipelineRunner).
"""

from evaluation import eval_mars
import time


def main():
    print("=" * 80)
    print("Running Three Mode Comparison - GSM_HARD")
    print("=" * 80)

    task = "gsm_hard"
    n_problems = 50
    n_reviewers = 2

    print(f"Task: {task}")
    print(f"Number of Problems: {n_problems}")
    print(f"Number of Reviewers: {n_reviewers}")
    print("Using current World Model configuration")
    print("=" * 80)
    print()

    model = None  # use default from config.yml
    results = {}

    # [1/3] Baseline
    print("[1/3] Running Baseline (Original MARS)...")
    print("-" * 80)
    start_time = time.time()
    baseline_score, baseline_score_vote, _, baseline_tokens, baseline_time = eval_mars(
        task=task,
        model=model,
        n_problems=n_problems,
        n_reviewers=n_reviewers,
        selected=True,
        mode="baseline",
        verbosity=0,
    )
    baseline_elapsed = time.time() - start_time
    results["baseline"] = {
        "score": baseline_score,
        "tokens": baseline_tokens,
        "time": baseline_elapsed,
    }
    print(f"✅ Baseline: {baseline_score}/{n_problems} ({baseline_score/n_problems*100:.1f}%)")
    print()

    # [2/3] Peer
    print("[2/3] Running Peer Mode (Improved World Model)...")
    print("-" * 80)
    start_time = time.time()
    peer_score, peer_score_vote, _, peer_tokens, peer_time = eval_mars(
        task=task,
        model=model,
        n_problems=n_problems,
        n_reviewers=n_reviewers,
        selected=True,
        mode="peer",
        verbosity=0,
    )
    peer_elapsed = time.time() - start_time
    results["peer"] = {
        "score": peer_score,
        "tokens": peer_tokens,
        "time": peer_elapsed,
    }
    print(f"✅ Peer: {peer_score}/{n_problems} ({peer_score/n_problems*100:.1f}%)")
    print()

    # [3/3] Centralized
    print("[3/3] Running Centralized Mode (World Model)...")
    print("-" * 80)
    start_time = time.time()
    centralized_score, centralized_score_vote, _, centralized_tokens, centralized_time = eval_mars(
        task=task,
        model=model,
        n_problems=n_problems,
        n_reviewers=n_reviewers,
        selected=True,
        mode="centralized",
        verbosity=0,
    )
    centralized_elapsed = time.time() - start_time
    results["centralized"] = {
        "score": centralized_score,
        "tokens": centralized_tokens,
        "time": centralized_elapsed,
    }
    print(f"✅ Centralized: {centralized_score}/{n_problems} ({centralized_score/n_problems*100:.1f}%)")
    print()

    # Print comparison
    print("=" * 80)
    print("GSM_HARD (50 problems) - Results Comparison")
    print("=" * 80)
    print(f"{'Mode':<20} {'Accuracy':<25} {'Total Tokens':<20} {'Avg Tokens':<20} {'Time (s)':<15}")
    print("-" * 80)

    baseline_acc = f"{results['baseline']['score']}/{n_problems} ({results['baseline']['score']/n_problems*100:.1f}%)"
    peer_acc = f"{results['peer']['score']}/{n_problems} ({results['peer']['score']/n_problems*100:.1f}%)"
    centralized_acc = f"{results['centralized']['score']}/{n_problems} ({results['centralized']['score']/n_problems*100:.1f}%)"

    baseline_total_tokens = int(results["baseline"]["tokens"] * n_problems)
    peer_total_tokens = int(results["peer"]["tokens"] * n_problems)
    centralized_total_tokens = int(results["centralized"]["tokens"] * n_problems)

    print(
        f"{'Baseline':<20} {baseline_acc:<25} {baseline_total_tokens:<20,} "
        f"{results['baseline']['tokens']:<20.0f} {results['baseline']['time']:<15.2f}"
    )

    peer_diff = results["peer"]["score"] - results["baseline"]["score"]
    peer_diff_str = f" ({peer_diff:+d} vs baseline)" if peer_diff != 0 else ""
    print(
        f"{'Peer':<20} {peer_acc + peer_diff_str:<25} {peer_total_tokens:<20,} "
        f"{results['peer']['tokens']:<20.0f} {results['peer']['time']:<15.2f}"
    )

    centralized_diff = results["centralized"]["score"] - results["baseline"]["score"]
    centralized_diff_str = f" ({centralized_diff:+d} vs baseline)" if centralized_diff != 0 else ""
    print(
        f"{'Centralized':<20} {centralized_acc + centralized_diff_str:<25} {centralized_total_tokens:<20,} "
        f"{results['centralized']['tokens']:<20.0f} {results['centralized']['time']:<15.2f}"
    )
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()

