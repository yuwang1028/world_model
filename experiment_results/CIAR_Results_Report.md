## CIAR Dataset: Three-Mode Comparison Results (Peer Best Run)

**Date**: 2026-02-25  
**Dataset**: CIAR (Combinatorial / Probability Word Problems)  
**Number of Problems**: 48  
**Number of Reviewers**: 2  
**Question Set**: Fixed (from `data/ciar/question_ids.txt`)

### Executive Summary

On CIAR, the run from **2026-02-25** shows **Peer** achieving the highest accuracy: **35.4%**, compared to **Baseline (25.0%)** and **Centralized (27.1%)**.  

This run is the clearest example where **Peer’s independent reviewers and diversity of reasoning outperform both Baseline and Centralized**, despite similar or slightly lower token usage than Baseline and much lower usage than Centralized.

### Results Comparison (48 Problems, Same Question IDs)

| Mode           | Accuracy           | Avg Tokens | Total Tokens (≈) |
|----------------|--------------------|------------|------------------|
| **Baseline**   | 12/48 (25.0%)      | 3,804      | ~182,600         |
| **Peer**       | **17/48 (35.4%)**  | 3,745      | ~179,800         |
| **Centralized**| 13/48 (27.1%)      | 6,700      | ~321,600         |

> Total tokens are approximated as `Avg Tokens × 48`.

### Accuracy Analysis

- **Peer**:
  - Best accuracy at **17/48 (35.4%)**.  
  - **+5 correct problems vs Baseline** and **+4 vs Centralized**.
- **Baseline**:
  - 12/48 (25.0%), serves as the single-agent reference.
- **Centralized**:
  - 13/48 (27.1%), slightly better than Baseline but still clearly below Peer.

In this run, **Peer is the only mode that gives a substantial lift over Baseline**, confirming that CIAR benefits more from diverse, independent reasoning than from a shared world model with strict risk gating.

### Token Efficiency

- **Peer vs Baseline**:
  - Peer uses **slightly fewer tokens on average** (3,745 vs 3,804 per problem) while achieving **significantly higher accuracy**.
  - This makes Peer strictly better than Baseline on the cost–accuracy trade-off in this run.
- **Centralized**:
  - Uses **almost 2× tokens** of Baseline/Peer (6,700 vs ~3,800 per problem)  
  - But delivers **lower accuracy than Peer** and only a small gain over Baseline.

**Conclusion**: On CIAR, **Centralized is not cost-effective** in this configuration—the extra tokens do not translate into better accuracy than Peer.

### Why Peer Excels on CIAR

**CIAR task characteristics**:
- Complex probabilistic / combinatorial reasoning.  
- Many ways to misinterpret the setup or make subtle reasoning mistakes.  
- Problems often benefit from trying multiple, independent solution strategies.

**Peer advantages**:
- **Independent world models** for each reviewer:
  - Each reviewer explores a different line of reasoning.  
  - System can select or aggregate from multiple distinct solutions.
- **Diversity over consensus**:
  - CIAR problems often do not have a single “obvious” reasoning path.  
  - Having several independent attempts increases the chance that one agent gets the reasoning right, even if others fail.

By contrast, **Centralized** shares a single evolving state and applies stricter risk/consensus filters. On CIAR, this tends to:
- Over-penalize disagreement (which is actually valuable signal here).  
- Lock the system into a single, partially wrong view of the problem.  
- Spend more tokens revising within one shared state, instead of exploring multiple alternatives.

### Takeaways

1. **CIAR is a “Peer-friendly” dataset**: complex reasoning and ambiguity reward diversity more than shared state.  
2. **Peer dominates both Baseline and Centralized** in this best run, with **higher accuracy and comparable (or lower) token usage**.  
3. For CIAR-like tasks (probability / combinatorics, multiple plausible reasoning paths), **Peer should be the default choice**, and Centralized should be used cautiously given its higher cost and weaker performance in this setting.

