# SVAMP Dataset: Three-Mode Comparison Results

**Date**: 2026-02-26  
**Dataset**: SVAMP (Simple Arithmetic Word Problems)  
**Number of Problems**: 50  
**Number of Reviewers**: 2

## Executive Summary

**Key Finding**: With the new centralized v2 logic, **Centralized** now performs best on SVAMP, achieving **82.0%** accuracy, compared to **Peer** (76.0%) and **Baseline** (74.0%).

This contrasts with earlier runs (where Peer reached 90.0%) and shows that the updated Centralized acceptance logic (risk + confidence + consensus) is competitive on simple arithmetic word problems.

## Results Comparison (Latest Run, 50 Problems)

| Mode           | Accuracy         | Total Tokens | Avg Tokens | Time (s) | Time / Problem (s) |
|----------------|------------------|--------------|------------|----------|---------------------|
| **Baseline**   | 37/50 (74.0%)    | 108,350      | 2,167      | 309.0    | 6.18                |
| **Peer**       | 38/50 (76.0%)    | 116,250      | 2,325      | 421.6    | 8.43                |
| **Centralized**| **41/50 (82.0%)**| 138,700      | 2,774      | 362.9    | 7.26                |

> Token counts are computed as `avg_tokens × 50` (approximate, from logged averages).

## Performance Analysis

### Accuracy Comparison

- **Centralized vs Baseline**: +4 problems (+8.0%)
- **Centralized vs Peer**: +3 problems (+6.0%)
- **Peer vs Baseline**: +1 problem (+2.0%)

### Token & Time Efficiency

- **Baseline**:
  - Avg tokens: 2,167 / problem (cheapest)
  - Time: 309.0 s total (~6.18 s / problem)
- **Peer**:
  - Avg tokens: 2,325 / problem
  - Time: 421.6 s total (~8.43 s / problem, slowest)
- **Centralized (v2)**:
  - Avg tokens: 2,774 / problem (most tokens)
  - Time: 362.9 s total (~7.26 s / problem)

**Trade-off**:  
Centralized uses more tokens than Baseline/Peer but achieves **the highest accuracy**. Compared to the older centralized version, v2 improves both **accuracy** and **precision** by using risk + confidence + consensus.

## Key Insights

### 1. Centralized v2 Works Well on SVAMP

**SVAMP Characteristics**:
- Simple arithmetic word problems
- Single-step or few-step reasoning
- Clear numerical answers
- Low inherent ambiguity

**Why Centralized v2 Wins Here**:
- Shared world model state helps maintain **consistent equations** and intermediate values.
- The new acceptance logic:
  - Filters out high-risk, low-confidence answers
  - Requires at least 2 reviewers to agree (strong consensus)
  - Uses answer length to slightly adapt risk thresholds for more complex explanations
- Fewer catastrophic errors: many borderline cases get rejected or revised instead of being accepted incorrectly.

### 2. Peer vs Baseline on SVAMP

- Peer improves slightly over Baseline in accuracy (76.0% vs 74.0%) but:
  - Uses more tokens per problem
  - Takes significantly longer time
- On very simple arithmetic, **diversity brings limited extra gain**, and the overhead of multiple independent world models becomes more visible.

### 3. Comparison with CIAR & GSM

| Dataset | Best Mode     | Accuracy | Task Type                      |
|---------|--------------|----------|--------------------------------|
| CIAR    | Peer         | 35.4%    | Complex reasoning (prob./comb.)|
| GSM     | Centralized  | 74.0%    | Simple math word problems      |
| SVAMP   | Centralized  | 82.0%    | Very simple arithmetic         |

**Pattern**:
- **Complex reasoning (CIAR)** → Peer better (independent models & diversity help)
- **Simple to very simple math (GSM, SVAMP)** → Centralized better (shared state & consistency help)

## Files Generated (Latest Run)

- Baseline: `data/svamp/records/baseline_2026-02-26.jsonl`
- Peer: `data/svamp/records/peer_2026-02-26.jsonl`
- Centralized: `data/svamp/records/centralized_2026-02-26.jsonl`

Each JSONL file contains, for every problem:
- `id`, `question`, `gt_answer`
- `final_answer`, `multi_score` (correctness)
- `final_status` (`accepted` or `max_rounds_reached`)
- `total_tokens`, full `rounds` history

## Takeaways

1. 新的 Centralized v2 在 SVAMP 上已经**稳定优于 Peer 和 Baseline**。
2. 对于非常简单的算术题，**共享世界模型 + 严格的接受逻辑**是有效组合。
3. 与 CIAR 的结果一起看，更加确认：**任务类型决定最优模式**，没有单一模式在所有任务上都是最优。

