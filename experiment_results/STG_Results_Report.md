## STG Dataset: Three-Mode Comparison Results

**Date**: 2026-03-02  
**Dataset**: STG (True/False QA)  
**Number of Problems**: 50  
**Number of Reviewers**: 2  
**Question Set**: Fixed (from `data/stg/question_ids.txt`)

### Executive Summary

**Key Finding**: On STG, **Peer** performs best in accuracy (**78.0%**), with **Baseline** close behind (**68.0%**). **Centralized (v2)** does **not** improve accuracy over Baseline and is substantially more expensive in tokens and time.

### Results Comparison (50 Problems, Same Question IDs)

| Mode                     | Accuracy         | Total Tokens (≈) | Avg Tokens | Time (s) | Time / Problem (s) |
|--------------------------|------------------|------------------|------------|----------|---------------------|
| **Baseline**             | 34/50 (68.0%)    | ~113,550         | 2,271      | 351.7    | 7.03                |
| **Peer**                 | 39/50 (78.0%)    | ~105,050         | 2,101      | 305.1    | 6.10                |
| **Centralized (v2 logic)** | 32/50 (64.0%) | ~209,350         | 4,187      | 463.9    | 9.28                |

> Total tokens are approximated as `avg_tokens × 50` from the logged averages.

### Performance Analysis

- **Accuracy**
  - **Peer vs Baseline**: +5 problems (+10.0%)
  - **Centralized (v2) vs Baseline**: −2 problems (−4.0%)
  - **Centralized (v2) vs Peer**: −7 problems (−14.0%)

- **Token & Time Efficiency**
  - **Baseline**: cheapest and relatively fast.
  - **Peer**: slightly fewer tokens than Baseline in this run and the best accuracy.
  - **Centralized (v2)**: uses **~2×** tokens of Baseline, is **slower**, but **does not gain accuracy**.

### Why Centralized (v2) Underperforms on STG

- **Task characteristics**:
  - Binary True/False decisions with short answers.
  - Limited room for multi-step reasoning or complex world-state tracking.
  - Many borderline cases where reviewers may disagree slightly but the underlying answer is still simple.

- **Centralized v2 logic characteristics**:
  - Uses **risk + reviewer confidence + consensus** as strict acceptance gates.
  - Requires sufficient consensus and confidence to accept; otherwise triggers extra rounds or rejects.
  - On STG, this adds overhead (more tokens、更多轮次) but并未显著过滤掉“致命错误”，因为任务本身比较简单。

**Conclusion**:  
For STG，**Peer** is the best choice：它在准确率上明显优于 Baseline 和 Centralized，同时成本也可控；Centralized v2 的复杂风控逻辑在这种 True/False 任务上没有换来有效的精度提升，反而增加了计算开销。+

