# MMLU–Moral Scenarios: Three-Mode Comparison Results

**Date**: 2026-03-30  
**Dataset**: MMLU – `moral_scenarios`  
**Number of Problems**: 50  
**Number of Reviewers**: 2

## Executive Summary

**Key Finding**: On MMLU–moral_scenarios，**Peer** clearly outperforms both **Baseline** and **Centralized**. Peer gains **+6 problems (+18 pp)** over Centralized and **+3 problems (+6 pp)** over Baseline, with only moderate extra tokens compared to Baseline and far fewer than Centralized.  

This matches the CUI signal: `mmlu_moral` has relatively high coordination uncertainty, and diverse moral reasoning from multiple independent reviewers appears more beneficial than centralized risk gating.

## Results Comparison (50 Problems, Same Question IDs)

| Mode               | Accuracy                      | Total Tokens (≈) | Avg Tokens | Time (s) | Time / Problem (s) |
|--------------------|-------------------------------|------------------|------------|----------|---------------------|
| **Baseline**       | 16/50 (32.0%)                 | ~125,442         | 2,509      | 390.39   | 7.81                |
| **Peer**           | 19/50 (38.0%) (+3 vs Baseline) | ~142,542         | 2,851      | 401.59   | 8.03                |
| **Centralized (v2)** | 10/50 (20.0%) (−6 vs Baseline) | ~286,536       | 5,731      | 656.36   | 13.13               |

> Total tokens are approximated as `Avg Tokens × 50`, matching the logged averages.

## Performance Analysis

- **Accuracy**
  - **Peer vs Baseline**: +3/50 (+6.0 pp)  
  - **Peer vs Centralized**: +9/50 (+18.0 pp)  
  - **Centralized vs Baseline**: −6/50 (−12.0 pp)  
- **Token & Time**
  - Peer uses slightly more tokens than Baseline but far fewer than Centralized.  
  - Centralized uses almost **2.3×** Baseline tokens and is much slower, yet performs worst.

## Why Peer Helps on Moral Scenarios

**Task characteristics**:

- Moral scenarios often admit multiple plausible answers or justifications.  
- Annotator and model judgments are noisy and value-laden, leading to high disagreement and high CUI.  

**Peer advantages**:

- Independent reviewers explore diverse moral reasoning paths.  
- Majority agreement across reviewers helps filter out obviously flawed reasoning while preserving genuinely persuasive justifications.  
- The final answer benefits from multiple aligned but independent “votes” rather than a single centralized controller.

**Centralized limitations**:

- Strict risk + confidence + consensus gates can over-reject answers in highly ambiguous cases.  
- When ground-truth labels are sharp but human/machine moral judgments are inherently fuzzy, aggressive centralized rejection can turn borderline-correct reasoning into outright wrong predictions.

## Takeaways

1. MMLU–moral_scenarios is a clear example where **high CUI corresponds to a strong Peer advantage** over Centralized.  
2. Peer delivers higher accuracy than both Baseline and Centralized at a moderate token cost.  
3. On value-laden, high-uncertainty moral tasks, **diversity of reasoning and simple consensus can outperform strict centralized risk control**.

