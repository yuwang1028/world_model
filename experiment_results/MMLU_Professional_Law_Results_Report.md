# MMLU–Professional Law: Three-Mode Comparison Results

**Date**: 2026-03-30  
**Dataset**: MMLU – `professional_law`  
**Number of Problems**: 50  
**Number of Reviewers**: 2

## Executive Summary

**Key Finding**: On MMLU–professional_law，**Peer** and **Centralized** achieve the **same accuracy** (44.0%) and both improve over Baseline (38.0%). Peer is slightly more token‑efficient and faster than Centralized.  

This dataset therefore acts as a **medium‑CUI “tie case”**: collaboration (Peer/Centralized) clearly helps over a single agent, but the choice between Peer and Centralized is not decisive.

## Results Comparison (50 Problems, Same Question IDs)

| Mode               | Accuracy                      | Total Tokens (≈) | Avg Tokens | Time (s) | Time / Problem (s) |
|--------------------|-------------------------------|------------------|------------|----------|---------------------|
| **Baseline**       | 19/50 (38.0%)                 | ~178,117         | 3,562      | 492.93   | 9.86                |
| **Peer**           | 22/50 (44.0%) (+3 vs Baseline) | ~172,324        | 3,446      | 434.38   | 8.69                |
| **Centralized (v2)** | 22/50 (44.0%) (+3 vs Baseline) | ~239,694      | 4,794      | 567.05   | 11.34               |

> Total tokens are approximated as `Avg Tokens × 50`, matching the logged averages.

## Performance Analysis

- **Accuracy**
  - **Peer vs Baseline**: +3/50 (+6.0 pp)  
  - **Centralized vs Baseline**: +3/50 (+6.0 pp)  
  - **Peer vs Centralized**: 0/50 (tie)  
- **Token & Time**
  - Peer uses slightly fewer tokens and less time than Baseline.  
  - Centralized is more expensive than Peer in both tokens and wall‑clock time, for the same accuracy.

## Interpretation

**Task characteristics**:

- Professional law questions have **well‑defined, technical correct answers**, often grounded in formal doctrine and precedent.  
- While reasoning chains can be long, there is typically a single correct option among A/B/C/D, reducing inherent ambiguity compared to moral scenarios.

**What the three modes show**:

- **Baseline** struggles most, likely due to missing subtle legal distinctions or misinterpreting key conditions in the prompt.  
- **Peer** and **Centralized** both help by surfacing additional reasoning and cross‑checking answers, correcting a portion of Baseline’s mistakes.  
- However, because the ground truth is relatively sharp and evidence is more objective than in moral scenarios, **Centralized’s extra risk gating does not yield clear gains over Peer**—the two end up tied in accuracy.

## Takeaways

1. On MMLU–professional_law，**collaborative modes (Peer/Centralized) are clearly better than a single agent**, but neither collaboration style dominates the other.  
2. Peer is slightly more efficient than Centralized (fewer tokens and less time) at the same accuracy level.  
3. This dataset provides an important **boundary case** for the overall CUI story: at **medium levels of coordination uncertainty**, collaboration helps, but the choice between Peer and Centralized may depend more on cost and implementation preferences than on accuracy alone.

