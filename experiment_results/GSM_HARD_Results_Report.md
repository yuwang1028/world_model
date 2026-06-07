## GSM_HARD Dataset: Three-Mode Comparison Results (Centralized V2)

**Date**: 2026-03-02  
**Dataset**: GSM_HARD (Adversarial Grade-School Math)  
**Number of Problems**: 50  
**Number of Reviewers**: 2  
**Centralized Logic**: **V2** (risk + confidence + consensus), no legacy shortcut  
**Question Set**: Fixed (from `data/gsm_hard/question_ids.txt`, truncated to the first 50 IDs)

### Executive Summary

On GSM_HARD, **Centralized (v2)** achieves **40.0%** accuracy, clearly outperforming **Baseline (30.0%)** and **Peer (32.0%)**, at the cost of higher token usage and runtime.  

This suggests that on harder, adversarial math problems, **Centralized v2’s combination of strict risk control and shared state can correct a subset of severe errors**, although the overall task remains very challenging and absolute accuracy is still modest.

### Results Comparison (50 Problems, Same Question IDs)

| Mode               | Accuracy                | Total Tokens (≈) | Avg Tokens | Time (s) | Time / Problem (s) |
|--------------------|-------------------------|------------------|------------|----------|---------------------|
| **Baseline**       | 15/50 (30.0%)           | ~157,045         | 3,141      | 470.2    | 9.40                |
| **Peer**           | 16/50 (32.0%) (+1 vs Baseline) | ~175,100  | 3,502      | 433.5    | 8.67                |
| **Centralized (v2)** | **20/50 (40.0%)** (+5 vs Baseline) | ~224,800 | 4,496      | 458.1    | 9.16                |

> Total tokens are approximated as `Avg Tokens × 50`, matching the logged averages.

### Accuracy Analysis

- **Centralized (v2)** is clearly better than the other modes:
  - vs Baseline: **+5 problems (+10.0%)**
  - vs Peer: **+4 problems (+8.0%)**
- **Peer vs Baseline**:
  - Very close (32.0% vs 30.0%), indicating that on GSM_HARD’s extreme / noisy math problems,  
    pure diversity with independent world models does not bring a large gain.

Overall, **Centralized v2 is currently the best configuration on GSM_HARD**, but the absolute accuracy is still low, reflecting the dataset’s adversarial difficulty and noise.

### Token & Time Efficiency

- **Baseline**:
  - Avg tokens: 3,141 / problem  
  - Time: 470.2 s total (~9.40 s / problem)
- **Peer**:
  - Avg tokens: 3,502 / problem (slightly higher than Baseline)  
  - Time: 433.5 s total (~8.67 s / problem, slightly faster)
- **Centralized (v2)**:
  - Avg tokens: 4,496 / problem (highest)  
  - Time: 458.1 s total (~9.16 s / problem)

**Trade-off**:
- Centralized v2 uses **more tokens** and slightly more time, in exchange for **+10 percentage points absolute accuracy** over Baseline.  
- On GSM_HARD, this “spend more to gain accuracy” pattern is similar to what we observed on SVAMP and GSM, but here the task is harder so the absolute scores are lower.

### Why Centralized V2 Helps on GSM_HARD

**GSM_HARD characteristics**:
- Adversarial perturbations of GSM; many problems have extreme numbers or odd setups.  
- Multi-step calculations where intermediate values can explode or go wrong.  
- Once a major error occurs, it is hard to fix with a single self-correction step.

**Centralized v2 advantages**:
- **Shared world-model state**:
  - Helps maintain consistent intermediate variables and computation paths across rounds.  
  - More stable when reasoning over very large or strange numbers.
- **Risk + confidence + consensus**:
  - High-risk, low-confidence answers are rejected or revised.  
  - Requires enough reviewers to genuinely agree before accepting.  
  - Makes it easier to flag and correct obviously wrong extreme values.

As a result, **Centralized v2 can clean up some of the most egregious errors, giving it an edge over Baseline and Peer on this adversarial math dataset.**

### Takeaways

1. **GSM_HARD reinforces a pattern**:  
   For **harder mathematical tasks (multi-step reasoning with extreme numbers)**, Centralized v2’s strict risk control and shared state are valuable.  
2. Compared to SVAMP / GSM, the **benefit of Centralized v2 still exists but is “compressed” by the overall difficulty of the dataset**, leading to lower absolute scores.  
3. If the goal is to maximize accuracy on adversarial GSM_HARD, **Centralized v2 is currently the best of the three modes**, despite its noticeably higher token and time cost.


