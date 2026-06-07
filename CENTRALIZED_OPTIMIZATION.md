# Centralized Mode Optimization Summary

## Problem Analysis

The original Centralized mode had the following issues:
1. **83% of problems accepted in Round 1** - Too aggressive, accepting answers before sufficient review
2. **Low accuracy (25.0%)** - Despite using fewer tokens, accuracy was worse than Baseline and Peer
3. **Shared state noise** - Errors in early rounds propagated to all agents
4. **No quality gates** - Accepted answers based solely on meta decision, without considering risk or consensus

## Optimizations Implemented

### 1. Enhanced Acceptance Criteria (`pipelines.py`)

**Before**: Accepted if `meta_decision == "right"`

**After**: For Centralized mode, requires ALL of:
- ✅ **Minimum 1 round of review** (`round_idx >= 1`)
- ✅ **Low risk score** (`risk < 0.3`)
- ✅ **Reviewer consensus** (at least 2 reviewers agree, or all if `n_reviewers <= 2`)

**Code Location**: `pipelines.py` lines 262-300

### 2. Improved Risk Assessment (`world_model/server.py`)

**Round-based Risk**:
- Round 0 (draft only): `risk = 0.5` (high - no review yet)
- Round 1 (first review): `risk = 0.3` (medium)
- Round 2+: `risk = 0.15 + 0.01 * depth` (lower - more thought)

**Reviewer Feedback Analysis**:
- Counts positive/negative indicators in reviews
- Extracts confidence scores from reviewers
- Adjusts risk based on:
  - Consensus ratio (< 50% agree → +0.25 risk)
  - Average confidence (< 0.6 → +0.15 risk)

**Code Location**: `world_model/server.py` lines 141-220

### 3. Consensus Mechanism

**Reviewer Agreement Detection**:
- Analyzes each review for positive/negative indicators
- Extracts reliability scores
- Counts reviewers who agree (positive > negative AND confidence > 0.6)

**Code Location**: `pipelines.py` lines 256-275

### 4. Shared State Quality Filtering

The improved rollout method now:
- Considers reviewer feedback when calculating risk
- Reduces risk only when there's strong consensus
- Increases risk when reviewers disagree or have low confidence

## Expected Improvements

1. **Reduced premature acceptance**: Round 1 acceptance should drop from 83% to < 50%
2. **Higher accuracy**: More thorough review process should catch errors
3. **Better risk assessment**: World Model now considers actual reviewer feedback
4. **Quality gates**: Multiple checks prevent accepting low-quality answers

## Testing Recommendations

Run the CIAR experiment again with the optimized Centralized mode:
```python
eval_mars(task="ciar", model=model, mode="centralized", n_problems=48)
```

Compare results:
- **Acceptance distribution**: Should see more Round 2+ acceptances
- **Accuracy**: Should improve from 25.0% baseline
- **Token usage**: May increase slightly due to more rounds, but should be more efficient per correct answer

## Key Changes Summary

| Component | Change | Impact |
|-----------|--------|--------|
| `pipelines.py` | Added acceptance criteria (round, risk, consensus) | Prevents premature acceptance |
| `world_model/server.py` | Enhanced rollout with round/feedback analysis | Better risk assessment |
| Consensus detection | Analyzes reviewer feedback | Ensures quality agreement |

## Notes

- Baseline and Peer modes remain unchanged (original behavior)
- Only Centralized mode uses the stricter acceptance criteria
- The optimization maintains backward compatibility with existing evaluation scripts
