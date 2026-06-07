# Centralized Mode Performance Fix

## Problem Identified

After the initial optimization, Centralized mode performance **worsened**:
- **Token usage doubled**: 3,231 → 6,622 (+105%)
- **Accuracy unchanged**: 25.0% (12/48)
- **Time increased**: 9.26s → 13.99s
- **Acceptance rate**: 29/48 (60.4%), but only 9/29 (31.0%) were correct
- **3 correct answers were rejected** (standards too strict)

## Root Causes

1. **Round 1 risk threshold too strict**: `risk < 0.3` with base_risk = 0.3 meant any adjustment pushed risk over threshold
2. **Consensus requirement too high**: Requiring 2 reviewers to agree was too strict
3. **More rounds without accuracy improvement**: Extra rounds didn't improve answer quality, just wasted tokens

## Fixes Applied

### 1. Lowered Round 1 Base Risk (`world_model/server.py`)
- **Before**: `base_risk = 0.3` for Round 1
- **After**: `base_risk = 0.25` for Round 1
- **Impact**: Gives more headroom for risk adjustments

### 2. Round-Based Risk Thresholds (`pipelines.py`)
- **Before**: Fixed threshold `risk < 0.3` for all rounds
- **After**: Dynamic thresholds:
  - Round 1: `risk <= 0.35`
  - Round 2: `risk <= 0.40`
  - Round 3+: `risk <= 0.45`
- **Impact**: More lenient in later rounds, allowing acceptance when appropriate

### 3. Relaxed Consensus Requirement (`pipelines.py`)
- **Before**: Required at least 2 reviewers to agree
- **After**: Requires only 1 reviewer to agree
- **Impact**: Reduces false rejections while maintaining quality check

### 4. Improved Consensus Detection (`pipelines.py`)
- **Expanded positive indicators**: Added "looks good", "seems correct", "satisfactory", etc.
- **More lenient logic**: Reviewer agrees if:
  - `positive_count > negative_count` OR
  - `confidence > 0.7` OR
  - `positive_count >= 2`
- **Impact**: Better detection of reviewer agreement, reducing false negatives

## Expected Improvements

1. **Reduced token usage**: Should decrease from 6,622 to ~4,000-5,000
2. **Maintained or improved accuracy**: Should stay at 25%+ or improve
3. **Better acceptance rate**: Should accept more correct answers (reduce false rejections)
4. **Faster execution**: Fewer unnecessary rounds

## Key Changes Summary

| Component | Change | Impact |
|-----------|--------|--------|
| `world_model/server.py` | Round 1 base_risk: 0.3 → 0.25 | Lower initial risk |
| `pipelines.py` | Dynamic risk thresholds by round | More lenient acceptance |
| `pipelines.py` | Consensus: 2 → 1 reviewer | Less strict requirement |
| `pipelines.py` | Improved consensus detection | Better agreement recognition |

## Testing

Re-run the CIAR experiment to verify improvements:
```python
python run_ciar_optimized.py
```

Expected results:
- Token usage: ~4,000-5,000 (down from 6,622)
- Accuracy: 25%+ (maintained or improved)
- Average rounds: ~1.5-2.0 (down from 2.12)
