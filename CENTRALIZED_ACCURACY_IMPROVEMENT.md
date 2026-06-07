# Centralized Mode Accuracy Improvement

## Problem Identified

**Current Performance**: 9/48 (18.8%) accuracy, but 100% acceptance rate
- **39/48 wrong answers were accepted**
- Round 1: 20 wrong answers accepted
- Round 2: 12 wrong answers accepted  
- Round 3: 7 wrong answers accepted

**Root Cause**: Acceptance criteria too lenient

## Improvements Implemented

### 1. Stricter Acceptance Criteria (`pipelines.py`)

**Before**:
- Risk threshold: 0.35 (Round 1), 0.40 (Round 2), 0.45 (Round 3+)
- Consensus: Only 1 reviewer needed
- No confidence requirement

**After**:
- Risk threshold: **0.25 (Round 1)**, **0.30 (Round 2)**, **0.35 (Round 3+)** - Stricter
- Consensus: **At least 2 reviewers** must agree (or all if n_reviewers <= 2)
- **New**: Average reviewer confidence must be **> 0.7**

**Code Location**: `pipelines.py` lines 301-340

### 2. Stricter Risk Assessment (`world_model/server.py`)

**Base Risk Adjustments**:
- Round 1: 0.25 → **0.30** (higher base risk)
- Round 2+: 0.15 → **0.20** (higher base risk)

**Consensus-Based Risk Adjustments**:
- Low consensus (< 50%): +0.30 risk (was +0.25)
- Moderate consensus (50-70%): +0.15 risk (was +0.1)
- Strong but not unanimous (70-100%): +0.05 risk (new)
- Unanimous: -0.1 risk

**Confidence-Based Risk Adjustments**:
- Low confidence (< 0.6): +0.20 risk (was +0.15)
- Moderate confidence (0.6-0.7): +0.10 risk (new)
- Good confidence (0.7-0.85): -0.05 risk (new)
- Very high confidence (> 0.85): -0.15 risk (was -0.1)

**Code Location**: `world_model/server.py` lines 153-220

### 3. Quality-Filtered Shared State (`world_model/server.py`)

**New Feature**: Filter low-quality information from shared state
- Detects error-prone content and returns warnings instead
- Identifies high-failure-rate patterns and suggests alternatives
- Prevents error propagation through shared state

**Code Location**: `world_model/server.py` lines 227-260

## Expected Improvements

1. **Higher Accuracy**: Should improve from 18.8% to 25-30%+
2. **Lower Acceptance Rate**: Should drop from 100% to 70-80%
3. **Better Quality Control**: Only high-quality answers accepted
4. **Reduced Error Propagation**: Filtered shared state prevents error amplification

## Key Changes Summary

| Component | Change | Impact |
|-----------|--------|--------|
| `pipelines.py` | Stricter risk thresholds (0.25/0.30/0.35) | Rejects more low-quality answers |
| `pipelines.py` | Require 2 reviewers + confidence > 0.7 | Stronger quality gates |
| `world_model/server.py` | Higher base risk (0.30/0.20) | More conservative risk assessment |
| `world_model/server.py` | Stricter consensus/confidence adjustments | Better risk calibration |
| `world_model/server.py` | Quality-filtered shared state | Prevents error propagation |

## Testing

Re-run the CIAR experiment to verify improvements:
```python
python run_ciar_optimized.py
```

Expected results:
- Accuracy: 25-30%+ (up from 18.8%)
- Acceptance rate: 70-80% (down from 100%)
- Token usage: May increase slightly due to more rounds, but better accuracy

## Trade-offs

**Benefits**:
- Higher accuracy (fewer wrong answers accepted)
- Better quality control
- Reduced error propagation

**Costs**:
- Lower acceptance rate (may reject some correct answers)
- Slightly more tokens (more rounds for borderline cases)
- Slower execution (more rounds)

The goal is to find the right balance: **accept fewer answers, but ensure accepted answers are more likely to be correct**.
