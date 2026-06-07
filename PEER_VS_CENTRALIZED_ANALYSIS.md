# Peer Mode vs Centralized Mode: Performance Analysis

## Executive Summary

**Key Finding**: Peer mode significantly outperforms Centralized mode (35.4% vs 18.8% accuracy) despite Centralized mode accepting 100% of problems.

## Performance Comparison

| Metric | Baseline | Peer | Centralized |
|--------|----------|------|-------------|
| **Accuracy** | 12/48 (25.0%) | **17/48 (35.4%)** | 9/48 (18.8%) |
| **Acceptance Rate** | 46/48 (95.8%) | 47/48 (97.9%) | **48/48 (100%)** |
| **Avg Tokens** | 3,804 | 3,745 | 5,011 |
| **Round 1 Acceptance** | 34/48 (70.8%) | 35/48 (72.9%) | 27/48 (56.3%) |

## Critical Discovery

**Centralized mode accepts 100% of problems but only 18.8% are correct!**

This means:
- **81.2% of accepted answers are wrong** in Centralized mode
- The optimization made Centralized **too lenient**, accepting almost everything
- Peer mode finds a better balance: 97.9% acceptance with 35.4% accuracy

## Why Peer Mode Performs Better

### 1. **Independent World Models Prevent Error Propagation**

**Peer Mode**:
- Each agent (author, reviewers, meta) has its **own independent** world model
- Errors in one agent's reasoning don't contaminate others
- Agents can disagree and correct each other from different perspectives

**Centralized Mode**:
- **Single shared world model** across all agents
- Early errors (from author's draft or first review) propagate to all agents
- All agents see the same (potentially flawed) state summary
- Creates "groupthink" where everyone converges on the same wrong answer

### 2. **Peer Mode Solves 11 Problems That Centralized Gets Wrong**

Analysis of peer-only correct problems shows:

| Problem ID | Peer Rounds | Centralized Rounds | Issue |
|------------|-------------|-------------------|-------|
| 9 | 2 | 2 | Different calculation paths |
| 17 | 1 | 3 | Centralized over-corrects |
| 18 | 1 | 1 | Centralized gets wrong answer immediately |
| 19 | 2 | 2 | Different reasoning approaches |
| 27 | 1 | 2 | Centralized second-guesses correct answer |
| 29 | 1 | 2 | Centralized over-thinks simple problem |
| 31 | 1 | 2 | Centralized makes calculation error |
| 38 | 1 | 2 | Centralized over-corrects |
| 40 | 1 | 3 | Centralized needs more rounds but still wrong |
| 41 | 1 | 1 | Both same rounds, but different answers |
| 45 | 2 | 3 | Centralized over-complicates |

**Pattern**: 
- Peer often gets it right in **1-2 rounds**
- Centralized needs **2-3 rounds** but still gets it wrong
- Centralized's shared state leads to **over-correction** or **wrong convergence**

### 3. **Shared State Amplifies Errors**

In Centralized mode:
1. Author creates initial draft (may contain error)
2. **Shared world model** captures this state
3. Reviewers see the **same shared state** (with potential error)
4. Reviewers' feedback is influenced by the shared state
5. Meta-reviewer sees shared state + reviews (all potentially biased)
6. **Error compounds** through the shared state

In Peer mode:
1. Author creates initial draft (may contain error)
2. **Each reviewer has independent world model**
3. Reviewers can see the problem from **different angles**
4. If one reviewer's perspective is wrong, others aren't affected
5. **Diversity of perspectives** helps catch errors

### 4. **Acceptance Criteria Issues**

**Current Centralized Acceptance Criteria**:
- Round >= 1
- Risk <= 0.35 (Round 1) / 0.40 (Round 2) / 0.45 (Round 3+)
- At least 1 reviewer agrees

**Problem**: These criteria are **too lenient**:
- Risk thresholds are too high (0.35-0.45)
- Only requiring 1 reviewer agreement is too weak
- Result: Accepts almost everything, including wrong answers

**Peer Mode**: Uses original MARS acceptance (meta decision = "right")
- Simpler but more effective
- Meta-reviewer makes decision based on independent reviews
- No shared state bias

## Root Cause Analysis

### Why Centralized Mode Fails

1. **Shared State Noise**: Early errors in shared state contaminate all subsequent reasoning
2. **Over-Correction**: Multiple rounds with shared state lead to over-thinking and wrong convergence
3. **Groupthink**: All agents converge on the same (wrong) answer because they see the same state
4. **Lenient Acceptance**: Current thresholds accept too many wrong answers

### Why Peer Mode Succeeds

1. **Diversity**: Independent world models provide diverse perspectives
2. **Error Isolation**: Errors in one agent don't affect others
3. **Natural Correction**: Agents can disagree and correct each other
4. **Better Balance**: Simpler acceptance criteria work better without shared state bias

## Recommendations

### For Centralized Mode

1. **Tighten Acceptance Criteria**:
   - Lower risk thresholds (0.25 for Round 1, 0.30 for Round 2)
   - Require 2 reviewers to agree (not just 1)
   - Add quality check: reject if risk > 0.3 even if meta says "right"

2. **Improve Shared State Quality**:
   - Filter out noisy information from shared state
   - Only share high-confidence information
   - Use consensus-based state updates (only update if multiple agents agree)

3. **Consider Hybrid Approach**:
   - Use independent world models for reviewers (like Peer)
   - Use shared state only for meta-reviewer's final decision
   - This combines diversity with efficiency

### For Peer Mode

1. **Current performance is good** - maintain current approach
2. **Consider**: Add lightweight shared state for meta-reviewer only (not for reviewers)

## Conclusion

**Peer mode's independence is the key advantage**:
- Independent world models prevent error propagation
- Diversity of perspectives catches errors that shared state misses
- Simpler acceptance criteria work better without shared state bias

**Centralized mode's shared state is a double-edged sword**:
- Can be efficient when state is correct
- But amplifies errors when state is wrong
- Current implementation is too lenient, accepting wrong answers

**Recommendation**: For CIAR-like complex reasoning problems, **Peer mode is superior**. Centralized mode needs significant improvements to match Peer's performance.
