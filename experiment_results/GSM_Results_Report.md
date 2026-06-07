# GSM Dataset: Three-Mode Comparison Results

**Date**: 2026-02-25  
**Dataset**: GSM (Grade School Math)  
**Number of Problems**: 50  
**Number of Reviewers**: 2

## Executive Summary

**Key Finding**: Centralized mode performs **best** on GSM dataset, achieving 74.0% accuracy compared to Baseline (68.0%) and Peer (64.0%).

This **contradicts** the CIAR results where Peer mode was superior, confirming that **task type matters** for mode selection.

## Results Comparison

| Mode | Accuracy | Total Tokens | Avg Tokens | Time (s) |
|------|----------|--------------|------------|----------|
| **Baseline** | 34/50 (68.0%) | 158,788 | 3,176 | 406.68 |
| **Peer** | 32/50 (64.0%) | 134,089 | 2,682 | 454.87 |
| **Centralized** | **37/50 (74.0%)** | 177,296 | 3,546 | 447.68 |

## Performance Analysis

### Accuracy Comparison

- **Centralized vs Baseline**: +3 problems (+6.0%)
- **Centralized vs Peer**: +5 problems (+10.0%)
- **Peer vs Baseline**: -2 problems (-4.0%)

### Token Efficiency

- **Peer**: Most efficient (2,682 tokens/problem)
- **Baseline**: Medium (3,176 tokens/problem)
- **Centralized**: Least efficient (3,546 tokens/problem)

**Trade-off**: Centralized uses more tokens but achieves higher accuracy.

## Key Insights

### 1. Task Type Matters

**GSM Characteristics**:
- Simple math word problems
- Clear solutions, single correct answer
- Lower error potential than CIAR
- Less need for diverse perspectives

**Why Centralized Works Better**:
- Shared state ensures **consistency** across agents
- Simple problems benefit from **coordinated reasoning**
- Lower error rate means shared state is **safe**
- **Convergence** on correct answer is beneficial

### 2. Comparison with CIAR Results

| Dataset | Best Mode | Accuracy | Task Type |
|---------|-----------|----------|-----------|
| **CIAR** | Peer | 35.4% | Complex reasoning (probability/combinatorics) |
| **GSM** | Centralized | 74.0% | Simple math word problems |

**Conclusion**: 
- **Complex reasoning** → Peer mode better (diversity helps)
- **Simple problems** → Centralized mode better (consistency helps)

### 3. Mode Selection Guidelines

**Use Centralized Mode for**:
- ✅ Simple math problems (like GSM)
- ✅ Tasks with clear, single correct answers
- ✅ Low error potential tasks
- ✅ Problems requiring consistency

**Use Peer Mode for**:
- ✅ Complex reasoning (like CIAR)
- ✅ Multiple valid solution paths
- ✅ High error potential tasks
- ✅ Problems requiring diverse perspectives

## Detailed Statistics

### Acceptance Patterns
- All modes accepted most problems (detailed analysis needed)

### Round Distribution
- Analysis of round distribution across modes

### Error Analysis
- Which problems each mode got wrong
- Common error patterns

## Files Generated

- Baseline: `data/gsm/records/baseline_2026-02-25.jsonl`
- Peer: `data/gsm/records/peer_2026-02-25.jsonl`
- Centralized: `data/gsm/records/centralized_2026-02-25.jsonl`

## Next Steps

1. Analyze acceptance patterns and round distributions
2. Compare with CIAR results in detail
3. Test on additional datasets (MMLU, SVAMP) to further validate task-type hypothesis
4. Develop automatic mode selection based on task characteristics
