"""
Improved World Model Server with actual learning capabilities
"""

import json
import os
from collections import defaultdict
import re

class WorldModelServer:
    def __init__(self, mode="centralized", history_file="world_model/history.json"):
        self.mode = mode
        self.states = {}
        self.counter = 0
        
        # Learning components
        self.history_file = history_file
        self.load_history()
        
        # Track patterns
        self.success_patterns = defaultdict(int)  # pattern -> success count
        self.failure_patterns = defaultdict(int)  # pattern -> failure count
        self.problem_types = {}  # problem signature -> best strategy
        
    def load_history(self):
        """Load historical data for learning"""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                    self.success_patterns = defaultdict(int, data.get('success_patterns', {}))
                    self.failure_patterns = defaultdict(int, data.get('failure_patterns', {}))
                    self.problem_types = data.get('problem_types', {})
            except:
                pass
    
    def save_history(self):
        """Save historical data"""
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
        with open(self.history_file, 'w') as f:
            json.dump({
                'success_patterns': dict(self.success_patterns),
                'failure_patterns': dict(self.failure_patterns),
                'problem_types': self.problem_types
            }, f, indent=2)
    
    def create_initial(self, user_query: str):
        """Create initial state from user query"""
        return self.transition(None, {"type": "init", "content": user_query})
    
    def transition(self, parent_state, action, observation=None):
        state_id = f"s{self.counter}"
        self.counter += 1
        
        self.states[state_id] = {
            "parent": parent_state,
            "action": action,
            "observation": observation
        }
        return state_id
    
    def extract_reasoning_pattern(self, answer_content: str) -> str:
        """Extract reasoning pattern from answer (e.g., 'step_by_step', 'direct_calculation')"""
        content_lower = answer_content.lower()
        
        # Detect patterns
        if "step" in content_lower or "first" in content_lower or "then" in content_lower:
            return "step_by_step"
        elif "calculate" in content_lower or "=" in answer_content:
            return "direct_calculation"
        elif "let" in content_lower or "assume" in content_lower:
            return "variable_assignment"
        elif "if" in content_lower or "when" in content_lower:
            return "conditional_reasoning"
        else:
            return "general"
    
    def extract_problem_signature(self, user_query: str) -> str:
        """Extract problem type signature (e.g., 'arithmetic', 'word_problem')"""
        query_lower = user_query.lower()
        
        # Detect problem types
        if any(word in query_lower for word in ["how much", "how many", "total", "sum"]):
            return "arithmetic_sum"
        elif any(word in query_lower for word in ["per", "each", "rate", "speed"]):
            return "rate_problem"
        elif any(word in query_lower for word in ["probability", "chance", "likely"]):
            return "probability"
        elif any(word in query_lower for word in ["percent", "%", "percentage"]):
            return "percentage"
        else:
            return "general"
    
    def validate(self, state_id, action):
        """Improved validation with pattern-based quality assessment"""
        content = action.get("content", "")
        
        # Basic checks
        if len(content) < 10:
            return {"ok": False, "suggestion": "Answer too short. Please provide more detailed reasoning."}
        
        # Pattern-based quality check
        pattern = self.extract_reasoning_pattern(content)
        
        # Check if this pattern has high failure rate
        total_attempts = self.success_patterns[pattern] + self.failure_patterns[pattern]
        if total_attempts > 10:
            failure_rate = self.failure_patterns[pattern] / total_attempts
            if failure_rate > 0.7:  # High failure rate
                return {
                    "ok": False,
                    "suggestion": f"Consider using a different reasoning approach. The '{pattern}' pattern has a high failure rate.",
                    "risk_score": failure_rate
                }
        
        # Check for common error patterns
        if self.has_common_errors(content):
            return {
                "ok": False,
                "suggestion": "Check for common calculation errors. Review each step carefully.",
                "risk_score": 0.6
            }
        
        return {"ok": True, "risk_score": 0.2}
    
    def has_common_errors(self, content: str) -> bool:
        """Detect common error patterns"""
        # Check for suspicious patterns
        error_indicators = [
            r"=\s*0\s*$",  # Result is zero (often wrong)
            r"undefined",  # Undefined operations
            r"infinity",   # Infinity results
            r"cannot",    # Cannot solve
        ]
        
        for pattern in error_indicators:
            if re.search(pattern, content, re.IGNORECASE):
                return True
        return False
    
    def rollout(self, state_id, plan):
        """Improved rollout with quality-based risk assessment"""
        # Calculate base risk from state depth
        depth = 0
        current = state_id
        while current in self.states and depth < 1000:
            parent = self.states[current].get("parent")
            if not parent or parent == current:
                break
            depth += 1
            current = parent
        
        # Base risk: start higher for first round, decrease with depth (more revisions = more thought)
        # Stricter base risk to improve accuracy
        round_idx = plan.get("round", 0) if isinstance(plan, dict) else 0
        if round_idx == 0:
            base_risk = 0.5  # High risk for first round (draft only, no review yet)
        elif round_idx == 1:
            base_risk = 0.30  # Higher risk after first review (was 0.25, too lenient)
        else:
            base_risk = 0.20 + (0.01 * depth)  # Higher base risk after multiple rounds (was 0.15)
        
        # Quality-based risk adjustment
        if isinstance(plan, dict):
            content = plan.get("content", "")
            pattern = self.extract_reasoning_pattern(content)
            
            # Adjust risk based on historical pattern performance
            total_attempts = self.success_patterns[pattern] + self.failure_patterns[pattern]
            if total_attempts > 5:
                failure_rate = self.failure_patterns[pattern] / total_attempts
                base_risk += failure_rate * 0.3  # Increase risk if pattern has high failure rate
            
            # Check for error indicators
            if self.has_common_errors(content):
                base_risk += 0.3
            
            # Analyze reviewers' feedback quality (if available)
            reviews = plan.get("reviews", [])
            if reviews:
                positive_feedback_count = 0
                negative_feedback_count = 0
                total_confidence = 0.0
                
                for review_str in reviews:
                    review_lower = str(review_str).lower()
                    # Count positive/negative indicators
                    positive_indicators = ["correct", "right", "accurate", "good", "agree", "valid", "acceptable"]
                    negative_indicators = ["wrong", "incorrect", "error", "mistake", "invalid", "disagree", "issue", "problem"]
                    
                    pos_count = sum(1 for ind in positive_indicators if ind in review_lower)
                    neg_count = sum(1 for ind in negative_indicators if ind in review_lower)
                    
                    if pos_count > neg_count:
                        positive_feedback_count += 1
                    elif neg_count > pos_count:
                        negative_feedback_count += 1
                    
                    # Extract confidence score
                    confidence_match = re.search(r"reliability score[:\s]+([\d.]+)", str(review_str), re.IGNORECASE)
                    if confidence_match:
                        total_confidence += float(confidence_match.group(1))
                
                n_reviews = len(reviews)
                if n_reviews > 0:
                    avg_confidence = total_confidence / n_reviews if total_confidence > 0 else 0.5
                    consensus_ratio = positive_feedback_count / n_reviews
                    
                    # ========================================
                    # 🔥 升级方向 1：引入"不确定性信号"
                    # ========================================
                    # disagreement_ratio = 1 - consensus_ratio
                    # 意义：reviewer 不一致 = 结构性不稳定
                    disagreement_ratio = 1.0 - consensus_ratio
                    base_risk += disagreement_ratio * 0.25  # Strong signal: disagreement increases risk
                    
                    # Keep existing consensus-based adjustments (but now with disagreement_ratio as base)
                    # Additional fine-tuning based on consensus level
                    if consensus_ratio < 0.5:  # Less than half agree
                        base_risk += 0.10  # Additional risk (reduced from 0.30 since disagreement_ratio already added)
                    elif consensus_ratio < 0.7:  # Moderate agreement
                        base_risk += 0.05  # Small additional risk
                    elif consensus_ratio < 1.0:  # Strong but not unanimous
                        base_risk += 0.02  # Very small risk increase
                    else:  # Unanimous agreement
                        base_risk -= 0.1  # Lower risk for unanimous agreement
                    
                    # Stricter adjustment based on average confidence
                    if avg_confidence < 0.6:
                        base_risk += 0.20  # Higher risk for low confidence
                    elif avg_confidence < 0.7:
                        base_risk += 0.10  # Medium risk for moderate confidence
                    elif avg_confidence > 0.85:
                        base_risk -= 0.15  # Lower risk for very high confidence
                    else:  # 0.7-0.85: good confidence
                        base_risk -= 0.05  # Small risk reduction
        
        # Normalize risk
        risk = min(1.0, max(0.0, base_risk))
        
        return {
            "risk_score": risk,
            "predicted_outcome": "likely_ok" if risk < 0.3 else "likely_error",
            "confidence": 1.0 - abs(risk - 0.5) * 2  # Higher confidence when risk is extreme
        }
    
    def update_from_result(self, user_query: str, answer_content: str, was_correct: bool):
        """Update learning from actual results"""
        pattern = self.extract_reasoning_pattern(answer_content)
        problem_type = self.extract_problem_signature(user_query)
        
        if was_correct:
            self.success_patterns[pattern] += 1
            # Update best strategy for this problem type
            if problem_type not in self.problem_types:
                self.problem_types[problem_type] = {"pattern": pattern, "success_rate": 1.0, "count": 1}
            else:
                current = self.problem_types[problem_type]
                if current["pattern"] == pattern:
                    current["count"] += 1
                    current["success_rate"] = current["count"] / (current["count"] + self.failure_patterns.get(pattern, 0))
                else:
                    # Check if this pattern is better
                    new_success_rate = self.success_patterns[pattern] / (self.success_patterns[pattern] + self.failure_patterns[pattern])
                    if new_success_rate > current["success_rate"]:
                        self.problem_types[problem_type] = {"pattern": pattern, "success_rate": new_success_rate, "count": 1}
        else:
            self.failure_patterns[pattern] += 1
        
        # Save periodically
        if (self.success_patterns[pattern] + self.failure_patterns[pattern]) % 10 == 0:
            self.save_history()
    
    def get_suggested_strategy(self, user_query: str) -> str:
        """Get suggested reasoning strategy based on problem type"""
        problem_type = self.extract_problem_signature(user_query)
        
        if problem_type in self.problem_types:
            best_pattern = self.problem_types[problem_type]["pattern"]
            success_rate = self.problem_types[problem_type]["success_rate"]
            
            strategy_map = {
                "step_by_step": "Break down the problem into clear steps",
                "direct_calculation": "Perform direct calculations",
                "variable_assignment": "Use variables to represent unknowns",
                "conditional_reasoning": "Consider different cases or conditions",
                "general": "Use general reasoning approach"
            }
            
            suggestion = strategy_map.get(best_pattern, "Use a structured approach")
            return f"Suggested strategy (success rate: {success_rate:.1%}): {suggestion}"
        
        return ""
    
    def get_state_summary(self, state_id: str, max_chars: int = 800) -> str:
        """Get enhanced state summary with learning insights and quality filtering"""
        state = self.get_state(state_id)
        if not state:
            return ""
        
        action = state.get("action", {})
        content = action.get("content", "")
        
        # Quality filtering: only include high-quality information in shared state
        # Check if this state has high confidence (based on validation)
        action_type = action.get("type", "")
        
        # For centralized mode, filter out low-quality states
        if self.mode == "centralized":
            # Check if content has error indicators
            if self.has_common_errors(content):
                # Don't share error-prone content, return minimal summary
                return f"[State {state_id[:8]}...] Previous step had potential issues. Please review carefully."
            
            # Check pattern quality
            pattern = self.extract_reasoning_pattern(content)
            total_attempts = self.success_patterns[pattern] + self.failure_patterns[pattern]
            if total_attempts > 5:
                failure_rate = self.failure_patterns[pattern] / total_attempts
                if failure_rate > 0.6:  # High failure rate pattern
                    # Return warning instead of full content
                    return f"[State {state_id[:8]}...] Using '{pattern}' approach (high failure rate: {failure_rate:.1%}). Consider alternative methods."
        
        # Add learning insights if available
        if state.get("parent") is None:  # Initial state
            # Could add problem type analysis here
            pass
        
        if len(content) > max_chars:
            content = content[:max_chars] + "..."
        
        return content
    
    def get_state(self, state_id):
        return self.states.get(state_id, {})
    
    def gc(self):
        """Garbage collection"""
        if len(self.states) > 50:
            self.states = dict(list(self.states.items())[-20:])
