from custom_agents import create_author_agent, create_reviewer_agents, create_meta_reviewer_agent
from utils import extract_meta_decision, extract_pred_answer
from prompt_templates import PromptBuilder
import re

# =========================
# Optional World Model Layer
# =========================
# The world model is designed to be a *service layer* and should NOT replace the MARS pipeline.
# If you haven't created world_model/ yet, the fallback implementation below keeps baseline runnable.
try:
    # Recommended: create world_model/server.py with WorldModelServer class
    from world_model.server import WorldModelServer  # type: ignore
except Exception:  # pragma: no cover
    class WorldModelServer:
        """
        Minimal fallback world model (in-memory) so the pipeline keeps running.
        """
        def __init__(self, mode: str = "centralized", gc_ttl_states: int = 200, keep_last_k: int = 50):
            self.mode = mode
            self._states = {}
            self._order = []
            self._gc_ttl_states = gc_ttl_states
            self._keep_last_k = keep_last_k
            self._counter = 0

        def create_initial(self, user_query: str):
            return self.transition(None, {"type": "init", "content": user_query}, observation=None)

        def transition(self, parent_state_id, action, observation=None):
            sid = f"s{self._counter}"
            self._counter += 1
            node = {"parent": parent_state_id, "action": action, "observation": observation}
            self._states[sid] = node
            self._order.append(sid)
            # return pointer + light delta
            delta = {"type": action.get("type"), "len": len(action.get("content", ""))}
            return sid, delta

        def get_state_summary(self, state_id: str, max_chars: int = 800) -> str:
            node = self._states.get(state_id, {})
            action = node.get("action", {})
            txt = str(action.get("content", ""))[:max_chars]
            return f"state_id={state_id}; last_action_type={action.get('type')}; last_action_preview={txt}"

        def validate(self, state_id: str, action):
            txt = (action.get("content") or "").strip()
            if len(txt) < 5:
                return {"ok": False, "risk_score": 0.9, "reason": "Answer too short.", "suggestion": "Please provide a complete answer."}
            return {"ok": True, "risk_score": 0.2, "reason": "OK", "suggestion": ""}

        def rollout(self, state_id: str, plan):
            # placeholder: low risk by default
            return {"risk_score": 0.2, "predicted_outcome": "likely_ok"}

        def gc(self):
            # keep last K only
            if len(self._order) > self._keep_last_k:
                to_drop = self._order[:-self._keep_last_k]
                for sid in to_drop:
                    self._states.pop(sid, None)
                self._order = self._order[-self._keep_last_k:]


class PipelineRunner:
    def __init__(self, task="gsm", model=None, use_world_model: bool = False, wm_mode: str = "centralized"):
        self.task = task
        self.templates = PromptBuilder(task=task)
        self.model = model

        # World-model config
        self.use_world_model = use_world_model
        self.wm_mode = wm_mode  # "centralized" | "peer"

        # Centralized world model lives on the runner
        self.world_model = WorldModelServer(mode="centralized") if use_world_model and wm_mode == "centralized" else None
        
        # Task-adaptive configuration for Centralized mode
        self.task_configs = {
            "gsm": {
                "base_threshold": 0.28,
                "min_consensus": 2,
                "min_confidence": 0.65,
                "full_consensus_confidence": 0.80,
                "full_consensus_risk_cap": 0.45,
            },
            "svamp": {
                "base_threshold": 0.28,
                "min_consensus": 2,
                "min_confidence": 0.65,
                "full_consensus_confidence": 0.80,
                "full_consensus_risk_cap": 0.45,
            },
            "ciar": {
                "base_threshold": 0.35,  # More lenient for complex reasoning
                "min_consensus": 1,      # Lower consensus requirement
                "min_confidence": 0.60,  # Lower confidence threshold
                "full_consensus_confidence": 0.75,
                "full_consensus_risk_cap": 0.50,
            },
            "gpqa": {
                "base_threshold": 0.40,  # Much more lenient for expert knowledge
                "min_consensus": 1,       # Lower consensus requirement
                "min_confidence": 0.55,  # Lower confidence threshold
                "full_consensus_confidence": 0.70,
                "full_consensus_risk_cap": 0.55,
            },
            "mmlu": {
                "base_threshold": 0.38,  # Moderate leniency for undergraduate knowledge
                "min_consensus": 1,       # Lower consensus requirement
                "min_confidence": 0.58,  # Moderate confidence threshold
                "full_consensus_confidence": 0.72,
                "full_consensus_risk_cap": 0.52,
            },
            "mmlu_pro": {
                "base_threshold": 0.42,  # More lenient for advanced knowledge
                "min_consensus": 1,       # Lower consensus requirement
                "min_confidence": 0.55,  # Lower confidence threshold
                "full_consensus_confidence": 0.70,
                "full_consensus_risk_cap": 0.55,
            },
        }

    def run_mars_pipeline(self, user_query, n_reviewers=2, max_rounds=3, verbosity=0):
        # 1. Initialize Agents (keep MARS roles unchanged)
        author = create_author_agent(model=self.model)
        reviewers = create_reviewer_agents(n_reviewers, model=self.model)
        meta = create_meta_reviewer_agent(model=self.model)

        # 2. Setup History Tracking
        author_history = []
        reviewer_histories = [[] for _ in range(n_reviewers)]  # separate conversation history for EACH reviewer
        full_history = {
            "user_query": user_query,
            "rounds": [],
            "final_status": "unresolved"
        }

        # 2.1 Initialize world model(s)
        # Centralized: one shared server; Peer: one local server per agent (no shared ground truth)
        wm_author = None
        wm_reviewers = None
        wm_meta = None
        shared_wm = None
        state_id = None  # pointer carried between stages (centralized only)
        state_delta = None

        if self.use_world_model:
            if self.wm_mode == "centralized":
                shared_wm = self.world_model or WorldModelServer(mode="centralized")
                state_id, state_delta = shared_wm.create_initial(user_query), None
                # create_initial returns (sid, delta) in fallback; handle both
                if isinstance(state_id, tuple):
                    state_id, state_delta = state_id
            elif self.wm_mode == "peer":
                wm_author = WorldModelServer(mode="peer")
                wm_reviewers = [WorldModelServer(mode="peer") for _ in range(n_reviewers)]
                wm_meta = WorldModelServer(mode="peer")
                # each maintains its own local init
                _sid = wm_author.create_initial(user_query)
                if isinstance(_sid, tuple):
                    _sid, _ = _sid
                # we keep author's pointer only as a message payload (not a shared truth)
            else:
                raise ValueError(f"Unknown wm_mode={self.wm_mode}")

        # 3. Step 1: Author's First Draft
        author_input = self.templates.construct_author_prompt(user_query)
        
        # Add World Model strategy suggestion if available
        if self.use_world_model:
            try:
                if self.wm_mode == "centralized" and shared_wm and hasattr(shared_wm, 'get_suggested_strategy'):
                    strategy = shared_wm.get_suggested_strategy(user_query)
                    if strategy:
                        author_input += f"\n\n[SUGGESTED_STRATEGY]\n{strategy}\n"
                elif self.wm_mode == "peer" and wm_author and hasattr(wm_author, 'get_suggested_strategy'):
                    strategy = wm_author.get_suggested_strategy(user_query)
                    if strategy:
                        author_input += f"\n\n[SUGGESTED_STRATEGY]\n{strategy}\n"
            except:
                pass  # Fallback if method doesn't exist
        
        author_history.append(author_input)

        author_response = author.run(author_history)
        author_history.append(author_response)
        current_answer_content = author_response["content"]

        # ===== World model: transition + validate after draft =====
        author_state_summary = None
        if self.use_world_model:
            action = {"type": "draft", "content": current_answer_content}

            if self.wm_mode == "centralized":
                result = shared_wm.transition(state_id, action, observation=None)
                if isinstance(result, tuple):
                    state_id, state_delta = result
                else:
                    state_id = result
                    state_delta = None
                validation = shared_wm.validate(state_id, action)
                author_state_summary = shared_wm.get_state_summary(state_id)
            else:  # peer
                result = wm_author.transition(None, action, observation=None)
                if isinstance(result, tuple):
                    sid_a, delta_a = result
                else:
                    sid_a = result
                    delta_a = None
                validation = wm_author.validate(sid_a, action)
                author_state_summary = wm_author.get_state_summary(sid_a)
                # in peer mode, we pass only summary/pointer-like text to others (NOT shared state)
                author_response["state_summary"] = author_state_summary

            # Early correction hook (keeps pipeline intact; just adjusts answer content)
            if not validation.get("ok", True):
                if verbosity:
                    print(f"\n[WorldModel] Draft validation failed: {validation.get('reason')}")
                suggestion = validation.get("suggestion") or ""
                if suggestion:
                    current_answer_content = suggestion

        # Variable to store the author's rebuttal logic to show reviewers in the next round
        latest_rebuttal_logic = "This is the initial answer."
        last_round_decision = "wrong"
        if verbosity:
            print(f"\n=== Round 0: Author's Initial Answer ===\n{current_answer_content}")

        # 4. The Review Loop
        for round_idx in range(1, max_rounds + 1):
            if verbosity:
                print(f"\n\n--- Starting Review Round {round_idx} ---")

            round_data = {
                "round": round_idx,
                "input_answer": current_answer_content,
                "reviews": [],
                "meta_review": None
            }

            # --- A. Reviewers Critique (Stateful) ---
            review_responses = []
            for i, reviewer in enumerate(reviewers):
                # Construct the prompt based on whether it is the first round or a follow-up
                if round_idx == 1:
                    prompt_content = self.templates.construct_reviewer_prompt(user_query, current_answer_content)

                    # Append world state pointer/summary (NO need to change prompt_templates.py)
                    if self.use_world_model:
                        if self.wm_mode == "centralized":
                            state_hint = (shared_wm.get_state_summary(state_id) if state_id else "")
                        else:
                            state_hint = (author_response.get("state_summary") or author_state_summary or "")
                        prompt_content += f"\n\n[WORLD_STATE]\n{state_hint}\n"

                    reviewer_histories[i] = [{"role": "user", "content": prompt_content}]
                else:
                    prev_status = last_round_decision
                    update_msg_dict = self.templates.construct_reviewer_update_prompt(
                        previous_status=prev_status,
                        author_logic=latest_rebuttal_logic,
                        current_answer=current_answer_content
                    )
                    if self.use_world_model:
                        if self.wm_mode == "centralized":
                            state_hint = (shared_wm.get_state_summary(state_id) if state_id else "")
                        else:
                            state_hint = (author_response.get("state_summary") or author_state_summary or "")
                            wm_reviewers[i].transition(None, {"type": "sync_from_author", "content": state_hint}, observation=None)
                        update_msg_dict["content"] += f"\n\n[WORLD_STATE]\n{state_hint}\n"
                    reviewer_histories[i].append(update_msg_dict)

                # RUN the reviewer
                review_response_dict = reviewer.run(reviewer_histories[i])
                review_content = review_response_dict["content"]

                confidence_score = review_response_dict.get("confidence_score", 0.5)
                formatted_review_str = (
                    f"Reliability Score: {confidence_score:.2f} / 1.0\n"
                    f"{review_content}"
                )
                review_responses.append(formatted_review_str)

                reviewer_histories[i].append(review_response_dict)

                round_data["reviews"].append({f"reviewer_{i+1}": formatted_review_str})
                if verbosity:
                    print(f"\n[Round {round_idx}] {reviewer.name}:\n{review_content}")

            # --- B. Meta-Reviewer Decision ---
            combined_reviews = "\n\n".join(
                [f"{reviewers[i].name}:\n{review_responses[i]}" for i in range(len(reviewers))]
            )

            meta_input = self.templates.construct_meta_prompt(user_query, current_answer_content, combined_reviews)

            # World-model rollout / risk estimation before meta decision
            risk = 0.5  # Default risk if no world model
            reviewers_consensus = 0  # Count of reviewers who agree (positive feedback)
            
            if self.use_world_model:
                if self.wm_mode == "centralized":
                    # Pass round_idx and reviews to rollout for better risk assessment
                    rollout = shared_wm.rollout(
                        state_id, 
                        plan={
                            "type": "evaluate_accept", 
                            "content": current_answer_content,
                            "round": round_idx,
                            "reviews": review_responses
                        }
                    )
                    risk = rollout.get("risk_score", 0.5)
                    meta_input += f"\n\n[WORLD_MODEL_ROLLOUT]\nRiskScore={risk:.3f}\n"
                else:
                    # peer meta uses its own reconstructed local state
                    sync_sid = wm_meta.transition(None, {"type": "sync_reviews", "content": combined_reviews}, observation=None)
                    rollout = wm_meta.rollout(sync_sid, plan={"type": "evaluate_accept", "content": current_answer_content})
                    risk = rollout.get("risk_score", 0.5) if isinstance(rollout, dict) else 0.5
                    meta_input += f"\n\n[WORLD_MODEL_ROLLOUT]\nPeerRiskScore={risk:.3f}\n"
            
            # Analyze reviewers' consensus (count positive feedback)
            for review_str in review_responses:
                review_lower = review_str.lower()
                # Count positive indicators (expanded list)
                positive_indicators = ["correct", "right", "accurate", "good", "agree", "valid", "acceptable", 
                                      "looks good", "seems correct", "appears right", "satisfactory", "fine"]
                negative_indicators = ["wrong", "incorrect", "error", "mistake", "invalid", "disagree", "issue", 
                                      "problem", "concern", "issue", "needs", "should", "incorrectly"]
                positive_count = sum(1 for ind in positive_indicators if ind in review_lower)
                negative_count = sum(1 for ind in negative_indicators if ind in review_lower)
                # Extract confidence score if available
                confidence_match = re.search(r"reliability score[:\s]+([\d.]+)", review_str, re.IGNORECASE)
                confidence = float(confidence_match.group(1)) if confidence_match else 0.5
                
                # Reviewer agrees if: (positive > negative) OR (confidence > 0.7) OR (positive_count >= 2)
                # More lenient consensus detection
                if (positive_count > negative_count) or (confidence > 0.7) or (positive_count >= 2):
                    reviewers_consensus += 1

            meta_decision = meta.run(meta_input)
            round_data["meta_review"] = meta_decision

            if verbosity:
                print(f"\n[Round {round_idx}] Meta-Reviewer Decision:\n{meta_decision}")
                print(f"[Round {round_idx}] Risk Score: {risk:.3f}, Reviewers Consensus: {reviewers_consensus}/{n_reviewers}")

            # --- C. Check Decision (Centralized V2 Logic) ---
            decision = extract_meta_decision(meta_decision)
            can_accept = False
            last_round_decision = decision.lower()
            round_data["decision"] = last_round_decision

            if decision.lower() == "right":
                if self.use_world_model and self.wm_mode == "centralized":
                    # ========================================
                    # Centralized V2: Meta decision language analysis
                    # ========================================
                    # Use meta decision text to adjust risk
                    meta_decision_lower = str(meta_decision).lower()
                    meta_risk_adjustment = 0.0
                    
                    # Negative signals (increase risk)
                    if "major flaw" in meta_decision_lower or "serious error" in meta_decision_lower:
                        meta_risk_adjustment += 0.25
                    elif "significant issue" in meta_decision_lower or "critical problem" in meta_decision_lower:
                        meta_risk_adjustment += 0.20
                    elif "minor issue" in meta_decision_lower or "small problem" in meta_decision_lower:
                        meta_risk_adjustment += 0.10
                    elif "concern" in meta_decision_lower or "uncertain" in meta_decision_lower:
                        meta_risk_adjustment += 0.05
                    
                    # Positive signals (decrease risk)
                    if "clearly correct" in meta_decision_lower or "definitely right" in meta_decision_lower:
                        meta_risk_adjustment -= 0.15
                    elif "well reasoned" in meta_decision_lower or "sound logic" in meta_decision_lower:
                        meta_risk_adjustment -= 0.10
                    elif "accurate" in meta_decision_lower and "completely" in meta_decision_lower:
                        meta_risk_adjustment -= 0.08
                    
                    # Apply meta_decision risk adjustment to original risk
                    risk_after_meta = risk + meta_risk_adjustment
                    risk_after_meta = min(1.0, max(0.0, risk_after_meta))  # Clamp to [0, 1]

                    # ===============================
                    # 1. Calculate average reviewer confidence
                    # ===============================
                    avg_confidence = 0.5
                    if review_responses:
                        total_conf = 0.0
                        conf_count = 0
                        for review_str in review_responses:
                            confidence_match = re.search(
                                r"reliability score[:\s]+([\d.]+)",
                                str(review_str),
                                re.IGNORECASE
                            )
                            if confidence_match:
                                total_conf += float(confidence_match.group(1))
                                conf_count += 1
                        if conf_count > 0:
                            avg_confidence = total_conf / conf_count

                    # ===============================
                    # 2. Complexity estimation (affects threshold only)
                    # ===============================
                    answer_length = len(current_answer_content.split())
                    complexity_factor = min(0.12, answer_length / 900.0)

                    # Task-adaptive base threshold
                    task_config = self.task_configs.get(self.task, self.task_configs["gsm"])
                    base_threshold = task_config["base_threshold"]
                    risk_threshold = base_threshold + complexity_factor

                    # ===============================
                    # 3. Effective risk (combining WM risk and reviewer confidence)
                    # ===============================
                    effective_risk = 0.7 * risk_after_meta + 0.3 * (1 - avg_confidence)

                    # Task-adaptive consensus and confidence requirements
                    min_consensus = min(task_config["min_consensus"], n_reviewers)
                    full_consensus = (reviewers_consensus == n_reviewers)

                    # ===============================
                    # 4. Case A: Regular acceptance
                    # ===============================
                    if (
                        effective_risk <= risk_threshold and
                        reviewers_consensus >= min_consensus and
                        avg_confidence > task_config["min_confidence"]
                    ):
                        can_accept = True

                    # ===============================
                    # 5. Case B: Full consensus override (with risk cap)
                    # ===============================
                    elif (
                        full_consensus and
                        avg_confidence > task_config["full_consensus_confidence"] and
                        effective_risk <= task_config["full_consensus_risk_cap"]
                    ):
                        can_accept = True
                    
                    # Debug output
                    if verbosity:
                        print(
                            f"[Centralized-v2] decision={decision}, "
                            f"risk={risk:.3f}, meta_adj={meta_risk_adjustment:+.3f}, "
                            f"risk_after_meta={risk_after_meta:.3f}, effective_risk={effective_risk:.3f}, "
                            f"threshold={risk_threshold:.3f}, "
                            f"consensus={reviewers_consensus}/{n_reviewers}, "
                            f"confidence={avg_confidence:.2f}, "
                            f"accept={can_accept}"
                        )
                else:
                    # Baseline or Peer mode: original behavior
                    can_accept = True

            if can_accept:
                full_history["final_status"] = "accepted"
                full_history["rounds"].append(round_data)
                if verbosity:
                    print(f"\n=== Answer Accepted in Round {round_idx} ===")
                break

            # --- D. Feedback & Revision (Prepare for next round) ---
            if round_idx < max_rounds:
                feedback_input = self.templates.construct_feedback_prompt(meta_decision, status=last_round_decision)
                author_history.append(feedback_input)

                author_rebuttal = author.run(author_history)
                author_history.append(author_rebuttal)

                current_answer_content = author_rebuttal["content"]
                latest_rebuttal_logic = author_rebuttal["content"]

                # World model update on revision
                if self.use_world_model:
                    action = {"type": "revision", "content": current_answer_content, "meta_feedback": meta_decision}
                    if self.wm_mode == "centralized":
                        result = shared_wm.transition(state_id, action, observation={"round": round_idx})
                        if isinstance(result, tuple):
                            state_id, state_delta = result
                        else:
                            state_id = result
                            state_delta = None
                        validation = shared_wm.validate(state_id, action)
                        author_state_summary = shared_wm.get_state_summary(state_id)
                    else:
                        result = wm_author.transition(None, action, observation={"round": round_idx})
                        if isinstance(result, tuple):
                            sid_a, delta_a = result
                        else:
                            sid_a = result
                            delta_a = None
                        validation = wm_author.validate(sid_a, action)
                        author_state_summary = wm_author.get_state_summary(sid_a)
                        author_response["state_summary"] = author_state_summary

                    if not validation.get("ok", True) and validation.get("suggestion"):
                        current_answer_content = validation["suggestion"]

                if verbosity:
                    print(f"\n=== Author's Revised Answer (Round {round_idx}) ===\n{current_answer_content}")
            else:
                # If we are at the last round, just mark the final status based on the LAST decision
                if decision == "right":
                    full_history["final_status"] = "accepted"
                else:
                    full_history["final_status"] = "rejected"

            full_history["rounds"].append(round_data)

            # World model GC (end of round)
            if self.use_world_model:
                if self.wm_mode == "centralized":
                    shared_wm.gc()
                else:
                    wm_author.gc()
                    for wm_r in wm_reviewers:
                        wm_r.gc()
                    wm_meta.gc()

        # Final Token Calculation
        agents = [author, *reviewers, meta]
        total_tokens = sum(agent.total_tokens for agent in agents)
        full_history["total_tokens"] = total_tokens
        full_history["final_answer"] = current_answer_content
        
        # Store world model references for learning (if available)
        if self.use_world_model:
            if self.wm_mode == "centralized" and shared_wm:
                full_history["_world_model"] = shared_wm  # Store reference for learning
            elif self.wm_mode == "peer" and wm_author:
                full_history["_world_model"] = wm_author  # Store reference for learning

        # Optional: expose final state pointer for analysis
        if self.use_world_model and self.wm_mode == "centralized":
            full_history["final_state_id"] = state_id
        if self.use_world_model and self.wm_mode == "peer":
            full_history["peer_mode"] = True

        return full_history

    def run_single_agent_pipeline(self, user_query, verbosity=0):
        agent = create_author_agent(model=self.model)
        author_input = self.templates.construct_initial_prompt(user_query)
        response = agent.run(author_input)
        if verbosity:
            print("\n=== Agent Answer ===\n", response)
        agent_history = {"response": response, "total_tokens": agent.total_tokens}
        return agent_history

    def run_self_reflection_pipeline(self, user_query, verbosity=0):
        agent = create_author_agent(model=self.model)
        # Step 1: Initial answer
        author_input = self.templates.construct_initial_prompt(user_query)
        response = agent.run(author_input)
        if verbosity:
            print("\n=== Initial Answer ===\n", response)

        # Step 2: Self-reflection
        reflection_prompt = self.templates.construct_reflection_prompt(user_query, response)
        reflection = agent.run(reflection_prompt)
        if verbosity:
            print("\n=== Final answer after self-reflection ===\n", reflection)
        reflection_history = {"response": response, "reflection": reflection, "total_tokens": agent.total_tokens}
        return reflection_history

    def run_self_consistency_pipeline(self, user_query, num_samples=3, verbosity=0):
        agent = create_author_agent(model=self.model)
        responses = []
        for i in range(num_samples):
            author_input = self.templates.construct_initial_prompt(user_query)
            response = agent.run(author_input)
            responses.append(response)
            if verbosity:
                print(f"\n=== Sample {i+1} Answer ===\n", response)
        sc_history = {"responses": responses, "total_tokens": agent.total_tokens}
        return sc_history

    def run_debate_pipeline(self, user_query, num_agents=3, num_rounds=2, verbosity=0) -> list[list[dict]]:
        agents = [
            create_author_agent(name=f"Agent_{i+1}", model=self.model)
            for i in range(num_agents)
        ]
        agent_histories = [[] for _ in range(num_agents)]

        # Round 0: each agent answers independently
        for i in range(num_agents):
            prompt = self.templates.construct_debate_prompt([], user_query, response_idx=0)
            agent_histories[i].append(prompt)
            response = agents[i].run(agent_histories[i])
            agent_histories[i].append(response)
            if verbosity:
                print(f"\n=== Round 0 Agent {i+1} Answer ===\n", response["content"])

        # Rounds >= 1: agents revise based on others
        for r in range(1, num_rounds):
            for i in range(num_agents):
                other_histories = agent_histories[:i] + agent_histories[i+1:]
                prompt = self.templates.construct_debate_prompt(other_histories, user_query, response_idx=2*r - 1)

                agent_histories[i].append(prompt)
                response = agents[i].run(agent_histories[i])
                agent_histories[i].append(response)
                if verbosity:
                    print(f"\n=== Round {r} Agent {i+1} Answer ===\n", response["content"])
        total_tokens = sum(agent.total_tokens for agent in agents)

        return agent_histories, total_tokens  # List of message histories per agent
