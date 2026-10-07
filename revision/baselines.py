"""
Additional multi-agent baselines for the revision experiments.

All baselines use the same backbone, decoding configuration and answer format as
the reviewer-style pipeline, so that they can be scored by the same extractor.

  - MAD        : Liang et al., "Encouraging Divergent Thinking in LLMs through
                 Multi-Agent Debate" (EMNLP 2024). Affirmative / negative debaters
                 with a judge that can stop the debate early.
  - ReConcile  : Chen et al., "ReConcile: Round-Table Conference Improves Reasoning
                 via Consensus among Diverse LLMs" (ACL 2024). Round-table
                 discussion with confidence-weighted voting. We use the
                 single-backbone variant (the original mixes different LLMs).
"""
import re
from collections import defaultdict

from custom_agents import create_author_agent
from prompt_templates import PromptBuilder
from utils import extract_pred_answer


def _is_math(task):
    return task in PromptBuilder(task=task).math_tasks


def _answer_format(task):
    if _is_math(task):
        return (
            "Thoughts: [your step-by-step computation process with immediate results]\n"
            "Answer: [the final numerical answer]\n\n"
            "Your final answer must be a single numerical number at the end of the response.\n"
        )
    return (
        "Thoughts: [Analysis of the question and options]\n"
        "Answer: [the final single capital letter answer in the form (X). X is chosen from [A,B,C,D]]\n\n"
        "Your final answer must be a single capital letter in the form (X) at the end of the response.\n"
    )


def _usage(agents):
    return {
        "total_tokens": sum(a.total_tokens for a in agents),
        "prompt_tokens": sum(c["prompt_tokens"] for a in agents for c in a.token_log),
        "completion_tokens": sum(c["completion_tokens"] for a in agents for c in a.token_log),
        "n_llm_calls": sum(len(a.token_log) for a in agents),
    }


def _key(ans):
    """Hashable key used to group equivalent answers when voting."""
    if isinstance(ans, float):
        return round(ans, 2)
    return ans


# ----------------------------------------------------------------------
# MAD: affirmative vs. negative debaters with a judge
# ----------------------------------------------------------------------
def run_mad(task, question, model=None, max_rounds=3):
    aff = create_author_agent(name="Affirmative", model=model)
    neg = create_author_agent(name="Negative", model=model)
    judge = create_author_agent(name="Judge", model=model)
    fmt = _answer_format(task)

    aff_hist = [{
        "role": "user",
        "content": (
            "You are a debater on the affirmative side. Solve the following problem:\n"
            f"{question}\n\n"
            "Give your reasoning and your answer in the following format:\n" + fmt
        ),
    }]
    aff_msg = aff.run(aff_hist)
    aff_hist.append(aff_msg)

    neg_hist = [{
        "role": "user",
        "content": (
            "You are a debater on the negative side. It is not necessary to fully agree with the "
            "other side, as our objective is to find the correct answer.\n"
            f"Problem: {question}\n\n"
            f"The affirmative side argues:\n```{aff_msg['content']}```\n\n"
            "You disagree with this answer. Provide your own answer and reasons in the following format:\n" + fmt
        ),
    }]
    neg_msg = neg.run(neg_hist)
    neg_hist.append(neg_msg)

    trace = [{"round": 1, "affirmative": aff_msg["content"], "negative": neg_msg["content"]}]
    final_text = None
    for r in range(1, max_rounds + 1):
        last = r == max_rounds
        judge_prompt = (
            "You are the judge of a debate between two debaters about the following problem.\n"
            f"Problem: {question}\n\n"
            f"Affirmative side:\n```{aff_msg['content']}```\n\n"
            f"Negative side:\n```{neg_msg['content']}```\n\n"
        )
        if last:
            judge_prompt += (
                "This is the final round. Decide which answer is correct (or give the correct one yourself).\n"
                "Respond in the following format:\n"
                "Preference: yes\nReason: [your reason]\n" + fmt
            )
        else:
            judge_prompt += (
                "Decide whether there is a clear preference for one answer. If there is, conclude the debate "
                "and give the final answer; otherwise the debate continues.\n"
                "Respond in the following format:\n"
                "Preference: [yes | no]\nReason: [your reason]\n"
                "If Preference is yes, also give the final answer as:\n" + fmt
            )
        verdict = judge.run(judge_prompt)
        trace[-1]["judge"] = verdict
        pref = re.search(r"Preference:\s*(yes|no)", verdict, re.IGNORECASE)
        if last or (pref and pref.group(1).lower() == "yes" and extract_pred_answer(verdict, task) is not None):
            final_text = verdict
            break

        aff_hist.append({
            "role": "user",
            "content": (
                f"The negative side argues:\n```{neg_msg['content']}```\n\n"
                "Do you agree with this perspective? Provide your reasons and your answer in the following format:\n" + fmt
            ),
        })
        aff_msg = aff.run(aff_hist)
        aff_hist.append(aff_msg)
        neg_hist.append({
            "role": "user",
            "content": (
                f"The affirmative side argues:\n```{aff_msg['content']}```\n\n"
                "Do you agree with this perspective? Provide your reasons and your answer in the following format:\n" + fmt
            ),
        })
        neg_msg = neg.run(neg_hist)
        neg_hist.append(neg_msg)
        trace.append({"round": r + 1, "affirmative": aff_msg["content"], "negative": neg_msg["content"]})

    pred = extract_pred_answer(final_text, task)
    if pred is None:
        pred = extract_pred_answer(aff_msg["content"], task)
    return {"pred": pred, "trace": trace, **_usage([aff, neg, judge])}


# ----------------------------------------------------------------------
# ReConcile: round-table discussion with confidence-weighted voting
# ----------------------------------------------------------------------
def _recalibrate(conf):
    # Confidence re-scaling used by ReConcile before weighted voting
    if conf >= 1.0:
        return 1.0
    if conf >= 0.9:
        return 0.8
    if conf >= 0.8:
        return 0.5
    if conf > 0.6:
        return 0.3
    return 0.1


def _parse_confidence(text):
    m = re.search(r"Confidence:\s*([01](?:\.\d+)?)", text, re.IGNORECASE)
    if not m:
        return 0.5
    return min(1.0, max(0.0, float(m.group(1))))


def _weighted_vote(answers, confs):
    weights = defaultdict(float)
    for ans, conf in zip(answers, confs):
        if ans is not None:
            weights[_key(ans)] += _recalibrate(conf)
    if not weights:
        return None
    return max(weights.items(), key=lambda kv: kv[1])[0]


def run_reconcile(task, question, model=None, n_agents=3, max_rounds=3):
    agents = [create_author_agent(name=f"Agent_{i+1}", model=model) for i in range(n_agents)]
    fmt = _answer_format(task) + "Confidence: [a number between 0.0 and 1.0: how likely your answer is correct]\n"

    histories = []
    for agent in agents:
        hist = [{
            "role": "user",
            "content": (
                "Please solve the following problem:\n"
                f"{question}\n\n"
                "Give your step-by-step reasoning, your answer, and your confidence in the following format:\n" + fmt
            ),
        }]
        hist.append(agent.run(hist))
        histories.append(hist)

    def collect():
        texts = [h[-1]["content"] for h in histories]
        # The confidence line follows the answer, so cut it off before extracting the answer
        answers = [extract_pred_answer(re.split(r"(?i)\n\s*confidence:", t)[0], task) for t in texts]
        confs = [_parse_confidence(t) for t in texts]
        return texts, answers, confs

    texts, answers, confs = collect()
    trace = [{"round": 0, "responses": texts, "answers": answers, "confidences": confs}]

    for r in range(1, max_rounds + 1):
        if len({_key(a) for a in answers}) == 1 and answers[0] is not None:
            break  # consensus reached
        groups = defaultdict(list)
        for i, (t, a, c) in enumerate(zip(texts, answers, confs)):
            groups[_key(a)].append((i, t, c))
        for i, agent in enumerate(agents):
            summary = "Carefully review the following solutions from all agents (grouped by answer):\n"
            for ans, members in groups.items():
                summary += f"\nThere are {len(members)} agent(s) who think the answer is {ans}.\n"
                for j, t, c in members:
                    who = "You" if j == i else f"Agent {j+1}"
                    summary += f"{who} (confidence {c:.2f}):\n```{t}```\n"
            summary += (
                "\nUsing these solutions and their confidences as additional information, reconsider the problem. "
                "Give your updated reasoning, answer and confidence in the following format:\n" + fmt
            )
            histories[i].append({"role": "user", "content": summary})
            histories[i].append(agent.run(histories[i]))
        texts, answers, confs = collect()
        trace.append({"round": r, "responses": texts, "answers": answers, "confidences": confs})

    pred = _weighted_vote(answers, confs)
    return {"pred": pred, "trace": trace, **_usage(agents)}


# ----------------------------------------------------------------------
# Exploratory variant (route-2 pilot): answer-agreement gate
# ----------------------------------------------------------------------
def _strip_commas(text):
    # "336,332,346" -> "336332346" so that thousands separators do not split a number
    return re.sub(r"(?<=\d),(?=\d{3})", "", text)


def _reviewer_answer(text, task):
    """Final answer of a reviewer's independent solution."""
    m = re.search(r"Final Answer:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    tail = m.group(1)
    if _is_math(task):
        return extract_pred_answer("Answer: " + _strip_commas(tail), task)
    m = re.search(r"\(?\b([ABCD])\b\)?", tail)
    return m.group(1) if m else None


def run_agreement_gate(task, question, model=None, n_reviewers=2, max_rounds=3, blind=False, reviewer_model=None):
    """
    Author + independent reviewers, no meta-reviewer.
    Gate: accept iff every reviewer's independently derived answer equals the author's.
    On disagreement the author receives the reviewers' full solutions and must reconcile them.
      blind=True          reviewers never see the author's draft (they simply solve the problem)
      reviewer_model=...  reviewers use a different backbone than the author
    """
    templates = PromptBuilder(task=task)
    author = create_author_agent(name="Author", model=model)
    reviewer_model = reviewer_model or model
    fmt = _answer_format(task)
    history = [templates.construct_author_prompt(question)]
    msg = author.run(history)
    history.append(msg)
    answer_text = msg["content"]
    draft_pred = extract_pred_answer(answer_text, task)

    reviewers_all, rounds = [], []
    final_pred = draft_pred
    for r in range(1, max_rounds + 1):
        final_pred = extract_pred_answer(answer_text, task)  # scored with the same extractor as every method
        cur = extract_pred_answer(_strip_commas(answer_text), task)  # used for the agreement test
        reviewers = [create_author_agent(name=f"Reviewer_{r}_{i+1}", model=reviewer_model) for i in range(n_reviewers)]
        reviewers_all += reviewers
        if blind:
            reviews = [rv.run(templates.construct_initial_prompt(question)) for rv in reviewers]
            rev_answers = [extract_pred_answer(_strip_commas(t), task) for t in reviews]
        else:
            reviews = [rv.run(templates.construct_reviewer_prompt(question, answer_text)) for rv in reviewers]
            rev_answers = [_reviewer_answer(t, task) for t in reviews]
        agree = cur is not None and all(a is not None and _key(a) == _key(cur) for a in rev_answers)
        rounds.append({"round": r, "author_answer": cur, "reviewer_answers": rev_answers, "accepted": bool(agree),
                       "input_answer": answer_text, "reviews": reviews})
        if agree:
            break
        if r == max_rounds:
            # no agreement: majority over the author's and reviewers' latest answers (author wins ties)
            votes = defaultdict(float)
            if cur is not None:
                votes[_key(cur)] += 1.01
            for a in rev_answers:
                if a is not None:
                    votes[_key(a)] += 1.0
            if votes:
                final_pred = max(votes.items(), key=lambda kv: kv[1])[0]
            break
        solutions = "\n\n".join(f"Independent solver {i+1}:\n```{t}```" for i, t in enumerate(reviews))
        history.append({
            "role": "user",
            "content": (
                "Independent solvers worked on the same problem from scratch and did not all reach your answer.\n\n"
                f"{solutions}\n\n"
                "Compare the solutions step by step, locate exactly where they diverge from yours, and decide which "
                "step is correct. Do not keep your previous answer unless you can point to a concrete error in the "
                "other solutions. Then give your answer in the following format:\n" + fmt
            ),
        })
        msg = author.run(history)
        history.append(msg)
        answer_text = msg["content"]
    return {"pred": final_pred, "draft_pred": draft_pred, "rounds": rounds, **_usage([author] + reviewers_all)}
