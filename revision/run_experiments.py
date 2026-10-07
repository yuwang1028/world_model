"""
Unified experiment runner for the revision.

Every method is evaluated on the same fixed question set of each dataset, and every
run is appended to
    revision/records/<model>/<dataset>/<method>_rep<k>.jsonl
so that all tables in the paper are generated from a single source of truth
(see revision/analyze.py). Runs are resumable: finished question ids are skipped.

Usage (from the repository root):
    python3 -m revision.run_experiments --model gpt-3.5-turbo --methods main --reps 2
"""
import argparse
import json
import os
import random
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from pipelines import PipelineRunner
from utils import extract_answer, extract_pred_answer, is_correct, load_data, most_frequent_element
from revision.baselines import run_agreement_gate, run_mad, run_reconcile

ROOT = os.path.dirname(os.path.abspath(__file__))

# dataset name -> (task, mmlu_subject, file with the original fixed question ids)
DATASETS = {
    "gsm": ("gsm", None, "data/gsm/question_ids.txt"),
    "svamp": ("svamp", None, "data/svamp/question_ids.txt"),
    "gsm_hard": ("gsm_hard", None, "data/gsm_hard/question_ids.txt"),
    "ciar": ("ciar", None, "data/ciar/question_ids.txt"),
    "stg": ("stg", None, "data/stg/question_ids.txt"),
    "mmlu_moral": ("mmlu", "moral_scenarios", "data/mmlu/moral_scenarios_question_ids.txt"),
    "mmlu_law": ("mmlu", "professional_law", "data/mmlu/professional_law_question_ids.txt"),
    "mmlu_elementary": ("mmlu", "elementary_mathematics", "data/mmlu/elementary_mathematics_question_ids.txt"),
}

MARS_METHODS = {
    # method -> (use_world_model, wm_mode, disabled components)
    "baseline": (False, "centralized", ()),
    "peer": (True, "peer", ()),
    "centralized": (True, "centralized", ()),
    "abl_no_state": (True, "centralized", ("state",)),
    "abl_no_validation": (True, "centralized", ("validation",)),
    "abl_no_risk": (True, "centralized", ("risk",)),
    "abl_no_gate": (True, "centralized", ("gate",)),
    # blind reviewers (never see the draft) with and without the world-model gate
    "blind_ungated": (False, "centralized", ()),
    "blind_wm": (True, "centralized", ()),
    # blind reviewers + blind meta-reviewer (judge never sees the candidate), with/without evidence-preserving repair
    "blind_judge": (False, "centralized", ()),
    "blind_judge_evidence": (False, "centralized", ()),
    "blind_judge_wm_evidence": (True, "centralized", ()),
}
BASELINE_METHODS = ["cot", "sc", "self_reflect", "debate", "mad", "reconcile"]
GROUPS = {
    "main": ["baseline", "peer", "centralized"],
    "ablation": ["abl_no_state", "abl_no_validation", "abl_no_risk", "abl_no_gate"],
    "baselines": BASELINE_METHODS,
}

N_QUESTIONS = 100
HETERO_REVIEWER = "gpt-4o-mini"  # reviewer backbone for the heterogeneous-collective variants
N_REVIEWERS = 2
MAX_ROUNDS = 3


def get_question_ids(dataset, n_total):
    """Original fixed ids first, then extended (seeded) up to N_QUESTIONS."""
    path = os.path.join(ROOT, "question_ids.json")
    table = json.load(open(path)) if os.path.exists(path) else {}
    if dataset not in table:
        ids = [int(x) for x in np.atleast_1d(np.loadtxt(DATASETS[dataset][2]))]
        ids = list(dict.fromkeys(ids))[:N_QUESTIONS]
        rest = [i for i in range(n_total) if i not in set(ids)]
        rng = random.Random(2026)
        ids += sorted(rng.sample(rest, min(len(rest), N_QUESTIONS - len(ids))))
        table[dataset] = ids
        json.dump(table, open(path, "w"), indent=1)
    return table[dataset]


def run_one(dataset, method, model, question):
    task = DATASETS[dataset][0]
    out = {}
    if method in MARS_METHODS:
        use_wm, wm_mode, disable = MARS_METHODS[method]
        runner = PipelineRunner(task=task, model=model, use_world_model=use_wm, wm_mode=wm_mode, disable=disable,
                                blind_reviewers=method.startswith("blind_"),
                                blind_judge=method.startswith("blind_judge"),
                                evidence_feedback=method.endswith("_evidence"))
        hist = runner.run_mars_pipeline(question, n_reviewers=N_REVIEWERS, max_rounds=MAX_ROUNDS)
        hist.pop("_world_model", None)
        out["pred"] = extract_pred_answer(hist["final_answer"], task)
        out["draft_pred"] = extract_pred_answer(hist["rounds"][0]["input_answer"], task)
        for k in ["total_tokens", "prompt_tokens", "completion_tokens", "n_llm_calls", "final_status",
                  "final_answer", "rounds", "draft_validation"]:
            if k in hist:
                out[k] = hist[k]
        return out

    runner = PipelineRunner(task=task, model=model)
    if method == "cot":
        from custom_agents import create_author_agent
        agent = create_author_agent(model=model)
        resp = agent.run(runner.templates.construct_initial_prompt(question))
        out.update(pred=extract_pred_answer(resp, task), trace=resp, total_tokens=agent.total_tokens,
                   n_llm_calls=len(agent.token_log),
                   prompt_tokens=sum(c["prompt_tokens"] for c in agent.token_log),
                   completion_tokens=sum(c["completion_tokens"] for c in agent.token_log))
    elif method == "sc":
        h = runner.run_self_consistency_pipeline(question, num_samples=5)
        answers = [extract_pred_answer(r, task) for r in h["responses"]]
        out.update(pred=most_frequent_element([a for a in answers if a is not None]), trace=h["responses"],
                   total_tokens=h["total_tokens"], n_llm_calls=5)
    elif method == "self_reflect":
        h = runner.run_self_reflection_pipeline(question)
        pred = extract_pred_answer(h["reflection"], task)
        if pred is None:
            pred = extract_pred_answer(h["response"], task)
        out.update(pred=pred, trace=[h["response"], h["reflection"]], total_tokens=h["total_tokens"], n_llm_calls=2)
    elif method == "debate":
        histories, total_tokens = runner.run_debate_pipeline(question, num_agents=3, num_rounds=2)
        finals = [h[-1]["content"] for h in histories]
        answers = [extract_pred_answer(r, task) for r in finals]
        out.update(pred=most_frequent_element([a for a in answers if a is not None]), trace=finals,
                   total_tokens=total_tokens, n_llm_calls=6)
    elif method == "mad":
        out.update(run_mad(task, question, model=model))
    elif method == "reconcile":
        out.update(run_reconcile(task, question, model=model))
    elif method == "agreement_gate":
        out.update(run_agreement_gate(task, question, model=model))
    elif method == "agree_blind":
        out.update(run_agreement_gate(task, question, model=model, blind=True))
    elif method == "agree_hetero":
        out.update(run_agreement_gate(task, question, model=model, reviewer_model=HETERO_REVIEWER))
    elif method == "agree_blind_hetero":
        out.update(run_agreement_gate(task, question, model=model, blind=True, reviewer_model=HETERO_REVIEWER))
    else:
        raise ValueError(method)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--methods", nargs="+", required=True, help="method names or groups: main ablation baselines")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="only the first K questions (pilot runs)")
    ap.add_argument("--start", type=int, default=0, help="skip the first K questions (held-out development split)")
    args = ap.parse_args()

    methods = []
    for m in args.methods:
        methods += GROUPS.get(m, [m])

    lock = threading.Lock()
    jobs = []
    for dataset in args.datasets:
        task, subject, _ = DATASETS[dataset]
        questions = load_data(task, mmlu_subject=subject) if task == "mmlu" else load_data(task)
        qids = get_question_ids(dataset, len(questions))
        qids = qids[args.start:]
        if args.limit:
            qids = qids[: args.limit]
        for method in methods:
            for rep in range(args.reps):
                out_dir = os.path.join(ROOT, "records", args.model, dataset)
                os.makedirs(out_dir, exist_ok=True)
                path = os.path.join(out_dir, f"{method}_rep{rep}.jsonl")
                done = set()
                if os.path.exists(path):
                    done = {json.loads(l)["id"] for l in open(path) if l.strip()}
                for qid in qids:
                    if qid not in done:
                        jobs.append((dataset, method, rep, qid, questions[qid], path))
    random.Random(0).shuffle(jobs)
    print(f"{len(jobs)} runs to do ({args.model}; methods={methods})", flush=True)

    def work(job):
        dataset, method, rep, qid, q, path = job
        task = DATASETS[dataset][0]
        gt = extract_answer(q["answer"], task)
        last_err = None
        for attempt in range(3):
            try:
                t0 = time.time()
                out = run_one(dataset, method, args.model, q["question"])
                out["time_s"] = time.time() - t0
                break
            except Exception as e:  # transient API / parsing failures: retry the whole run
                last_err = f"{type(e).__name__}: {e}"
                time.sleep(5 * (attempt + 1))
        else:
            print(f"FAILED {dataset}/{method}/{qid}: {last_err}", flush=True)
            return None
        try:
            correct = bool(is_correct(out["pred"], gt, task))
        except Exception:
            correct = False
        rec = {"id": qid, "dataset": dataset, "method": method, "model": args.model, "rep": rep,
               "correct": int(correct), "gt": gt, **out}
        if "draft_pred" in rec:
            try:
                rec["draft_correct"] = int(bool(is_correct(rec["draft_pred"], gt, task)))
            except Exception:
                rec["draft_correct"] = 0
        with lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, default=float) + "\n")
        return rec

    t_start = time.time()
    n_done = n_fail = tokens = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(work, j) for j in jobs]
        for fut in as_completed(futures):
            try:
                rec = fut.result()
            except Exception:
                traceback.print_exc()
                rec = None
            if rec is None:
                n_fail += 1
            else:
                n_done += 1
                tokens += rec.get("total_tokens", 0)
            if (n_done + n_fail) % 200 == 0:
                el = time.time() - t_start
                print(f"[{el/60:.1f} min] done={n_done} failed={n_fail} tokens={tokens/1e6:.2f}M", flush=True)
    print(f"FINISHED done={n_done} failed={n_fail} tokens={tokens/1e6:.2f}M "
          f"time={(time.time()-t_start)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
