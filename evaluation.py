from pipelines import PipelineRunner
from utils import (
    extract_answer,
    extract_pred_answer,
    save_jsonl,
    load_data,
    extract_debate_answer,
    extract_pred_answer_majority,
    is_correct,
    most_frequent_element,
)
from tqdm import tqdm
import numpy as np
from datetime import date
import random
import time
import os


# ============================================================
# MARS Evaluation (Now Supports baseline / peer / centralized)
# ============================================================

def eval_mars(
    task="gsm",
    model=None,
    n_problems=5,
    n_reviewers=2,
    selected=True,
    voting=False,
    verbosity=0,
    mode="baseline",
    mmlu_subject: str | None = None,
):
    """
    mode:
        "baseline"
        "peer"
        "centralized"
    """

    # For MMLU, allow restricting to a single subject via mmlu_subject
    if task == "mmlu":
        all_questions = load_data(task=task, mmlu_subject=mmlu_subject)
    else:
        all_questions = load_data(task=task)
    mars_scores = []
    mars_scores_vote = []
    token_usages = []
    records = []
    rectified_collections = []

    if selected and not (task == "mmlu" and mmlu_subject is not None):
        # For generic tasks, reuse pre-saved question_ids.txt
        loaded = np.loadtxt(f"data/{task}/question_ids.txt")
        # Handle both scalar (single value) and array cases
        if loaded.ndim == 0:
            question_list = [int(loaded)]
        else:
            question_list = list(loaded.astype(int))
    else:
        # For new runs (or subject-specific MMLU), sample within current all_questions
        question_list = sorted(
            random.sample(range(len(all_questions)), n_problems)
        )
        save_dir = f"data/{task}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # For MMLU subject runs, keep subject-specific id files
        if task == "mmlu" and mmlu_subject is not None:
            ids_path = f"{save_dir}/{mmlu_subject}_question_ids.txt"
        else:
            ids_path = f"{save_dir}/question_ids.txt"
        np.savetxt(ids_path, question_list)

    start_time = time.time()
    
    # Setup save directory and filename (for incremental saving)
    date_str = date.today().isoformat()
    save_dir = f"data/{task}/records"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Per-subject filenames so multiple MMLU subjects on the same day do not overwrite.
    if task == "mmlu" and mmlu_subject is not None:
        save_name = f"{save_dir}/{mmlu_subject}_{mode}_{date_str}.jsonl"
    else:
        save_name = f"{save_dir}/{mode}_{date_str}.jsonl"
    # Clear the file at the start (in case of restart)
    if os.path.exists(save_name):
        os.remove(save_name)

    for i in tqdm(question_list):

        question = all_questions[i]["question"]
        gt_answer = all_questions[i]["answer"]

        if verbosity:
            print("question: ", question)
            print("gt_answer: ", gt_answer)
            print("===================")

        answer = extract_answer(gt_answer, task)

        # =============================
        # MODE SWITCH (Only Change Here)
        # =============================

        if mode == "baseline":
            runner = PipelineRunner(
                task=task,
                model=model,
                use_world_model=False
            )

        elif mode == "peer":
            runner = PipelineRunner(
                task=task,
                model=model,
                use_world_model=True,
                wm_mode="peer"
            )

        elif mode == "centralized":
            # Use v2 logic (risk + confidence + consensus)
            runner = PipelineRunner(
                task=task,
                model=model,
                use_world_model=True,
                wm_mode="centralized"
            )

        else:
            raise ValueError("Unknown mode")

        review_history = runner.run_mars_pipeline(
            question,
            n_reviewers=n_reviewers,
            max_rounds=3,
            verbosity=verbosity
        )

        # Extract initial draft
        initial_draft = review_history['rounds'][0]['input_answer']
        single_agent_answer = extract_pred_answer(initial_draft, task)

        # Extract final answer
        mars_final_content = review_history['final_answer']
        mars_answer = extract_pred_answer(mars_final_content, task)

        # Voting logic (unchanged)
        mars_answer_vote = None
        if voting:
            voting_dict = {}
            last_round = review_history['rounds'][-1]
            last_round_reviews = last_round['reviews']

            for idx, rev_item in enumerate(last_round_reviews):
                content = list(rev_item.values())[0]
                voting_dict[f"review{idx+1}"] = content

            voting_dict['author_response'] = review_history['rounds'][0]['input_answer']
            voting_dict['author_rebuttal'] = review_history['final_answer']
            voting_dict['meta_review'] = last_round['meta_review']

            mars_answer_vote = extract_pred_answer_majority(
                voting_dict,
                n_reviewers,
                task
            )
        else:
            mars_answer_vote = mars_answer

        # Scoring
        mars_scores.append(1 if is_correct(mars_answer, answer, task) else 0)
        mars_scores_vote.append(1 if is_correct(mars_answer_vote, answer, task) else 0)

        if not is_correct(single_agent_answer, answer, task) and \
           is_correct(mars_answer, answer, task):
            rectified_collections.append(i)

        review_history['id'] = int(i)
        review_history['mode'] = mode
        review_history['multi_score'] = int(mars_scores[-1])
        review_history['question'] = question
        review_history['gt_answer'] = gt_answer

        token_usages.append(review_history['total_tokens'])
        records.append(review_history)
        
        # Update world model learning from results
        if mode in ["peer", "centralized"]:
            try:
                # Get world model from review_history (stored in pipeline)
                wm = review_history.get("_world_model")
                if wm and hasattr(wm, 'update_from_result'):
                    wm.update_from_result(
                        question,
                        review_history.get('final_answer', ''),
                        mars_scores[-1] == 1  # was_correct
                    )
                # Clean up reference (don't save it to JSON)
                if "_world_model" in review_history:
                    del review_history["_world_model"]
            except Exception as e:
                if verbosity > 0:
                    print(f"Warning: Could not update world model: {e}")
                # Clean up reference even on error
                if "_world_model" in review_history:
                    del review_history["_world_model"]
        
        # Save incrementally after each question (for real-time monitoring)
        save_jsonl([review_history], save_name, append=True)

    end_time = time.time()
    avg_time = (end_time - start_time) / len(question_list)

    return (
        sum(mars_scores),
        sum(mars_scores_vote),
        rectified_collections,
        np.mean(token_usages),
        avg_time
    )


# ============================================================
# Single Agent (Unchanged)
# ============================================================

def eval_single_agent(task="gsm", model=None,
                      n_problems=5,
                      selected=True,
                      verbosity=0):

    all_questions = load_data(task=task)
    scores = []
    token_usages = []
    records = []
    hard_collections = []

    if selected:
        loaded = np.loadtxt(f"data/{task}/question_ids.txt")
        # Handle both scalar (single value) and array cases
        if loaded.ndim == 0:
            question_list = [int(loaded)]
        else:
            question_list = list(loaded.astype(int))
    else:
        question_list = sorted(
            random.sample(range(len(all_questions)), n_problems)
        )

    start_time = time.time()

    for i in tqdm(question_list):
        question = all_questions[i]["question"]
        gt_answer = all_questions[i]["answer"]

        answer = extract_answer(gt_answer, task)

        runner = PipelineRunner(task=task, model=model)
        agent_history = runner.run_single_agent_pipeline(
            question,
            verbosity=verbosity
        )

        pred_answer = extract_pred_answer(
            agent_history['response'],
            task
        )

        scores.append(1 if is_correct(pred_answer, answer, task) else 0)

        agent_history['id'] = int(i)
        agent_history['score'] = scores[-1]
        agent_history['question'] = question
        agent_history['gt_answer'] = gt_answer

        token_usages.append(agent_history['total_tokens'])
        records.append(agent_history)

    end_time = time.time()
    avg_time = (end_time - start_time) / len(question_list)

    return sum(scores), hard_collections, np.mean(token_usages), avg_time


def eval_single_agent_baseline(task="gsm", model=None, n_problems=5, selected=True, verbosity=0):
    all_questions = load_data(task=task)
    scores = []
    token_usages = []
    records = []  # record the full review process of each question
    hard_collections = []  # record the incorrectly answered questions
    if selected:
        question_list = list(np.loadtxt(f"data/{task}/question_ids.txt").astype(int))
    else:
        question_list = sorted(random.sample(range(len(all_questions)), n_problems))
        # question_list = range(len(all_questions))
        np.savetxt(f"data/{task}/question_ids.txt", question_list)
    start_time = time.time()
    for i in tqdm(question_list):
        question = all_questions[i]["question"]
        gt_answer = all_questions[i]["answer"]
        if verbosity:
            print("question: ", question)
            print("gt_answer: ", gt_answer)
            print("===================")
        answer = extract_answer(gt_answer, task)
        # Run single-agent pipeline
        runner = PipelineRunner(task=task, model=model)
        agent_history = runner.run_single_agent_pipeline(question, verbosity=verbosity)
        pred_answer = extract_pred_answer(agent_history['response'], task)
        if verbosity:
            print("GT answer and predicted answer: ", answer, pred_answer)
            print("\n")
        if pred_answer is not None and is_correct(pred_answer, answer, task):
            scores.append(int(1))
        else:
            scores.append(int(0))
            hard_collections.append(i)
        agent_history['id'] = int(i)
        agent_history['score'] = scores[-1]
        agent_history['question'] = question
        agent_history['gt_answer'] = gt_answer
        token_usages.append(agent_history['total_tokens'])
        records.append(agent_history)
    end_time = time.time()
    avg_time = (end_time - start_time) / len(question_list)
    date_str = date.today().isoformat()
    save_name = f"baselines/single_agent_logs/{task}_{date_str}.jsonl"
    save_jsonl(records, save_name)
    return sum(scores), hard_collections, np.mean(token_usages), avg_time


def eval_self_reflection(task="gsm", model=None, n_problems=5, selected=True, verbosity=0):
    all_questions = load_data(task=task)
    scores = []
    token_usages = []
    records = []  # record the full review process of each question
    hard_collections = []  # record the incorrectly answered questions
    if selected:
        question_list = list(np.loadtxt(f"data/{task}/question_ids.txt").astype(int))
    else:
        question_list = sorted(random.sample(range(len(all_questions)), n_problems))
    start_time = time.time()
    for i in tqdm(question_list):
        question = all_questions[i]["question"]
        gt_answer = all_questions[i]["answer"]
        if verbosity:
            print("question: ", question)
            print("gt_answer: ", gt_answer)
            print("===================")
        answer = extract_answer(gt_answer, task)
        # Run reflection pipeline
        runner = PipelineRunner(task=task, model=model)
        reflection_history = runner.run_self_reflection_pipeline(question, verbosity=verbosity)
        initial_answer = extract_pred_answer(reflection_history['response'], task)
        pred_answer = extract_pred_answer(reflection_history['reflection'], task)
        if verbosity:
            print("GT answer and predicted answer: ", answer, pred_answer)
        if pred_answer is not None and pred_answer == answer:
            scores.append(int(1))
        else:
            scores.append(int(0))
            hard_collections.append(i)
        reflection_history['id'] = int(i)
        reflection_history['score'] = scores[-1]
        token_usages.append(reflection_history['total_tokens'])
        records.append(reflection_history)
    end_time = time.time()
    avg_time = (end_time - start_time) / len(question_list)
    date_str = date.today().isoformat()
    save_name = f"baselines/reflection_logs/{task}_{date_str}.jsonl"
    save_jsonl(records, save_name)
    return sum(scores), hard_collections, np.mean(token_usages), avg_time


def eval_self_consistency(task="gsm", model=None, n_problems=5, n_samples=3, selected=True, verbosity=0):
    all_questions = load_data(task=task)
    scores = []
    token_usages = []
    records = []  # record the full review process of each question
    hard_collections = []  # record the incorrectly answered questions
    if selected:
        question_list = list(np.loadtxt(f"data/{task}/question_ids.txt").astype(int))
    else:
        question_list = sorted(random.sample(range(len(all_questions)), n_problems))
    start_time = time.time()
    for i in tqdm(question_list):
        question = all_questions[i]["question"]
        gt_answer = all_questions[i]["answer"]
        if verbosity:
            print("question: ", question)
            print("gt_answer: ", gt_answer)
            print("===================")
        answer = extract_answer(gt_answer, task)
        # Run self-consistency pipeline
        runner = PipelineRunner(task=task, model=model)
        consistency_history = runner.run_self_consistency_pipeline(question, num_samples=n_samples, verbosity=verbosity)
        sampled_answers = [extract_pred_answer(response, task) for response in consistency_history['responses']]
        pred_answer = most_frequent_element(sampled_answers)
        if verbosity:
            print("GT answer and predicted answer: ", answer, pred_answer)
            print("\n")
        if pred_answer is not None and is_correct(pred_answer, answer, task):
            scores.append(int(1))
        else:
            scores.append(int(0))
            hard_collections.append(i)
        consistency_history['id'] = int(i)
        consistency_history['score'] = int(scores[-1])
        consistency_history['question'] = question
        consistency_history['gt_answer'] = gt_answer
        token_usages.append(consistency_history['total_tokens'])
        records.append(consistency_history)
    end_time = time.time()
    avg_time = (end_time - start_time) / len(question_list)
    date_str = date.today().isoformat()
    save_name = f"baselines/consistency_logs/{task}_{date_str}.jsonl"
    save_jsonl(records, save_name)
    return sum(scores), hard_collections, np.mean(token_usages), avg_time


def eval_debate(task="gsm", model=None, n_problems=5, n_agents=3, selected=True, verbosity=0):
    all_questions = load_data(task=task)
    scores = []
    token_usages = []
    records = []  # record the full review process of each question
    hard_collections = []  # record the incorrectly answered questions
    if selected:
        question_list = list(np.loadtxt(f"data/{task}/question_ids.txt").astype(int))
    else:
        question_list = sorted(random.sample(range(len(all_questions)), n_problems))
    start_time = time.time()
    for i in tqdm(question_list):
        question = all_questions[i]["question"]
        gt_answer = all_questions[i]["answer"]
        if verbosity:
            print("question: ", question)
            print("gt_answer: ", gt_answer)
            print("===================")
        answer = extract_answer(gt_answer, task)
        # Run debate pipeline
        runner = PipelineRunner(task=task, model=model)
        debate_history, total_tokens = runner.run_debate_pipeline(question, num_agents=n_agents, verbosity=verbosity)
        pred_answer = extract_debate_answer(debate_history, task)  # need update !!!
        if verbosity:
            print("GT answer and predicted answer: ", answer, pred_answer)
        if pred_answer is not None and is_correct(pred_answer, answer, task=task):
            scores.append(int(1))
        else:
            scores.append(int(0))
            hard_collections.append(i)
        token_usages.append(total_tokens)
        records.append({"id": int(i), "score": int(scores[-1]), "total_tokens": total_tokens, "debate_history": debate_history})
    end_time = time.time()
    avg_time = (end_time - start_time) / len(question_list)
    date_str = date.today().isoformat()
    save_name = f"baselines/debate_logs/{task}_{date_str}.jsonl"
    save_jsonl(records, save_name)
    return sum(scores), hard_collections, np.mean(token_usages), avg_time