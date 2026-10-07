"""
Single source of truth for every number in the revised paper.

Reads revision/records/<model>/<dataset>/<method>_rep<k>.jsonl and writes
  revision/results/*.csv          master spreadsheet + per-analysis tables
  revision/results/summary.json   scalar statistics quoted in the text
  paper/tables/*.tex              LaTeX table bodies (\\input by the paper)
  paper/figures/*.pdf             vector figures

Usage (from the repository root):  python3 -m revision.analyze
"""
import glob
import json
import os
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(ROOT)
REC = os.path.join(ROOT, "records")
OUT = os.path.join(ROOT, "results")
TAB = os.path.join(REPO, "paper", "tables")
FIG = os.path.join(REPO, "paper", "figures")
for d in (OUT, TAB, FIG):
    os.makedirs(d, exist_ok=True)

DATASETS = ["gsm", "svamp", "gsm_hard", "mmlu_elementary", "ciar", "stg", "mmlu_moral", "mmlu_law"]
DS_LABEL = {"gsm": "GSM8K", "svamp": "SVAMP", "gsm_hard": "GSM-Hard", "mmlu_elementary": "MMLU-Elem.",
            "ciar": "CIAR", "stg": "StrategyQA", "mmlu_moral": "MMLU-Moral", "mmlu_law": "MMLU-Law"}
MATH = {"gsm", "svamp", "gsm_hard", "ciar"}
MODELS = ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4.1-nano", "gpt-4.1-mini"]
ABL_MODEL = "gpt-4o-mini"  # backbone used for ablations and the cost table
# number of questions per dataset evaluated with each backbone (prefix of the fixed id list)
MODEL_N = {"gpt-3.5-turbo": 50, "gpt-4o-mini": 50, "gpt-4.1-mini": 50, "gpt-4.1-nano": 50}
MODEL_LABEL = {"gpt-3.5-turbo": "GPT-3.5-Turbo", "gpt-4o-mini": "GPT-4o-mini", "gpt-4.1-mini": "GPT-4.1-mini", "gpt-4.1-nano": "GPT-4.1-nano"}
MAIN = ["baseline", "peer", "centralized"]
ABLATIONS = ["centralized", "abl_no_state", "abl_no_validation", "abl_no_risk", "abl_no_gate"]
BASELINES = ["cot", "sc", "self_reflect", "debate", "mad", "reconcile"]
METHOD_LABEL = {
    "cot": "CoT (single agent)", "sc": "Self-Consistency (5)", "self_reflect": "Self-Reflection",
    "debate": "Multi-Agent Debate", "mad": "MAD", "reconcile": "ReConcile",
    "baseline": "Reviewer pipeline", "peer": "Ours (Peer)", "centralized": "Ours (Centralized)",
    "abl_no_state": "w/o shared state summary", "abl_no_validation": "w/o transition validation",
    "abl_no_risk": "w/o trajectory risk", "abl_no_gate": "w/o WM-guided acceptance",
}
RNG = np.random.default_rng(0)


# ----------------------------------------------------------------------
# Loading and per-task feature extraction
# ----------------------------------------------------------------------
def norm_answer(text, dataset):
    if text is None:
        return None
    text = str(text)
    if dataset in MATH:
        nums = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?%?", text)
        if not nums:
            return None
        tok = nums[-1].replace(",", "")
        try:
            val = float(tok.rstrip("%")) / (100 if tok.endswith("%") else 1)
        except ValueError:
            return None
        return round(val, 2)
    m = re.search(r"\(?\b([ABCD])\b\)?", text)
    return m.group(1) if m else None


def entropy(items):
    items = [x for x in items if x is not None]
    if len(items) < 2:
        return 0.0
    _, counts = np.unique([str(x) for x in items], return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum())


def task_features(rec):
    """CUI ingredients of one reviewer-pipeline run (first round + risk trajectory)."""
    rounds = rec.get("rounds") or []
    if not rounds:
        return {}
    ds = rec["dataset"]
    first = rounds[0]
    votes, confs, answers = [], [], [rec.get("draft_pred")]
    for rev in first.get("reviews", []):
        text = " ".join(str(v) for v in rev.values())
        m = re.search(r"Decision:\s*[\[\(\*]*\s*(right|wrong)", text, re.IGNORECASE)
        if m:
            votes.append(1 if m.group(1).lower() == "right" else 0)
        m = re.search(r"reliability score[:\s]+([\d.]+)", text, re.IGNORECASE)
        confs.append(float(m.group(1)) if m else 0.5)
        m = re.search(r"Final Answer:\s*(.+)", text, re.IGNORECASE)
        answers.append(norm_answer(m.group(1), ds) if m else None)
    m = re.search(r"Final Answer:\s*(.+)", str(first.get("meta_review", "")), re.IGNORECASE)
    answers.append(norm_answer(m.group(1), ds) if m else None)
    if isinstance(answers[0], float):
        answers[0] = round(answers[0], 2)
    p = float(np.mean(votes)) if votes else 1.0
    h_rev = 0.0 if p in (0.0, 1.0) else float(-(p * np.log(p) + (1 - p) * np.log(1 - p)))
    risks = [r["wm"]["risk_probe"] for r in rounds if "wm" in r]
    return {
        "H_rev": h_rev,
        "V_conf": float(np.var(confs)) if confs else 0.0,
        "V_risk": float(np.var(risks)) if len(risks) > 1 else 0.0,
        "H_ans": entropy(answers),
        "n_rounds": len(rounds),
    }


def load():
    rows = []
    for path in glob.glob(os.path.join(REC, "*", "*", "*.jsonl")):
        for line in open(path):
            if not line.strip():
                continue
            r = json.loads(line)
            row = {k: r.get(k) for k in ["model", "dataset", "method", "rep", "id", "correct", "total_tokens",
                                         "n_llm_calls", "time_s", "final_status", "draft_correct"]}
            if r.get("method") in MAIN + ABLATIONS:
                row.update(task_features(r))
            rounds = (r.get("rounds") or []) if r.get("method") in MAIN + ABLATIONS else []
            if rounds and "wm" in rounds[0]:
                row["pass_r1"] = int(bool(rounds[0]["wm"].get("accepted")))
                row["meta_right_r1"] = int(str(rounds[0].get("decision", "")).lower() == "right")
            rows.append(row)
    df = pd.DataFrame(rows)
    # reviewer-style configurations have two independent runs (rep 0 and 1); everything else one
    df = df[df.dataset.isin(DATASETS) & df.rep.isin([0, 1])]
    qids = json.load(open(os.path.join(ROOT, "question_ids.json")))
    keep = {(m, ds, i) for m in MODELS for ds in DATASETS for i in qids[ds][: MODEL_N[m]]}
    df = df[[(m, d, i) in keep for m, d, i in zip(df.model, df.dataset, df.id)]]
    return df


# ----------------------------------------------------------------------
# Statistics helpers
# ----------------------------------------------------------------------
def per_question(df, model, dataset, method):
    sub = df[(df.model == model) & (df.dataset == dataset) & (df.method == method)]
    return sub.groupby("id").correct.mean()


def boot_ci(values, n_boot=4000):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return (np.nan, np.nan)
    RNG = np.random.default_rng(0)
    idx = RNG.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]))


def paired_test(a, b, n_perm=20000):
    """Paired sign-flip permutation test on per-question differences (a - b)."""
    ids = a.index.intersection(b.index)
    d = (a.loc[ids] - b.loc[ids]).values
    if len(d) == 0:
        return np.nan, (np.nan, np.nan), np.nan
    obs = d.mean()
    signs = np.random.default_rng(0).choice([-1, 1], size=(n_perm, len(d)))
    perm = (signs * d).mean(axis=1)
    p = (np.sum(np.abs(perm) >= abs(obs) - 1e-12) + 1) / (n_perm + 1)
    return float(obs), boot_ci(d), float(p)


def minmax(x, lo=None, hi=None):
    x = np.asarray(x, dtype=float)
    lo = x.min() if lo is None else lo
    hi = x.max() if hi is None else hi
    return np.clip((x - lo) / (hi - lo + 1e-8), 0, 1)


COMPONENTS = ["H_rev", "V_conf", "V_risk", "H_ans"]


def dataset_cui(df, model, source="peer", ids=None):
    """Dataset-level CUI from the logs of one coordination mode (rep 0)."""
    rows = []
    for ds in DATASETS:
        sub = df[(df.model == model) & (df.dataset == ds) & (df.method == source) & (df.rep == 0)]
        if ids is not None:
            sub = sub[sub.id.isin(ids[ds])]
        if sub.empty:
            continue
        rows.append({"dataset": ds, **{c: sub[c].mean() for c in COMPONENTS}})
    out = pd.DataFrame(rows).set_index("dataset")
    for c in COMPONENTS:
        out[c + "_norm"] = minmax(out[c])
    out["CUI"] = out[[c + "_norm" for c in COMPONENTS]].mean(axis=1)
    return out


# ----------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------
def fmt_acc(x):
    return "--" if pd.isna(x) else f"{100 * x:.1f}"


def write(name, text):
    with open(os.path.join(TAB, name), "w") as f:
        f.write(text)


def master_table(df, name="master_table.csv"):
    rows = []
    for (model, ds, method), sub in df.groupby(["model", "dataset", "method"]):
        pq = sub.groupby("id").correct.mean()
        lo, hi = boot_ci(pq.values)
        rows.append({
            "model": model, "dataset": ds, "method": method, "n_questions": pq.size, "n_runs": len(sub),
            "accuracy": pq.mean(), "ci_lo": lo, "ci_hi": hi,
            "avg_tokens": sub.total_tokens.mean(), "avg_llm_calls": sub.n_llm_calls.mean(),
            "avg_time_s": sub.time_s.mean(), "avg_rounds": sub.n_rounds.mean() if "n_rounds" in sub else np.nan,
        })
    m = pd.DataFrame(rows)
    m.to_csv(os.path.join(OUT, name), index=False)
    return m


def acc(master, model, ds, method, col="accuracy"):
    sel = master[(master.model == model) & (master.dataset == ds) & (master.method == method)]
    return sel[col].iloc[0] if len(sel) else np.nan


def main_results(df, master, summary):
    rows = []
    for model in MODELS:
        for ds in DATASETS:
            p, c, b = (per_question(df, model, ds, m) for m in ["peer", "centralized", "baseline"])
            if p.empty or c.empty:
                continue
            d_pc, ci_pc, p_pc = paired_test(p, c)
            d_pb, ci_pb, p_pb = paired_test(p, b)
            d_cb, ci_cb, p_cb = paired_test(c, b)
            rows.append({"model": model, "dataset": ds, "n": p.size, "baseline": b.mean(), "peer": p.mean(),
                         "centralized": c.mean(), "delta_peer_minus_central": d_pc, "delta_ci_lo": ci_pc[0],
                         "delta_ci_hi": ci_pc[1], "delta_p": p_pc, "peer_vs_base": d_pb, "peer_vs_base_p": p_pb,
                         "central_vs_base": d_cb, "central_vs_base_p": p_cb})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "main_results.csv"), index=False)

    lines = []
    for ds in DATASETS:
        cells = [DS_LABEL[ds]]
        for model in MODELS:
            r = res[(res.model == model) & (res.dataset == ds)]
            if r.empty:
                cells += ["--"] * 3
                continue
            r = r.iloc[0]
            vals = {m: r[m] for m in MAIN}
            best = max(vals.values())
            for m in MAIN:
                s = fmt_acc(vals[m])
                cells.append(f"\\textbf{{{s}}}" if abs(vals[m] - best) < 1e-9 else s)
        lines.append(" & ".join(cells) + " \\\\")
    cells = ["\\midrule\nAverage"]
    for model in MODELS:
        r = res[res.model == model]
        if r.empty:
            cells += ["--"] * 4
            continue
        for m in MAIN:
            cells.append(fmt_acc(r[m].mean()))
        summary[f"avg_acc/{model}"] = {m: float(r[m].mean()) for m in MAIN}
    lines.append(" & ".join(cells) + " \\\\")
    write("main_results.tex", "\n".join(lines) + "\n")

    # pooled paired comparisons (all questions of all datasets), per backbone and overall
    def pooled(models):
        cols = {m: [] for m in MAIN}
        for model in models:
            for ds in DATASETS:
                pq = {m: per_question(df, model, ds, m) for m in MAIN}
                ids = pq["baseline"].index.intersection(pq["peer"].index).intersection(pq["centralized"].index)
                for m in MAIN:
                    cols[m] += list(pq[m].loc[ids].values)
        a = {m: pd.Series(v) for m, v in cols.items()}
        out = {"n": int(len(a["baseline"])), **{m: float(a[m].mean()) for m in MAIN}}
        for x, y in [("peer", "baseline"), ("centralized", "baseline"), ("centralized", "peer")]:
            d, ci, p = paired_test(a[x], a[y])
            out[f"{x}-{y}"] = {"diff": d, "ci": [float(ci[0]), float(ci[1])], "p": p}
        return out
    lines = []
    for label, models in [(MODEL_LABEL[m], [m]) for m in MODELS] + [("All backbones", MODELS)]:
        st = pooled(models)
        summary[f"pooled/{label}"] = st
        cells = [label, str(st["n"])] + [fmt_acc(st[m]) for m in MAIN]
        for key in ["peer-baseline", "centralized-baseline", "centralized-peer"]:
            v = st[key]
            cells.append(f"{100*v['diff']:+.1f} [{100*v['ci'][0]:+.1f}, {100*v['ci'][1]:+.1f}]")
            cells.append(f"{v['p']:.3f}")
        if label == "All backbones":
            lines.append("\\midrule")
        lines.append(" & ".join(cells) + " \\\\")
    write("pooled.tex", "\n".join(lines) + "\n")
    return res


def baselines_table(master, summary):
    for model in MODELS:
        lines = []
        methods = BASELINES + MAIN
        avgs = {}
        for method in methods:
            vals = [acc(master, model, ds, method) for ds in DATASETS]
            if all(pd.isna(v) for v in vals):
                continue
            avgs[method] = (float(np.nanmean(vals)),
                            float(np.nanmean([acc(master, model, ds, method, "avg_tokens") for ds in DATASETS])),
                            float(np.nanmean([acc(master, model, ds, method, "avg_llm_calls") for ds in DATASETS])))
        col_best = {ds: np.nanmax([acc(master, model, ds, m) for m in avgs]) for ds in DATASETS} if avgs else {}
        best_avg = max(v[0] for v in avgs.values()) if avgs else np.nan
        for method in avgs:
            if method == "baseline":
                lines.append("\\midrule")
            cells = [METHOD_LABEL[method]]
            for ds in DATASETS:
                v = acc(master, model, ds, method)
                s = fmt_acc(v)
                cells.append(f"\\textbf{{{s}}}" if not pd.isna(v) and abs(v - col_best[ds]) < 1e-9 else s)
            a, tok, calls = avgs[method]
            s = fmt_acc(a)
            cells.append(f"\\textbf{{{s}}}" if abs(a - best_avg) < 1e-9 else s)
            cells.append(f"{tok / 1000:.1f}k")
            lines.append(" & ".join(cells) + " \\\\")
        write(f"baselines_{model}.tex", "\n".join(lines) + "\n")
        summary[f"baselines_avg/{model}"] = {m: v[0] for m, v in avgs.items()}
        summary[f"baselines_tokens/{model}"] = {m: v[1] for m, v in avgs.items()}


def ablation_table(df, master, summary):
    """Ablations on ABL_MODEL: within-run statistics of each variant plus the paired difference to the full system."""
    model = ABL_MODEL
    # the ablation variants were run once; compare them with the first run of the reference configurations
    df = df[df.rep == 0]
    full = df[(df.model == model) & (df.method == "centralized")].set_index(["dataset", "id"])
    lines, out = [], {}
    for method in ["baseline"] + ABLATIONS:
        sub = df[(df.model == model) & (df.method == method)]
        if sub.empty:
            continue
        ok, bad = sub[sub.draft_correct == 1], sub[sub.draft_correct == 0]
        blocked_bad = bad[bad.pass_r1 == 0]
        d = (sub.correct - sub.draft_correct).values.astype(float)
        signs = np.random.default_rng(0).choice([-1, 1], size=(20000, len(d)))
        net_p = float((np.sum(np.abs((signs * d).mean(axis=1)) >= abs(d.mean()) - 1e-12) + 1) / 20001)
        st = {"n": int(len(sub)), "draft": sub.draft_correct.mean(), "final": sub.correct.mean(),
              "net": sub.correct.mean() - sub.draft_correct.mean(), "net_p": net_p,
              "w2r": ((sub.draft_correct == 0) & (sub.correct == 1)).mean(),
              "r2w": ((sub.draft_correct == 1) & (sub.correct == 0)).mean(),
              "pass_correct": ok.pass_r1.mean(), "pass_wrong": bad.pass_r1.mean(),
              "blocked_wrong_fixed": blocked_bad.correct.mean() if len(blocked_bad) else np.nan,
              "n_blocked_wrong": int(len(blocked_bad)), "rounds": sub.n_rounds.mean(), "tokens": sub.total_tokens.mean()}
        # paired difference of final accuracy against the full system on the same questions
        joined = sub.set_index(["dataset", "id"]).join(full[["correct"]], rsuffix="_full", how="inner")
        diff = (joined.correct - joined.correct_full).values.astype(float)
        if method != "centralized" and len(diff):
            signs = np.random.default_rng(0).choice([-1, 1], size=(20000, len(diff)))
            st["diff_vs_full"] = float(diff.mean())
            st["diff_vs_full_p"] = float((np.sum(np.abs((signs * diff).mean(axis=1)) >= abs(diff.mean()) - 1e-12) + 1) / 20001)
        out[method] = {k: (float(v) if v is not None else v) for k, v in st.items()}
        label = {"baseline": "Reviewer pipeline (no world model)", "centralized": "Full centralized"}.get(method, "\\quad " + METHOD_LABEL[method])
        fixed = "--" if np.isnan(st["blocked_wrong_fixed"]) else f"{100*st['blocked_wrong_fixed']:.0f} ({st['n_blocked_wrong']})"
        dv = "--" if "diff_vs_full" not in st else f"{100*st['diff_vs_full']:+.1f} ({st['diff_vs_full_p']:.2f})"
        lines.append(" & ".join([label, f"{100*st['draft']:.1f}", f"{100*st['final']:.1f}", f"{100*st['net']:+.1f}",
                                 f"{100*st['w2r']:.1f}", f"{100*st['r2w']:.1f}", f"{100*st['pass_correct']:.0f}",
                                 f"{100*st['pass_wrong']:.0f}", fixed, f"{st['rounds']:.2f}", f"{st['tokens']/1000:.1f}k", dv]) + " \\\\")
        if method == "centralized":
            lines.append("\\midrule")
    write("ablation.tex", "\n".join(lines) + "\n")
    summary["ablation"] = out


def efficiency_table(master):
    model = ABL_MODEL
    lines = []
    for ds in DATASETS:
        cells = [DS_LABEL[ds]]
        for col, f in [("avg_tokens", "{:.0f}"), ("avg_llm_calls", "{:.1f}"), ("avg_time_s", "{:.1f}")]:
            for m in MAIN:
                v = acc(master, model, ds, m, col)
                cells.append("--" if pd.isna(v) else f.format(v))
        lines.append(" & ".join(cells) + " \\\\")
    cells = ["\\midrule\nAverage"]
    for col, f in [("avg_tokens", "{:.0f}"), ("avg_llm_calls", "{:.1f}"), ("avg_time_s", "{:.1f}")]:
        for m in MAIN:
            cells.append(f.format(np.nanmean([acc(master, model, ds, m, col) for ds in DATASETS])))
    lines.append(" & ".join(cells) + " \\\\")
    write("efficiency.tex", "\n".join(lines) + "\n")


# ----------------------------------------------------------------------
# CUI analyses
# ----------------------------------------------------------------------
def corr_stats(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    out = {"n": int(len(x))}
    if len(x) < 4 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return out
    r, p = pearsonr(x, y)
    rho, ps = spearmanr(x, y)
    rs = []
    RNG = np.random.default_rng(0)
    for _ in range(10000):
        idx = RNG.integers(0, len(x), len(x))
        if np.std(x[idx]) > 1e-12 and np.std(y[idx]) > 1e-12:
            rs.append(pearsonr(x[idx], y[idx])[0])
    lo, hi = np.quantile(rs, [0.025, 0.975]) if len(rs) > 50 else (np.nan, np.nan)
    out.update(pearson_r=float(r), pearson_p=float(p), spearman_rho=float(rho), spearman_p=float(ps),
               pearson_boot_lo=float(lo), pearson_boot_hi=float(hi))
    return out


def policy_acc(cui, accs, tau):
    """Mean accuracy over datasets of: centralized if CUI < tau else peer."""
    return float(np.mean([accs[ds]["centralized"] if cui[ds] < tau else accs[ds]["peer"] for ds in cui.index]))


def cui_analysis(df, res, summary):
    taus = np.round(np.arange(0.0, 1.0001, 0.02), 2)
    fig_s, axes_s = plt.subplots(1, len(MODELS), figsize=(13, 3.2), sharey=True)
    fig_t, axes_t = plt.subplots(1, len(MODELS), figsize=(13, 3.0), sharey=False)
    cui_rows, sens_rows = [], []
    for k, model in enumerate(MODELS):
        r = res[res.model == model].set_index("dataset")
        if r.empty:
            continue
        table = dataset_cui(df, model, "peer").loc[r.index]
        table["delta"] = r.delta_peer_minus_central
        table["delta_p"] = r.delta_p
        table["peer"], table["centralized"], table["baseline"] = r.peer, r.centralized, r.baseline
        table["model"] = model
        cui_rows.append(table.reset_index())
        summary[f"cui_corr/{model}"] = corr_stats(table.CUI, table.delta)
        # robustness: CUI computed from the logs of other modes
        for src in ["baseline", "centralized"]:
            alt = dataset_cui(df, model, src)
            common = alt.index.intersection(table.index)
            if len(common) >= 4:
                summary[f"cui_corr_src_{src}/{model}"] = corr_stats(alt.loc[common].CUI, table.loc[common].delta)
                summary[f"cui_agreement_peer_vs_{src}/{model}"] = float(
                    spearmanr(alt.loc[common].CUI, table.loc[common].CUI)[0])

        # ---- threshold sensitivity (in-sample) ----
        accs = {ds: {"peer": r.peer[ds], "centralized": r.centralized[ds], "baseline": r.baseline[ds]}
                for ds in r.index}
        curve = [policy_acc(table.CUI, accs, t) for t in taus]
        fixed = {m: float(np.mean([accs[ds][m] for ds in accs])) for m in MAIN}
        oracle = float(np.mean([max(accs[ds]["peer"], accs[ds]["centralized"]) for ds in accs]))
        for t, a in zip(taus, curve):
            sens_rows.append({"model": model, "tau": t, "adaptive_acc": a})
        best = max(curve)
        best_taus = taus[np.isclose(curve, best)]
        summary[f"threshold/{model}"] = {
            "fixed": fixed, "oracle": oracle, "best_adaptive": best,
            "best_tau_range": [float(best_taus.min()), float(best_taus.max())],
            "adaptive_at_0.3": policy_acc(table.CUI, accs, 0.3),
            "tau_range_beating_both_fixed": [float(t) for t in
                                             (lambda g: [g.min(), g.max()] if len(g) else [])(
                                                 taus[np.array(curve) > max(fixed["peer"], fixed["centralized"]) + 1e-9])],
        }

        # ---- held-out evaluation: calibration half -> CUI, tau from other datasets, test half -> accuracy ----
        qids = json.load(open(os.path.join(ROOT, "question_ids.json")))
        used = {ds: qids[ds][: MODEL_N[model]] for ds in DATASETS}
        cal = {ds: set(used[ds][: len(used[ds]) // 2]) for ds in DATASETS}
        test = {ds: set(used[ds][len(used[ds]) // 2:]) for ds in DATASETS}

        def split_acc(ids):
            o = {}
            for ds in r.index:
                o[ds] = {}
                for m in MAIN:
                    pq = per_question(df, model, ds, m)
                    o[ds][m] = float(pq[pq.index.isin(ids[ds])].mean())
            return o
        acc_cal, acc_test = split_acc(cal), split_acc(test)
        raw = dataset_cui(df, model, "peer", ids=cal).loc[r.index]
        chosen = {}
        for held in r.index:
            others = [d for d in r.index if d != held]
            norm = pd.DataFrame(index=r.index)
            for c in COMPONENTS:
                norm[c] = minmax(raw[c], raw.loc[others, c].min(), raw.loc[others, c].max())
            cui = norm.mean(axis=1)
            scores = [np.mean([acc_cal[d]["centralized"] if cui[d] < t else acc_cal[d]["peer"] for d in others])
                      for t in taus]
            good = taus[np.isclose(scores, max(scores))]
            tau = float(good[len(good) // 2])
            chosen[held] = "centralized" if cui[held] < tau else "peer"
        held_out = {
            "adaptive": float(np.mean([acc_test[d][chosen[d]] for d in r.index])),
            "always_peer": float(np.mean([acc_test[d]["peer"] for d in r.index])),
            "always_centralized": float(np.mean([acc_test[d]["centralized"] for d in r.index])),
            "baseline": float(np.mean([acc_test[d]["baseline"] for d in r.index])),
            "oracle": float(np.mean([max(acc_test[d]["peer"], acc_test[d]["centralized"]) for d in r.index])),
            "chosen": chosen,
            "n_correct_choices": int(sum(acc_test[d][chosen[d]] >= max(acc_test[d]["peer"], acc_test[d]["centralized"]) - 1e-9
                                         for d in r.index)),
        }
        summary[f"held_out_policy/{model}"] = held_out

        # ---- figures ----
        ax = axes_s[k]
        colors = ["#c0392b" if d > 0 else "#2c6fbb" if d < 0 else "#7f7f7f" for d in table.delta]
        ax.axhline(0, color="gray", lw=0.8, ls="--")
        ax.scatter(table.CUI, 100 * table.delta, c=colors, s=38, edgecolor="black", linewidth=0.5, zorder=3)
        for ds, row in table.iterrows():
            ax.annotate(DS_LABEL[ds], (row.CUI, 100 * row.delta), xytext=(4, 3), textcoords="offset points", fontsize=6.5)
        st = summary[f"cui_corr/{model}"]
        if "spearman_rho" in st:
            ax.set_title(f"{MODEL_LABEL[model]}  (Spearman $\\rho$={st['spearman_rho']:.2f}, p={st['spearman_p']:.2f})",
                         fontsize=8.5)
        ax.set_xlabel("CUI", fontsize=8.5)
        if k == 0:
            ax.set_ylabel("Acc(Peer) $-$ Acc(Centralized) [pp]", fontsize=8.5)
        ax.tick_params(labelsize=7.5)

        ax = axes_t[k]
        ax.plot(taus, 100 * np.array(curve), color="black", lw=1.6, label="CUI-adaptive")
        ax.axhline(100 * fixed["centralized"], color="#2c6fbb", ls="--", lw=1.1, label="Always centralized")
        ax.axhline(100 * fixed["peer"], color="#c0392b", ls="-.", lw=1.1, label="Always peer")
        ax.axhline(100 * oracle, color="gray", ls=":", lw=1.1, label="Per-dataset oracle")
        ax.set_title(MODEL_LABEL[model], fontsize=8.5)
        ax.set_xlabel("CUI threshold $\\tau$", fontsize=8.5)
        if k == 0:
            ax.set_ylabel("Mean accuracy [%]", fontsize=8.5)
        ax.tick_params(labelsize=7.5)
        if k == 0:
            ax.legend(fontsize=6.5, frameon=False, loc="best")
    fig_s.tight_layout()
    fig_s.savefig(os.path.join(FIG, "cui_scatter.pdf"))
    fig_t.tight_layout()
    fig_t.savefig(os.path.join(FIG, "threshold_sensitivity.pdf"))
    plt.close("all")

    if cui_rows:
        cui_df = pd.concat(cui_rows)
        cui_df.to_csv(os.path.join(OUT, "cui_dataset_level.csv"), index=False)
        pd.DataFrame(sens_rows).to_csv(os.path.join(OUT, "threshold_sensitivity.csv"), index=False)
        lines = []
        for ds in DATASETS:
            cells = [DS_LABEL[ds]]
            for model in MODELS:
                row = cui_df[(cui_df.model == model) & (cui_df.dataset == ds)]
                if row.empty:
                    cells += ["--"] * 6
                    continue
                row = row.iloc[0]
                cells += [f"{row[c + '_norm']:.2f}" for c in COMPONENTS]
                cells += [f"{row.CUI:.3f}", f"{100 * row.delta:+.1f}"]
            lines.append(" & ".join(cells) + " \\\\")
        write("cui.tex", "\n".join(lines) + "\n")
        lines = []
        for ds in DATASETS:
            cells = [DS_LABEL[ds]]
            for model in MODELS:
                row = cui_df[(cui_df.model == model) & (cui_df.dataset == ds)]
                cells += ["--", "--"] if row.empty else [f"{row.iloc[0].CUI:.2f}", f"{100 * row.iloc[0].delta:+.1f}"]
            lines.append(" & ".join(cells) + " \\\\")
        lines.append("\\midrule")
        for name, key in [("Pearson $r$", "pearson_r"), ("\\quad $p$", "pearson_p"),
                          ("Spearman $\\rho$", "spearman_rho"), ("\\quad $p$", "spearman_p")]:
            cells = [name]
            for model in MODELS:
                st = summary.get(f"cui_corr/{model}", {})
                cells.append("\\multicolumn{2}{c}{" + (f"{st[key]:+.2f}" if "_p" not in key else f"{st[key]:.3f}") + "}")
            lines.append(" & ".join(cells) + " \\\\")
        write("cui_compact.tex", "\n".join(lines) + "\n")
        # raw component values for the primary backbone
        lines = []
        for ds in DATASETS:
            row = cui_df[(cui_df.model == ABL_MODEL) & (cui_df.dataset == ds)]
            if row.empty:
                continue
            row = row.iloc[0]
            lines.append(" & ".join([DS_LABEL[ds]] + [f"{row[c]:.4f}" for c in COMPONENTS]) + " \\\\")
        write("cui_raw.tex", "\n".join(lines) + "\n")


def fine_grained(df, summary):
    """Task-level and batch-level predictiveness of CUI (pooled min-max normalised task features)."""
    lines = []
    for model in MODELS:
        feats = df[(df.model == model) & (df.method == "peer") & (df.rep == 0)][["dataset", "id"] + COMPONENTS]
        if feats.empty:
            continue
        feats = feats.copy()
        for c in COMPONENTS:
            feats[c] = minmax(feats[c])
        feats["CUI"] = feats[COMPONENTS].mean(axis=1)
        parts = []
        for ds in DATASETS:
            p, c = per_question(df, model, ds, "peer"), per_question(df, model, ds, "centralized")
            if p.empty or c.empty:
                continue
            t = pd.DataFrame({"peer": p, "central": c}).dropna().reset_index()
            t["dataset"] = ds
            parts.append(t)
        tasks = pd.concat(parts).merge(feats, on=["dataset", "id"])
        tasks["label"] = (tasks.peer > tasks.central).astype(int)

        def fit_eval(frame, x, y):
            out = {}
            X, Y, G = frame[[x]].values, frame[y].values, frame["dataset"].values
            if len(np.unique(Y)) < 2:
                return out
            clf = LogisticRegression(max_iter=5000).fit(X, Y)
            out["coef"] = float(clf.coef_[0][0])
            out["acc_in"] = float(accuracy_score(Y, clf.predict(X)))
            out["auc_in"] = float(roc_auc_score(Y, clf.predict_proba(X)[:, 1]))
            out["majority_rate"] = float(max(Y.mean(), 1 - Y.mean()))
            accs, aucs = [], []
            for tr, te in GroupKFold(n_splits=min(5, len(np.unique(G)))).split(X, Y, G):
                if len(np.unique(Y[tr])) < 2:
                    continue
                c2 = LogisticRegression(max_iter=5000).fit(X[tr], Y[tr])
                accs.append(accuracy_score(Y[te], c2.predict(X[te])))
                if len(np.unique(Y[te])) > 1:
                    aucs.append(roc_auc_score(Y[te], c2.predict_proba(X[te])[:, 1]))
            out["cv_acc"] = float(np.mean(accs)) if accs else np.nan
            out["cv_auc"] = float(np.mean(aucs)) if aucs else np.nan
            return out

        task_stats = fit_eval(tasks, "CUI", "label")
        task_stats["n"] = int(len(tasks))
        summary[f"task_level/{model}"] = task_stats
        if task_stats.get("auc_in") is not None:
            lines.append(" & ".join([MODEL_LABEL[model], "1 (task)", str(len(tasks)), "--",
                                     f"{task_stats.get('auc_in', np.nan):.3f}", f"{task_stats.get('cv_acc', np.nan):.3f}",
                                     f"{task_stats.get('cv_auc', np.nan):.3f}"]) + " \\\\")
        for bs in (5, 10, 20):
            rows = []
            for ds, sub in tasks.groupby("dataset"):
                sub = sub.sort_values("id").reset_index(drop=True)
                for b in range(int(np.ceil(len(sub) / bs))):
                    ch = sub.iloc[b * bs:(b + 1) * bs]
                    if len(ch) < max(2, bs // 2):
                        continue
                    rows.append({"dataset": ds, "CUI": ch.CUI.mean(), "delta": ch.peer.mean() - ch.central.mean()})
            bdf = pd.DataFrame(rows)
            if bdf.empty:
                continue
            bdf["label"] = (bdf.delta > 0).astype(int)
            st = fit_eval(bdf, "CUI", "label")
            lin = LinearRegression().fit(bdf[["CUI"]].values, bdf.delta.values)
            st.update(n=int(len(bdf)), lin_coef=float(lin.coef_[0]), r2=float(lin.score(bdf[["CUI"]].values, bdf.delta.values)))
            summary[f"batch_level/{model}/bs{bs}"] = st
            lines.append(" & ".join([MODEL_LABEL[model] if False else "", str(bs), str(len(bdf)), f"{st['r2']:.3f}",
                                     f"{st.get('auc_in', np.nan):.3f}", f"{st.get('cv_acc', np.nan):.3f}",
                                     f"{st.get('cv_auc', np.nan):.3f}"]) + " \\\\")
        lines.append("\\midrule")
    if lines and lines[-1] == "\\midrule":
        lines.pop()
    write("fine_grained.tex", "\n".join(lines) + "\n")


def gate_table(df, summary):
    """Within-run effect of each mode: draft vs. final accuracy and gate discrimination."""
    lines, out = [], {}
    for model in MODELS:
        for m in MAIN:
            sub = df[(df.model == model) & (df.method == m)]
            if sub.empty:
                continue
            ok, bad = sub[sub.draft_correct == 1], sub[sub.draft_correct == 0]
            blocked_bad = bad[bad.pass_r1 == 0]
            st = {
                "draft": sub.draft_correct.mean(), "final": sub.correct.mean(),
                "w2r": ((sub.draft_correct == 0) & (sub.correct == 1)).mean(),
                "r2w": ((sub.draft_correct == 1) & (sub.correct == 0)).mean(),
                "pass_correct": ok.pass_r1.mean(), "pass_wrong": bad.pass_r1.mean(),
                "meta_right_on_wrong": bad.meta_right_r1.mean(),
                "blocked_wrong_fixed": blocked_bad.correct.mean() if len(blocked_bad) else np.nan,
                "n_blocked_wrong": int(len(blocked_bad)), "rounds": sub.n_rounds.mean(),
                "tokens": sub.total_tokens.mean(), "calls": sub.n_llm_calls.mean(), "n": int(len(sub)),
            }
            # paired within-run test: final vs draft
            d = (sub.correct - sub.draft_correct).values.astype(float)
            signs = np.random.default_rng(0).choice([-1, 1], size=(20000, len(d)))
            st["net_p"] = float((np.sum(np.abs((signs * d).mean(axis=1)) >= abs(d.mean()) - 1e-12) + 1) / 20001)
            out[f"{model}/{m}"] = {k: float(v) for k, v in st.items()}
            label = {"baseline": "Reviewer pipeline", "peer": "Peer", "centralized": "Centralized"}[m]
            fixed = "--" if np.isnan(st["blocked_wrong_fixed"]) else f"{100*st['blocked_wrong_fixed']:.0f} ({st['n_blocked_wrong']})"
            lines.append(" & ".join([
                MODEL_LABEL[model] if m == "baseline" else "", label,
                f"{100*st['draft']:.1f}", f"{100*st['final']:.1f}", f"{100*(st['final']-st['draft']):+.1f}",
                f"{100*st['w2r']:.1f}", f"{100*st['r2w']:.1f}",
                f"{100*st['pass_correct']:.0f}", f"{100*st['pass_wrong']:.0f}", fixed,
                f"{st['rounds']:.2f}", f"{st['tokens']/1000:.1f}k"]) + " \\\\")
        lines.append("\\midrule")
    if lines:
        lines.pop()
    write("gate.tex", "\n".join(lines) + "\n")
    summary["gate"] = out


def rep_consistency(df, summary):
    """Per-run accuracies and pooled differences, to show how stable the main comparison is across the two runs."""
    out, lines = {}, []
    for model in MODELS:
        row = {}
        for r in (0, 1):
            sub = df[(df.model == model) & (df.rep == r)]
            if sub.empty:
                continue
            accs = {m: sub[sub.method == m].correct.mean() for m in MAIN}
            ids = None
            for m in MAIN:
                s_ids = set(zip(sub[sub.method == m].dataset, sub[sub.method == m].id))
                ids = s_ids if ids is None else ids & s_ids
            row[f"rep{r}"] = {**{m: float(accs[m]) for m in MAIN}, "n": len(ids)}
        if "rep0" in row and "rep1" in row:
            out[model] = row
            cells = [MODEL_LABEL[model]]
            for r in ("rep0", "rep1"):
                cells += [fmt_acc(row[r][m]) for m in MAIN]
            cells += [f"{100*(row['rep0']['centralized']-row['rep0']['baseline']):+.1f}",
                      f"{100*(row['rep1']['centralized']-row['rep1']['baseline']):+.1f}"]
            lines.append(" & ".join(cells) + " \\\\")
    write("rep_consistency.tex", "\n".join(lines) + "\n")
    summary["rep_consistency"] = out


def correction_flow_figure(df, summary):
    """Where collective correction succeeds and fails: fate of correct and incorrect drafts, ungated vs gated."""
    rows = []
    for model in MODELS:
        for m in ["baseline", "centralized"]:
            sub = df[(df.model == model) & (df.method == m)]
            if sub.empty or "pass_r1" not in sub:
                continue
            bad, ok = sub[sub.draft_correct == 0], sub[sub.draft_correct == 1]
            rows.append({
                "model": model, "method": m, "n_bad": len(bad), "n_ok": len(ok),
                "bad_false_consensus": (bad.pass_r1 == 1).mean(),
                "bad_intercepted_corrected": ((bad.pass_r1 == 0) & (bad.correct == 1)).mean(),
                "bad_intercepted_uncorrected": ((bad.pass_r1 == 0) & (bad.correct == 0)).mean(),
                "bad_passed_then_corrected": ((bad.pass_r1 == 1) & (bad.correct == 1)).mean(),
                "ok_accepted": (ok.pass_r1 == 1).mean(),
                "ok_challenged_kept": ((ok.pass_r1 == 0) & (ok.correct == 1)).mean(),
                "ok_corrupted": (ok.correct == 0).mean(),
            })
    flow = pd.DataFrame(rows)
    flow.to_csv(os.path.join(OUT, "correction_flow.csv"), index=False)
    summary["correction_flow"] = {f"{r['model']}/{r['method']}": {k: float(v) for k, v in r.items() if k not in ("model", "method")}
                                  for r in rows}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), sharey=True)
    short = {"gpt-3.5-turbo": "3.5-Turbo", "gpt-4o-mini": "4o-mini", "gpt-4.1-nano": "4.1-nano", "gpt-4.1-mini": "4.1-mini"}
    labels = [f"{short[m]}\n{'ungated' if meth == 'baseline' else 'gated'}" for m in MODELS for meth in ["baseline", "centralized"]]
    order = [(m, meth) for m in MODELS for meth in ["baseline", "centralized"]]
    x = np.arange(len(order))
    # incorrect drafts
    ax = axes[0]
    keys = [("bad_false_consensus", "accepted at once (false consensus)", "#c0392b"),
            ("bad_intercepted_uncorrected", "challenged, not repaired", "#e59866"),
            ("bad_intercepted_corrected", "challenged and repaired", "#2e8b57")]
    bottom = np.zeros(len(order))
    for key, lab, col in keys:
        vals = np.array([100 * flow[(flow.model == m) & (flow.method == meth)][key].iloc[0] for m, meth in order])
        ax.bar(x, vals, bottom=bottom, color=col, label=lab, width=0.7, edgecolor="white", linewidth=0.5)
        bottom += vals
    ax.set_title("Incorrect initial answers", fontsize=9)
    ax.set_ylabel("Share of drafts [%]", fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=6.5)
    ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    ax.tick_params(labelsize=7.5)
    # correct drafts
    ax = axes[1]
    keys = [("ok_accepted", "accepted at once", "#2e8b57"),
            ("ok_challenged_kept", "challenged, kept", "#a9dfbf"),
            ("ok_corrupted", "lost (changed to wrong)", "#c0392b")]
    bottom = np.zeros(len(order))
    for key, lab, col in keys:
        vals = np.array([100 * flow[(flow.model == m) & (flow.method == meth)][key].iloc[0] for m, meth in order])
        ax.bar(x, vals, bottom=bottom, color=col, label=lab, width=0.7, edgecolor="white", linewidth=0.5)
        bottom += vals
    ax.set_title("Correct initial answers", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=6.5)
    ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    ax.tick_params(labelsize=7.5)
    for ax in axes:
        for i in range(1, len(order), 2):
            ax.axvline(i + 0.5, color="gray", lw=0.4, ls=":")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "correction_flow.pdf"))
    plt.close(fig)


def behaviour_stats(df, summary):
    """Mechanism statistics for the main modes (primary backbone)."""
    out = {}
    for model in MODELS:
        for m in MAIN + ABLATIONS[1:]:
            sub = df[(df.model == model) & (df.method == m)]
            if sub.empty:
                continue
            fixed = ((sub.draft_correct == 0) & (sub.correct == 1)).mean()
            broken = ((sub.draft_correct == 1) & (sub.correct == 0)).mean()
            out[f"{model}/{m}"] = {"avg_rounds": float(sub.n_rounds.mean()), "draft_acc": float(sub.draft_correct.mean()),
                                   "final_acc": float(sub.correct.mean()), "wrong_to_right": float(fixed),
                                   "right_to_wrong": float(broken),
                                   "multi_round_rate": float((sub.n_rounds > 1).mean())}
    summary["behaviour"] = out


def main():
    df = load()
    print("records:", len(df))
    print(df.groupby(["model", "method"]).size().unstack(0))
    summary = {}
    master = master_table(df)
    res = main_results(df, master, summary)
    # baselines and ablations are run on the first 50 questions of each dataset; compare on that subset
    qids = json.load(open(os.path.join(ROOT, "question_ids.json")))
    first50 = {(ds, i) for ds in DATASETS for i in qids[ds][:50]}
    df50 = df[[(d, i) in first50 for d, i in zip(df.dataset, df.id)]]
    master50 = master_table(df50, "master_table_first50.csv")
    baselines_table(master50, summary)
    ablation_table(df50, master50, summary)
    efficiency_table(master)
    cui_analysis(df, res, summary)
    fine_grained(df, summary)
    behaviour_stats(df, summary)
    gate_table(df, summary)
    rep_consistency(df, summary)
    correction_flow_figure(df, summary)
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1, default=float)
    pd.set_option("display.width", 250)
    print(res.round(3).to_string())
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
