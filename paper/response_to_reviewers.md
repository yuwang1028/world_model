# Response to Reviewers

**Manuscript (previous title):** How Can World Models Fit in Cyber-Physical AI Multi-Agent Reasoning Systems?
**Revised title:** Do World Models Help Multi-Agent LLM Reasoning? A Controlled Study of State-Centric Coordination

We thank both reviewers for their careful reading. Their comments led us to re-run every experiment under stricter conditions, and the outcome changed the conclusions of the paper. We therefore begin with a summary of what changed and why, before answering each comment.

## Summary of major changes

**1. The evaluation protocol was rebuilt, and the main empirical claim of the previous version did not survive it.**
While reconciling the tables (Reviewer 1, D9) we audited the run logs of the previous version and found that the inconsistencies were symptoms of a deeper problem: on several benchmarks the three coordination modes had been evaluated on *different* randomly sampled questions, some reported numbers came from different runs than others, and one component of the CUI (risk instability) had never been logged and was identically zero. The accuracy differences reported in the previous version were therefore not paired comparisons.

In the revision every method is run on the same fixed questions, with one runner script, and every table and figure is generated from the per-question logs by one analysis script. Under this protocol:

- the previously reported regime ("centralized coordination is better for low-uncertainty tasks, peer coordination for high-uncertainty tasks") is **not reproduced**. None of the 32 backbone–benchmark combinations shows a significant difference between the two modes, and the sign of the difference changes across backbones;
- peer-to-peer coordination is statistically indistinguishable from the message-based baseline;
- centralized coordination changes the behaviour of the collective (it forces reconsideration of most drafts) but not its accuracy: over two independent runs its net gain over the author's draft is 0.0–1.8 points on the four backbones, no larger than that of the ungated pipeline, at 1.7–4.3 times the token cost. A gain of 4.3 points observed on one backbone in the first run did not reproduce in the second;
- the Coordination Uncertainty Index does not predict the preferable mode.

We have rewritten the title, abstract, introduction, results, and conclusion accordingly. The paper is now a controlled study that reports these findings, explains them (reviewer agents accept 81–99% of incorrect drafts; the state-level gate separates correct from incorrect drafts poorly; blocked errors are rarely repaired), and positions the framework honestly against external baselines. We withdraw the claims of the previous version that the framework "consistently enhances both solution quality and coordination efficiency" and that the CUI "predicts coordination regimes."

**2. The description of the peer-to-peer mode was corrected.** The previous text described peer coordination as exploring multiple reasoning trajectories in parallel. That description did not match the implementation, in which the peer mode keeps local state replicas and leaves acceptance to the agents. The method section now describes what the system actually does, and the "error locking vs. parallel exploration" discussion has been removed.

**3. One implementation defect was fixed.** In the previous code, a candidate that failed transition validation was replaced by the validator's hint text instead of being returned to the author. The revised implementation asks the author to re-submit, as the algorithm in the paper specifies. All results in the revision use the corrected code.

**4. The scope was narrowed.** All experiments concern abstract reasoning benchmarks. The term "cyber-physical" has been removed from the title and abstract; the paper no longer makes claims about cyber-physical systems and lists this as a limitation.

---

## Reviewer 1

### D1 — Memory-based approaches

> The related work section lacks discussion on memory-based approaches, which structurally store dialogue context as memory. These works are highly similar to the present study.

We agree that this relationship should have been discussed. The Related Work section now contains a paragraph on memory-based agents (Generative Agents, MemGPT, MemoryBank, Reflexion, Voyager, and the CoALA taxonomy). We state explicitly that, in CoALA's terms, our state store is a structured working memory for a single task and that we do not claim a new memory architecture. The distinction we draw is functional: memory systems decide which past information to place in the context (write, retrieve, summarize), whereas the world model in our framework decides whether a proposed step may be committed and whether the current candidate may be accepted (typed transition, validation, trajectory risk, acceptance gate). We also note that our state is not persistent across tasks, so none of the reported effects can come from cross-task memory. The positioning table (Table I) includes memory agents as a column and marks persistent memory as a capability our framework does not have.

### D2 — Recent multi-agent literature

> The authors are encouraged to expand the related-work discussion to include recent studies on role-specialized multi-agent collaboration, workflow-level communication optimization, and graph-structured agent orchestration (e.g., Zero-shot Knowledge Graph Question Generation via Multi-agent LLMs and Small Models Synthesis, AgentVerse, GPTSwarm).

Done. A new paragraph covers role specialization and dynamic team formation (AgentVerse; DyLAN; Zhao et al., CIKM 2024), and communication-structure optimization (MacNet, GPTSwarm, AgentPrune). We explain that these methods decide who communicates with whom, whereas our study keeps the topology fixed and asks what the agents coordinate through; the two are complementary. We also now cite MARS, the author–reviewer–meta-reviewer system on which our baseline is built, which the previous version did not credit clearly.

### D3 — World-model construction

> The core of the paper is the world model, yet Section IV does not elaborate on its specific construction and internal state composition.

Section IV has been rewritten to be a complete specification:

- **State composition**: each state is a node ⟨id, parent, action, observation⟩ in a rooted tree; an action is ⟨type, content, meta⟩ with types INIT, DRAFT, REVISION, and REVALIDATED.
- **Transition T**, **validation V** (the two rule-based checks and what happens on failure), and **state summary σ** (what reviewers receive).
- **Rollout risk R**: the full formula (Eq. 6) with all constants, including how reviewer polarity and reviewer reliability (exponential of the mean token log-probability) enter.
- **Acceptance gate**: the effective-risk formula, the two acceptance conditions (Eqs. 8–9), and a table of all hyperparameters.
- **Peer vs. centralized**: precisely which of these operations each architecture uses.
- **Algorithm 1** has been corrected (the previous version assigned the risk twice and was inconsistent in its indexing).

We state plainly that the world model is a symbolic state store with rule-based evaluation functions: it involves no additional LLM calls and no learned parameters, and its constants were set by hand. Figures 1–3 were redrawn to show the state-update and validation flow.

### D4 — Underlying LLM and generalizability

> The paper does not specify the underlying LLM used by the agents, nor does it conduct a generalizability analysis with multiple LLMs.

Section V-A now reports the exact model snapshots, the API, and the decoding configuration (temperature 1.0, default nucleus sampling and output length, fresh random seed per call, log-probabilities enabled, no system prompt or tools).

We ran the main comparison with four backbones: gpt-3.5-turbo-0125 (the model of the previous version), gpt-4o-mini-2024-07-18, gpt-4.1-nano-2025-04-14, and gpt-4.1-mini-2025-04-14, on identical questions (398 per backbone). The answer to the generalizability question is that there is **no accuracy effect to generalize**: over two independent runs the within-run gain of the centralized gate is 0.0, +1.8, +1.4, and +0.4 points on the four backbones (none significant), no larger than that of the ungated pipeline; what does generalize is the behavioural pattern (ungated reviewers accept 80–97% of incorrect drafts; the gate forces reconsideration but the author rarely repairs). We also added a table of run-to-run variation showing that a fixed configuration on fixed questions varies by up to 4 points between runs, which is why the single-run result of the first run (+4.3 points on GPT-4o-mini, p = 0.01) did not reproduce.

### D5 — Stronger baselines

> The experiments only use a single simple strategy as the baseline. It is recommended to include recent work on multi-agent collaborative reasoning from the past two years.

We re-implemented six baselines in the same pipeline, with the same backbone, decoding configuration, answer format, questions, and scoring: chain-of-thought, self-consistency (5 samples), self-reflection, Multi-Agent Debate (Du et al., ICML 2024), MAD (Liang et al., EMNLP 2024), and ReConcile (Chen et al., ACL 2024; single-backbone variant). They were run on two backbones (Table XI). No numbers are copied from other papers.

The result does not favor our framework: apart from MAD, all methods lie within 3.5 points of each other; a single chain-of-thought call is within 1.2–2.0 points of the best multi-agent method; and the centralized configuration ranks first on one backbone and second-to-last on the other while being the most expensive method on both. We state in the paper that we do not claim to outperform existing methods. We explain in Section V-C why AgentVerse and GPTSwarm were not included as baselines (they require task-specific role recruitment and a training split, respectively).

### D6 — CUI threshold sensitivity

> No hyperparameter analysis is conducted on the choice of the threshold.

We added the requested analysis (Section VI-D, Fig. 5): the accuracy of the CUI-thresholded policy is plotted for every threshold in [0, 1], for each backbone, against the two fixed policies and a per-benchmark oracle. We also added an out-of-sample evaluation in which the CUI is computed on half of the questions of a benchmark, the threshold is chosen on the other seven benchmarks, and the selected mode is scored on the held-out half.

The analysis shows that the threshold rule of the previous version is not supported. No single threshold is good for all backbones (the best threshold is "always centralized" for one backbone and "always peer" for another); the previously proposed value τ ≈ 0.3 is best for GPT-3.5-Turbo but costs 2.2 points for GPT-4o-mini; and out of sample the adaptive policy is one point better than the better fixed mode on one backbone, half a point worse on another, and identical on two. The correlation between the CUI and the accuracy gap is positive on two backbones, negative on one, and absent on one. We now present the CUI as a descriptive statistic of reviewer disagreement and report its failure as a selector as a negative result. The "theoretical formulation" and Proposition 1 of the previous version, which had no formal statement, have been removed.

We also corrected the definition of the CUI: the reviewer-entropy component now matches its stated definition, the answer-entropy component (previously undefined) is defined and computed over normalized answers, the risk-instability component is now actually logged, and the normalization and weights are stated.

### D7 — Ablation

> The experiments lack ablation studies to analyze the contribution of each component.

We added an ablation (Section VI-C, Table VIII) with GPT-4o-mini, the backbone on which the centralized configuration has its largest effect. Starting from the full configuration we remove, one at a time, (i) the shared state summary shown to reviewers, (ii) transition validation, (iii) the trajectory-risk estimate, and (iv) the world-model-guided acceptance gate. Each variant is run on the same 398 questions, and we report within-run draft-to-final gains, first-round acceptance rates for correct and incorrect drafts, repair rates, rounds, tokens, and paired tests against the full configuration.

Because the ablation variants were run once and the reference run of the full system turned out to be a high draw (its second run gained −0.8 points), we do not interpret the accuracy differences between variants as component contributions; they are within run-to-run variation. The behavioural columns are stable and informative: removing the state summary, validation, or the risk estimate leaves the gate's acceptance behaviour, rounds, and cost unchanged (transition validation in particular does nothing), while the variant that merely reports the risk to the meta-reviewer without gating lowers the first-round acceptance of incorrect drafts from 85% to 78% at a third of the cost. We discuss the implication (make the risk signal visible to the deciding agent; forced rounds cost tokens without buying accuracy) in the ablation section and the discussion.

### D8 — AI-generated figures

> The motivation diagram, framework diagram, and other figures appear to be AI-generated, with visible watermarks, insufficient resolution, and unclear logic.

The reviewer is right, and we apologize; these figures should not have been submitted. Figures 1–3 have been replaced by new vector diagrams typeset from source code (TikZ) within the manuscript. They contain no raster content, and their arrows correspond one-to-one to the operations defined in Section IV. The data figures are vector graphics generated by the analysis script. An explicit statement on the use of generative AI tools has been added to the Acknowledgment.

### D9 — Table inconsistencies

> Tables II and III list 7 datasets, but Table VI lists 8, adding MMLU-Elementary, and Table V omits CIAR.

Thank you for catching this. As described in the summary above, investigating these inconsistencies revealed that the underlying comparisons were not paired, which is why we re-ran all experiments rather than only correcting the tables. In the revision the same eight benchmarks appear in every table, each benchmark is introduced with its citation in Section V-B, and all tables are produced from a single set of logs by one script. A master spreadsheet (dataset × method × backbone × accuracy × tokens × calls × time) is provided with the code.

---

## Reviewer 2

### Comment 1 — Generative-AI watermark and disclosure

We acknowledge that the previous submission contained figures with a visible generative-AI watermark and did not disclose the use of generative AI tools. This should have been addressed before submission, and we apologize. Figures 1–3 have been redrawn as vector diagrams from source code and contain no generated imagery. The Acknowledgment now discloses that generative AI tools were used for language editing, for assistance with experiment and analysis code, and for producing the source code of the diagrams, and that the authors reviewed all content and take responsibility for it.

### Comment 2 — LLM specification and fairness of comparison

> The manuscript does not specify which large language model is used (model name, version, deployment setting, etc.). It is difficult to assess whether the comparison is fair.

Section V-A now lists the four model snapshots, the API through which they are accessed, and all decoding parameters. Regarding fairness: within every comparison, all methods use the same backbone, the same decoding configuration, the same answer format, the same scoring script, and the same questions; the reviewer-style configurations additionally share all prompts. As explained in the summary above, this was not the case for part of the previous version, where different modes had been evaluated on different question samples; the earlier conclusions changed once it was enforced. Comparisons are accompanied by paired permutation tests and bootstrap confidence intervals; the reviewer-style configurations were run twice, and the run-to-run variation is reported so that readers can judge the single-run comparisons. The remaining limitations (at most 50 questions per benchmark; single runs for baselines and ablations) are discussed in the limitations section.
