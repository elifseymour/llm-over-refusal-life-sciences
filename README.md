# Over-Refusal in the Life Sciences

An evaluation framework and expert-designed dataset for measuring false refusals of legitimate research queries in frontier LLMs.

## What this is

This project ships two durable artifacts: an expert-designed 93-query dataset of unambiguously safe life-science queries, and a reproducible evaluation framework (model runner, validated LLM-as-judge scorer, metrics, figures) for measuring how often a model wrongly refuses or returns a not-useful answer on that dataset. It includes a small demonstration run on two Anthropic models to show the pipeline end to end. The dataset is fully releasable, so others can extend the evaluation to any model.

## Research question

How often do frontier language models refuse or return not-useful help on legitimate, safe life-science queries, and what query features cause these false refusals?

## Why this matters

Frontier models are increasingly tuned to refuse dangerous requests. In the life sciences, that caution spills over onto legitimate work: scientists report that current models block or water down routine questions about inhibitors, pathogens, assays, and lab procedure. Each false refusal is a safety system failing in the opposite direction from the one usually measured. This project quantifies that helpfulness cost and diagnoses what triggers it. Because the over-refusals studied here come from safety safeguards misfiring on benign content, measuring and reducing them matters for keeping AI genuinely useful to the researchers, clinicians, and public-health workers it is meant to help, without weakening protections against genuine misuse.

## What this measures

Each model response to a safe query is scored into one of three primary categories, judged on the response as delivered:

- Refusal: declines or blocks the request.
- Useful response: answers in a way that actually helps the legitimate task.
- Not-useful response: responds but the answer does not help (heavy hedging, over-disclaiming, watered-down content, or deflection).

Each not-useful response also carries a secondary cause tag, caution-driven versus other, so the not-useful rate can be separated from general capability. See `rubrics/scoring_rubric.md`.

## Scope: framework plus demonstration

The deliverables are the dataset and the framework. The demonstration run deliberately stays small: a pilot sample of queries on two Anthropic models (Sonnet 5 vs Opus 4.8), scored by a non-Claude judge. This shows the pipeline works without bulk-running biosecurity-adjacent content across many providers. The pilot rates are illustrative, not findings; the number reported with confidence is judge-versus-expert agreement (Cohen's kappa). GPT, Gemini, and open-weight models are defined in `src/run_models.py` as framework-supported targets for future runs.

## Outputs

- A public, expert-designed 93-query dataset of safe life-science queries (fully releasable; no infohazard).
- A reproducible evaluation framework and a validated LLM-as-judge scorer for refusal / useful / not-useful plus the cause tag.
- Judge-versus-expert agreement from the pilot, and illustrative false-refusal and not-useful rates for the two Claude models.
- A taxonomy of the query features that trigger unnecessary refusals.
- A short paper-style report and a one-page summary.

## Repository structure

- `data/` : the query set and its metadata schema.
- `rubrics/` : query design guidelines and the scoring rubric.
- `src/` : Python code for running models and scoring responses.
- `notebooks/` : analysis notebooks.
- `results/` : model responses, scored data, metrics, and figures.
- `docs/` : the paper-style report and one-page summary.

## How to reproduce

```bash
pip install -r requirements.txt
cp config.example.env .env   # then fill in ANTHROPIC_API_KEY and OPENAI_API_KEY

# 1. Collect responses for a topic-balanced pilot (15 infectious-disease + 15
#    chemistry-control queries) on the two active Claude models
python src/run_models.py --stratify --per-life-topic 15 \
  --life-topics infectious_disease_diagnostics --n-control 15

# 2. Score them with the non-Claude judge
python src/judge.py

# 3. Compute metrics and figures in the analysis notebook
jupyter notebook notebooks/analysis.ipynb
```

To run the full 93-query set, pass a large `--pilot-n` (e.g. `--pilot-n 999`). To add other providers, flip their `active` flag in `src/run_models.py` and set the matching key in `.env`.

## Safety note

Every query in this benchmark is safe by construction, and the failure mode under study is refusal of safe content, not disclosure of unsafe content. The full dataset, code, and results are safe to release.

## Status

In development. See the project plan for the detailed schedule.
