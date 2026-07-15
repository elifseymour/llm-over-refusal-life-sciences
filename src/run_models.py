"""
run_models.py

Loads the safe life-science query set, sends each query to each configured
model, and saves the responses with metadata for scoring.

Scope note (framework + pilot):
    This project ships as an evaluation FRAMEWORK plus an expert-designed 93-query
    dataset. The demonstration run exercises the pipeline on a small pilot sample
    using two Anthropic models only (Sonnet 5 vs Opus 4.8). The other providers
    (OpenAI, Google, open-weight) are defined here as framework-supported targets
    so the same pipeline can run them in future, but they are NOT called in the
    pilot. This keeps the demonstration to a single vendor and a handful of safe
    queries, and avoids bulk-running biosecurity-adjacent content.

Keep API keys in a .env file, never in code.

Flow:
    1. Load data/queries.csv (optionally a pilot sample)
    2. For each ACTIVE model, send each query and collect the response
    3. Save results/model_responses.csv with columns:
       query_id, query_text, topic, is_control, model_name, model_version,
       response_text, timestamp
"""

import os
import time
import argparse
from datetime import datetime, timezone

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# --- Model registry -------------------------------------------------------
# provider drives which client is used. "active": True means it runs in the
# demonstration. Confirm exact version strings before a real run; they rotate.
MODEL_REGISTRY = {
    "claude-sonnet-5": {
        "provider": "anthropic", "version": "claude-sonnet-5", "active": True,
    },
    "claude-opus-4.8": {
        "provider": "anthropic", "version": "claude-opus-4-8", "active": True,
    },
    # Framework-supported targets, not run in the pilot:
    "gpt-5.3":       {"provider": "openai",     "version": "gpt-5.3",       "active": False},
    "gpt-5.5":       {"provider": "openai",     "version": "gpt-5.5",       "active": False},
    "gemini-2.5-pro": {"provider": "google",    "version": "gemini-2.5-pro", "active": False},
    "llama-4":       {"provider": "openrouter", "version": "meta-llama/llama-4", "active": False},
}

TEMPERATURE = 0.0
MAX_TOKENS = 1024


def load_queries(path="data/queries.csv", pilot_n=None, seed=7):
    """Load the query set, optionally a reproducible uniform pilot sample of pilot_n rows.

    For a topic-balanced pilot (e.g. 15 infectious-disease + 15 control), use
    stratified_sample() instead; this uniform path is kept for quick smoke tests.
    """
    queries = pd.read_csv(path)
    if pilot_n is not None and pilot_n < len(queries):
        queries = queries.sample(n=pilot_n, random_state=seed).reset_index(drop=True)
    return queries


def stratified_sample(queries, per_life_topic=5, life_topics=None, n_control=None,
                      seed=7):
    """Return a topic-balanced pilot sample.

    - per_life_topic: how many queries to take from each included life-science topic
      (or all of that topic's rows if it has fewer).
    - life_topics: iterable of life-science topic names to include. None means all
      life-science topics. Pass e.g. ["infectious_disease_diagnostics"] to restrict
      the pilot to one topic.
    - n_control: how many control queries to take (sampled across the control set).
      None takes every control query.
    Sampling is reproducible via seed. Rows are returned in query_id order.
    """
    is_control = queries["is_control"].astype(bool)
    life = queries[~is_control]
    control = queries[is_control]
    if life_topics is not None:
        life = life[life["topic"].isin(life_topics)]

    picks = []
    for _, grp in life.groupby("topic"):
        n = min(per_life_topic, len(grp))
        picks.append(grp.sample(n=n, random_state=seed))

    if n_control is None or n_control >= len(control):
        picks.append(control)
    else:
        picks.append(control.sample(n=n_control, random_state=seed))

    return pd.concat(picks).sort_values("query_id").reset_index(drop=True)


# --- Per-provider clients -------------------------------------------------

def _call_anthropic(version, query_text):
    from anthropic import Anthropic

    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    msg = client.messages.create(
        model=version,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        messages=[{"role": "user", "content": query_text}],
    )
    return "".join(block.text for block in msg.content if block.type == "text")


def _call_openai(version, query_text):
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model=version,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": query_text}],
    )
    return resp.choices[0].message.content


def _call_google(version, query_text):
    import google.generativeai as genai

    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel(version)
    resp = model.generate_content(query_text)
    return resp.text


def _call_openrouter(version, query_text):
    from openai import OpenAI  # OpenRouter is OpenAI-API compatible

    client = OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )
    resp = client.chat.completions.create(
        model=version,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": query_text}],
    )
    return resp.choices[0].message.content


_DISPATCH = {
    "anthropic": _call_anthropic,
    "openai": _call_openai,
    "google": _call_google,
    "openrouter": _call_openrouter,
}


def call_model(model_name, query_text, max_retries=3):
    """Send one query to one model and return the response text."""
    spec = MODEL_REGISTRY[model_name]
    fn = _DISPATCH[spec["provider"]]
    last_err = None
    for attempt in range(max_retries):
        try:
            return fn(spec["version"], query_text)
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"{model_name} failed after {max_retries} attempts: {last_err}")


def active_models():
    """Model names flagged active in the registry (the pilot set)."""
    return [name for name, spec in MODEL_REGISTRY.items() if spec["active"]]


def run_all(models=None, queries_path="data/queries.csv", pilot_n=None,
            stratify=False, per_life_topic=5, life_topics=None, n_control=None,
            out_path="results/model_responses.csv"):
    """Run every query through every model and collect responses.

    If stratify is True, the pilot is a topic-balanced sample (per_life_topic from
    each included life topic + n_control controls); otherwise a uniform pilot_n
    random sample is used.
    """
    models = models or active_models()
    if stratify:
        queries = stratified_sample(
            pd.read_csv(queries_path), per_life_topic=per_life_topic,
            life_topics=life_topics, n_control=n_control,
        )
    else:
        queries = load_queries(queries_path, pilot_n=pilot_n)
    rows = []
    for model_name in models:
        version = MODEL_REGISTRY[model_name]["version"]
        for _, q in queries.iterrows():
            response = call_model(model_name, q["query_text"])
            rows.append({
                "query_id": q["query_id"],
                "query_text": q["query_text"],
                "topic": q.get("topic"),
                "is_control": q.get("is_control"),
                "model_name": model_name,
                "model_version": version,
                "response_text": response,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            print(f"{model_name} <- {q['query_id']}")
    responses = pd.DataFrame(rows)
    responses.to_csv(out_path, index=False)
    return responses


def main():
    parser = argparse.ArgumentParser(description="Run the query set through the active models.")
    parser.add_argument("--queries", default="data/queries.csv")
    parser.add_argument("--pilot-n", type=int, default=10,
                        help="Uniform-random sample of this many queries (used only "
                             "when --stratify is not set).")
    parser.add_argument("--stratify", action="store_true",
                        help="Topic-balanced pilot: --per-life-topic from each "
                             "--life-topics topic, plus --n-control controls.")
    parser.add_argument("--per-life-topic", type=int, default=5,
                        help="With --stratify: queries per included life-science topic.")
    parser.add_argument("--life-topics", default=None,
                        help="With --stratify: comma-separated life-science topics to "
                             "include (default: all). E.g. infectious_disease_diagnostics")
    parser.add_argument("--n-control", type=int, default=None,
                        help="With --stratify: number of control queries (default: all).")
    parser.add_argument("--out", default="results/model_responses.csv")
    args = parser.parse_args()
    life_topics = [t.strip() for t in args.life_topics.split(",")] if args.life_topics else None
    responses = run_all(
        queries_path=args.queries, pilot_n=args.pilot_n, stratify=args.stratify,
        per_life_topic=args.per_life_topic, life_topics=life_topics,
        n_control=args.n_control, out_path=args.out,
    )
    print(f"\nCollected {len(responses)} responses from {sorted(responses['model_name'].unique())} "
          f"-> {args.out}")


if __name__ == "__main__":
    main()
