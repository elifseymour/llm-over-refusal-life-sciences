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
    # temperature is deprecated for the Claude 5 family, so it is not sent.
    msg = client.messages.create(
        model=version,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": query_text}],
    )
    text = "".join(block.text for block in msg.content if block.type == "text")
    # stop_reason is returned so a classifier refusal (stop_reason == "refusal",
    # which yields no text blocks) is captured rather than saved as blank text.
    return text, msg.stop_reason


def _call_openai(version, query_text):
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model=version,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": query_text}],
    )
    return resp.choices[0].message.content, resp.choices[0].finish_reason


def _call_google(version, query_text):
    import google.generativeai as genai

    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel(version)
    resp = model.generate_content(query_text)
    finish = None
    if resp.candidates:
        finish = getattr(resp.candidates[0], "finish_reason", None)
    return resp.text, finish


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
    return resp.choices[0].message.content, resp.choices[0].finish_reason


_DISPATCH = {
    "anthropic": _call_anthropic,
    "openai": _call_openai,
    "google": _call_google,
    "openrouter": _call_openrouter,
}


# Stop/finish reasons that indicate the provider declined the request rather
# than answering it. A refusal yields no usable text, so we record it explicitly.
REFUSAL_STOP_REASONS = {"refusal", "content_filter"}


def call_model(model_name, query_text, max_retries=3):
    """Send one query to one model and return (response_text, stop_reason)."""
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


def _existing_pairs(out_path):
    """Return the set of (query_id, model_name) already present in out_path."""
    if not os.path.exists(out_path):
        return set(), None
    prior = pd.read_csv(out_path)
    if not {"query_id", "model_name"}.issubset(prior.columns):
        return set(), prior
    pairs = set(zip(prior["query_id"].astype(str), prior["model_name"].astype(str)))
    return pairs, prior


def run_all(models=None, queries_path="data/queries.csv", pilot_n=None,
            stratify=False, per_life_topic=5, life_topics=None, n_control=None,
            out_path="results/model_responses.csv", skip_existing=False):
    """Run every query through every model and collect responses.

    If stratify is True, the pilot is a topic-balanced sample (per_life_topic from
    each included life topic + n_control controls); otherwise a uniform pilot_n
    random sample is used.

    If skip_existing is True, any (query_id, model_name) already present in
    out_path is skipped and the new rows are APPENDED to it (resume mode), so a
    partial run can be extended to more queries without re-calling the API on
    responses already collected. Otherwise out_path is overwritten.
    """
    models = models or active_models()
    if stratify:
        queries = stratified_sample(
            pd.read_csv(queries_path), per_life_topic=per_life_topic,
            life_topics=life_topics, n_control=n_control,
        )
    else:
        queries = load_queries(queries_path, pilot_n=pilot_n)

    done_pairs, prior = (set(), None)
    if skip_existing:
        done_pairs, prior = _existing_pairs(out_path)

    rows = []
    skipped = 0
    for model_name in models:
        version = MODEL_REGISTRY[model_name]["version"]
        for _, q in queries.iterrows():
            if (str(q["query_id"]), str(model_name)) in done_pairs:
                skipped += 1
                print(f"{model_name} <- {q['query_id']}  [skip: already present]")
                continue
            response, stop_reason = call_model(model_name, q["query_text"])
            # A classifier refusal (stop_reason == "refusal") or an otherwise empty
            # response is recorded with an explicit marker so it is never confused
            # with an errored/blank response when scoring or hand-labeling.
            refused = stop_reason in REFUSAL_STOP_REASONS or not (response and response.strip())
            if refused:
                if stop_reason is None:
                    stop_reason = "empty_response"
                response = (
                    "[REFUSAL — the model returned no content; "
                    f"stop_reason={stop_reason}]"
                )
            rows.append({
                "query_id": q["query_id"],
                "query_text": q["query_text"],
                "topic": q.get("topic"),
                "is_control": q.get("is_control"),
                "model_name": model_name,
                "model_version": version,
                "response_text": response,
                "stop_reason": stop_reason,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            print(f"{model_name} <- {q['query_id']}"
                  + (f"  [{stop_reason}]" if refused else ""))

    new_rows = pd.DataFrame(rows)
    if skip_existing and prior is not None:
        combined = pd.concat([prior, new_rows], ignore_index=True)
    else:
        combined = new_rows
    combined.to_csv(out_path, index=False)
    if skip_existing:
        print(f"\nAppended {len(new_rows)} new rows (skipped {skipped} already present); "
              f"{out_path} now holds {len(combined)} rows.")
    return new_rows


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
    parser.add_argument("--skip-existing", action="store_true",
                        help="Resume mode: skip (query_id, model) pairs already in "
                             "--out and append only the new rows (instead of overwriting).")
    args = parser.parse_args()
    life_topics = [t.strip() for t in args.life_topics.split(",")] if args.life_topics else None
    responses = run_all(
        queries_path=args.queries, pilot_n=args.pilot_n, stratify=args.stratify,
        per_life_topic=args.per_life_topic, life_topics=life_topics,
        n_control=args.n_control, out_path=args.out, skip_existing=args.skip_existing,
    )
    print(f"\nCollected {len(responses)} new responses"
          + (f" from {sorted(responses['model_name'].unique())}" if len(responses) else "")
          + f" -> {args.out}")


if __name__ == "__main__":
    main()
