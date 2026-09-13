"""
select_validation_subset.py

Marks a recommended validation subset in the blind expert-labeling sheet, so you
know which responses to hand-label first for the judge-vs-expert agreement
(Cohen's kappa). You do NOT have to label all responses — labeling this subset is
enough to validate the judge, and the kappa step ignores every unlabeled row.

The subset deliberately oversamples the informative rows:
  - every refusal (stop_reason == refusal / content_filter / empty_response) —
    the only refusal signal in the set,
  - every truncated response (stop_reason == max_tokens) — the most likely place
    for the judge to be wrong,
  - a stratified random sample across topic x model for ordinary rows.

Oversampling edge cases for validation is standard; just disclose it in the
writeup ("the validation subset oversampled refusals and truncated responses").

It writes a neutral 'priority' column (1 = label this one) into the sheet WITHOUT
recording why a row was chosen, so blind labeling is preserved. It never touches
expert_label / expert_cause_tag, so any labels you have already entered are safe.

Usage:
    python src/select_validation_subset.py            # marks ~50 rows
    python src/select_validation_subset.py --per-cell 5   # larger subset

Re-run after adding more responses (it overwrites the 'priority' column each time).
"""

import argparse
import pandas as pd

# stop_reason values that mean the model declined / returned nothing usable.
REFUSAL_STOP = {"refusal", "content_filter", "empty_response"}


def _pairs(df, key):
    """Set of (query_id, model_name) string tuples for a frame."""
    return {(str(a), str(b)) for a, b in df[key].itertuples(index=False)}


def select_subset(responses_path="results/model_responses.csv",
                  sheet_path="results/expert_labels.csv",
                  per_cell=3, seed=17):
    resp = pd.read_csv(responses_path)
    sheet = pd.read_csv(sheet_path)
    key = ["query_id", "model_name"]

    if "stop_reason" not in resp.columns:
        raise KeyError(
            f"{responses_path} has no 'stop_reason' column. Re-run src/run_models.py "
            f"(the current version records it) so refusals/truncations can be found."
        )

    refusal = resp[resp["stop_reason"].isin(REFUSAL_STOP)]
    trunc = resp[resp["stop_reason"] == "max_tokens"]
    forced = _pairs(refusal, key) | _pairs(trunc, key)

    # Stratified sample of the ordinary (non-forced) rows, per topic x model.
    resp_key = list(zip(resp["query_id"].astype(str), resp["model_name"].astype(str)))
    remainder = resp[[p not in forced for p in resp_key]]
    parts = []
    for _, grp in remainder.groupby(["topic", "model_name"]):
        parts.append(grp.sample(min(per_cell, len(grp)), random_state=seed))
    sampled = pd.concat(parts) if parts else remainder.iloc[0:0]
    subset = forced | _pairs(sampled, key)

    sheet_key = list(zip(sheet["query_id"].astype(str), sheet["model_name"].astype(str)))
    sheet["priority"] = [1 if p in subset else "" for p in sheet_key]
    sheet.to_csv(sheet_path, index=False)

    print(f"Marked {len(subset)} priority rows in {sheet_path}:")
    print(f"  refusals (stop_reason in {sorted(REFUSAL_STOP)}): {len(_pairs(refusal, key))}")
    print(f"  truncated (stop_reason == max_tokens):            {len(_pairs(trunc, key))}")
    print(f"  stratified sample (~{per_cell} per topic x model): {len(_pairs(sampled, key))}")
    print(f"  -> total to hand-label for validation:            {len(subset)}")
    print("\nThe 'priority' column is a neutral flag (1 = label this row first). It does "
          "not reveal why a row was picked, so blind labeling is preserved. Existing "
          "expert_label / expert_cause_tag values were left untouched.")
    return subset


def main():
    p = argparse.ArgumentParser(
        description="Mark a recommended validation subset in the labeling sheet.")
    p.add_argument("--responses", default="results/model_responses.csv")
    p.add_argument("--sheet", default="results/expert_labels.csv")
    p.add_argument("--per-cell", type=int, default=3,
                   help="Stratified sample size per (topic, model) cell for ordinary rows.")
    p.add_argument("--seed", type=int, default=17)
    args = p.parse_args()
    select_subset(args.responses, args.sheet, per_cell=args.per_cell, seed=args.seed)


if __name__ == "__main__":
    main()
