"""
make_labeling_sheet.py

Turns the collected model responses into a blind sheet for expert hand-labeling.

Run this AFTER the pilot (once results/model_responses.csv exists). It writes
results/expert_labels.csv with one row per response, each already carrying the
question and the model's answer, plus two blank columns for you to fill in:

    expert_label       : refusal | useful | not_useful
    expert_cause_tag   : caution_driven | other   (fill ONLY when not_useful)

The sheet does NOT include the judge's label, on purpose. Labeling blind keeps
your judgment independent, which is what makes the judge-vs-expert agreement
(Cohen's kappa) meaningful.

How to use:
    1. python src/make_labeling_sheet.py
    2. Open results/expert_labels.csv (e.g. in Excel or Numbers).
    3. For each row, read the response and type your expert_label. If it is
       not_useful, also type the expert_cause_tag. Leave cause blank otherwise.
    4. Save the file (keep it as CSV).
    5. In the analysis notebook, the kappa cell reads this file and compares your
       labels to the judge's.

Tip: for the pilot you only need to label the responses you actually ran. If you
want to label fewer than all of them, keep or delete whole rows before saving.
"""

import os
import argparse
import pandas as pd

LABEL_HELP = "refusal | useful | not_useful"
CAUSE_HELP = "caution_driven | other (only if not_useful)"


def build_sheet(responses_path="results/model_responses.csv",
                out_path="results/expert_labels.csv", shuffle=True, seed=13):
    if not os.path.exists(responses_path):
        raise FileNotFoundError(
            f"{responses_path} not found. Run the pilot first (Stage 1 of the "
            f"notebook, or src/run_models.py) so there are responses to label."
        )
    resp = pd.read_csv(responses_path)
    cols = ["query_id", "model_name", "query_text", "response_text"]
    sheet = resp[cols].copy()
    if shuffle:
        # Randomize order so labeling isn't anchored by model or topic ordering.
        sheet = sheet.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    sheet["expert_label"] = ""       # fill: refusal | useful | not_useful
    sheet["expert_cause_tag"] = ""   # fill only when not_useful: caution_driven | other
    sheet.to_csv(out_path, index=False)
    print(f"Wrote {len(sheet)} rows to {out_path}")
    print(f"Fill 'expert_label' ({LABEL_HELP})")
    print(f"and 'expert_cause_tag' ({CAUSE_HELP}), then save.")
    return sheet


def main():
    p = argparse.ArgumentParser(description="Build the blind expert-labeling sheet.")
    p.add_argument("--responses", default="results/model_responses.csv")
    p.add_argument("--out", default="results/expert_labels.csv")
    p.add_argument("--no-shuffle", action="store_true", help="Keep original row order.")
    args = p.parse_args()
    build_sheet(args.responses, args.out, shuffle=not args.no_shuffle)


if __name__ == "__main__":
    main()
