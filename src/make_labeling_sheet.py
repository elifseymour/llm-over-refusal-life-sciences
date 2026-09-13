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

Extending the sheet after adding more responses: re-run with --skip-existing.
That KEEPS every row you have already hand-labeled and appends blank rows only
for the new responses. Without --skip-existing the script refuses to overwrite a
sheet that already contains labels, so your work cannot be clobbered by accident.
"""

import os
import argparse
import pandas as pd

LABEL_HELP = "refusal | useful | not_useful"
CAUSE_HELP = "caution_driven | other (only if not_useful)"


def _labeled_count(df):
    """How many rows in an existing sheet already carry a hand label."""
    if df is None or "expert_label" not in df.columns:
        return 0
    return int(df["expert_label"].fillna("").astype(str).str.strip().ne("").sum())


def build_sheet(responses_path="results/model_responses.csv",
                out_path="results/expert_labels.csv", shuffle=True, seed=13,
                skip_existing=False):
    """Build (or extend) the blind expert-labeling sheet.

    Default: write a fresh sheet with one blank row per response. To avoid
    destroying hand labels, this refuses to overwrite an existing sheet that
    already has any filled-in expert_label — use skip_existing instead.

    skip_existing=True (resume/merge): keep every existing row AND its labels,
    and append blank rows only for (query_id, model_name) pairs not already in
    the sheet. This is the safe way to extend the sheet after adding more
    responses without losing labeling work already done.
    """
    if not os.path.exists(responses_path):
        raise FileNotFoundError(
            f"{responses_path} not found. Run the pilot first (Stage 1 of the "
            f"notebook, or src/run_models.py) so there are responses to label."
        )
    resp = pd.read_csv(responses_path)
    cols = ["query_id", "model_name", "query_text", "response_text"]
    sheet = resp[cols].copy()

    prior = pd.read_csv(out_path) if os.path.exists(out_path) else None
    done_pairs = set()
    if prior is not None and {"query_id", "model_name"}.issubset(prior.columns):
        done_pairs = set(zip(prior["query_id"].astype(str), prior["model_name"].astype(str)))
    labeled = _labeled_count(prior)

    if not skip_existing:
        # Guard: never silently wipe hand-labeled rows.
        if labeled > 0:
            raise SystemExit(
                f"{out_path} already has {labeled} hand-labeled row(s). Refusing to "
                f"overwrite. Re-run with --skip-existing to keep those labels and only "
                f"append blank rows for newly added responses."
            )
        new_sheet = sheet
        if shuffle:
            # Randomize order so labeling isn't anchored by model or topic ordering.
            new_sheet = new_sheet.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        new_sheet = new_sheet.assign(expert_label="", expert_cause_tag="")
        new_sheet.to_csv(out_path, index=False)
        print(f"Wrote {len(new_sheet)} rows to {out_path}")
        print(f"Fill 'expert_label' ({LABEL_HELP})")
        print(f"and 'expert_cause_tag' ({CAUSE_HELP}), then save.")
        return new_sheet

    # Resume/merge: only rows whose (query_id, model_name) isn't already present.
    is_new = [
        (str(qid), str(m)) not in done_pairs
        for qid, m in zip(sheet["query_id"], sheet["model_name"])
    ]
    new_sheet = sheet[is_new].copy()
    if shuffle:
        new_sheet = new_sheet.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    new_sheet = new_sheet.assign(expert_label="", expert_cause_tag="")
    combined = pd.concat([prior, new_sheet], ignore_index=True) if prior is not None else new_sheet
    combined.to_csv(out_path, index=False)
    print(f"Appended {len(new_sheet)} new blank rows (kept {0 if prior is None else len(prior)} "
          f"existing, {labeled} already labeled); {out_path} now holds {len(combined)} rows.")
    return new_sheet


def main():
    p = argparse.ArgumentParser(description="Build the blind expert-labeling sheet.")
    p.add_argument("--responses", default="results/model_responses.csv")
    p.add_argument("--out", default="results/expert_labels.csv")
    p.add_argument("--no-shuffle", action="store_true", help="Keep original row order.")
    p.add_argument("--skip-existing", action="store_true",
                   help="Resume/merge: keep existing rows and their labels, and append "
                        "blank rows only for responses not already in the sheet.")
    args = p.parse_args()
    build_sheet(args.responses, args.out, shuffle=not args.no_shuffle,
                skip_existing=args.skip_existing)


if __name__ == "__main__":
    main()
