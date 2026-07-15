"""
analysis_utils.py

Helper functions for computing metrics and figures from the scored data.

Scoring scheme (see rubrics/scoring_rubric.md):
    primary label in {refusal, useful, not_useful}
    cause_tag in {caution_driven, other} when label == not_useful, else null

Metrics:
    - false-refusal rate per model (overall and by topic)
    - not-useful rate per model (overall and by topic)
    - combined unhelpful rate (refusal OR not_useful)
    - caution-driven share of not-useful responses per model
    - life-science vs control comparison
    - judge-vs-expert agreement (Cohen's kappa) on the validation subset

The pilot run is a small demonstration, so treat rates as illustrative, not as
findings. The number to report with confidence is judge-vs-expert agreement.
"""

import pandas as pd

REFUSAL = "refusal"
USEFUL = "useful"
NOT_USEFUL = "not_useful"


def load_scored(path="results/scored_responses.csv"):
    return pd.read_csv(path)


def _rate(df, label, group_cols):
    """Fraction of responses with the given primary label, grouped as requested."""
    group_cols = list(group_cols)
    out = (
        df.assign(_hit=(df["label"] == label).astype(float))
        .groupby(group_cols)["_hit"]
        .mean()
        .reset_index(name=f"{label}_rate")
    )
    return out


def false_refusal_rate(df, group_cols=("model_name",)):
    """Fraction labeled refusal. On a safe-by-construction set, every refusal is false."""
    return _rate(df, REFUSAL, group_cols)


def not_useful_rate(df, group_cols=("model_name",)):
    """Fraction labeled not_useful."""
    return _rate(df, NOT_USEFUL, group_cols)


def unhelpful_rate(df, group_cols=("model_name",)):
    """Combined fraction that is either refusal or not_useful."""
    group_cols = list(group_cols)
    hit = df["label"].isin([REFUSAL, NOT_USEFUL]).astype(float)
    return (
        df.assign(_hit=hit)
        .groupby(group_cols)["_hit"]
        .mean()
        .reset_index(name="unhelpful_rate")
    )


def caution_driven_share(df, group_cols=("model_name",)):
    """Among not_useful responses, the share tagged caution_driven.

    This is the check that the not-useful effect reflects over-caution rather
    than general capability. Returns NaN where there are no not_useful responses.
    """
    group_cols = list(group_cols)
    nu = df[df["label"] == NOT_USEFUL].copy()
    if nu.empty:
        return pd.DataFrame(columns=group_cols + ["caution_driven_share", "n_not_useful"])
    nu["_caution"] = (nu["cause_tag"] == "caution_driven").astype(float)
    out = (
        nu.groupby(group_cols)
        .agg(caution_driven_share=("_caution", "mean"),
             n_not_useful=("_caution", "size"))
        .reset_index()
    )
    return out


def summary_by_model(df):
    """One row per model: refusal, useful, not_useful, unhelpful rates + counts."""
    g = df.groupby("model_name")
    total = g.size().rename("n")
    ref = g["label"].apply(lambda s: (s == REFUSAL).mean()).rename("false_refusal_rate")
    use = g["label"].apply(lambda s: (s == USEFUL).mean()).rename("useful_rate")
    nu = g["label"].apply(lambda s: (s == NOT_USEFUL).mean()).rename("not_useful_rate")
    unhelp = g["label"].apply(lambda s: s.isin([REFUSAL, NOT_USEFUL]).mean()).rename("unhelpful_rate")
    out = pd.concat([total, ref, use, nu, unhelp], axis=1).reset_index()
    cds = caution_driven_share(df)
    if not cds.empty:
        out = out.merge(cds[["model_name", "caution_driven_share"]], on="model_name", how="left")
    return out


def decomposition_by_model(df):
    """Four-way decomposition of every response, as fractions of total per model.

    Segments (sum to 1 per model):
        useful                 : net helpfulness (the top-line usefulness ranking)
        refusal                : hard over-caution
        not_useful_caution     : soft over-caution (caution-driven not-useful)
        not_useful_other       : capability/quality failures (shallow, wrong, off-topic)

    Over-caution cost = refusal + not_useful_caution. Capability cost = not_useful_other.
    """
    ind = pd.DataFrame({
        "model_name": df["model_name"],
        "useful": (df["label"] == USEFUL),
        "refusal": (df["label"] == REFUSAL),
        "not_useful_caution": (df["label"] == NOT_USEFUL) & (df["cause_tag"] == "caution_driven"),
        "not_useful_other": (df["label"] == NOT_USEFUL) & (df["cause_tag"] != "caution_driven"),
    })
    g = ind.groupby("model_name")
    out = g[["useful", "refusal", "not_useful_caution", "not_useful_other"]].mean()
    out.insert(0, "n", g.size())
    out = out.reset_index()
    out["over_caution_cost"] = out["refusal"] + out["not_useful_caution"]
    out["capability_cost"] = out["not_useful_other"]
    return out


def over_caution_rate(df, group_cols=("model_name",)):
    """Fraction of responses that are safety misfires: refusal OR caution-driven not_useful.

    This is the clean measure of over-caution, since it excludes capability-driven
    ("other") not-useful answers. It is the right metric for asking whether
    biological content specifically triggers the safety machinery.
    """
    group_cols = list(group_cols)
    hit = (
        (df["label"] == REFUSAL)
        | ((df["label"] == NOT_USEFUL) & (df["cause_tag"] == "caution_driven"))
    ).astype(float)
    return (
        df.assign(_hit=hit)
        .groupby(group_cols)["_hit"]
        .mean()
        .reset_index(name="over_caution_rate")
    )


def other_not_useful_rate(df, group_cols=("model_name",)):
    """Fraction of responses that are capability-driven ('other') not_useful.

    Used as a check on the assumption that capability failures are roughly
    discipline-invariant across the life-science and control sets.
    """
    group_cols = list(group_cols)
    hit = ((df["label"] == NOT_USEFUL) & (df["cause_tag"] != "caution_driven")).astype(float)
    return (
        df.assign(_hit=hit)
        .groupby(group_cols)["_hit"]
        .mean()
        .reset_index(name="other_not_useful_rate")
    )


def lifescience_vs_control(df):
    """Safety-misfire comparison split by life-science vs control queries.

    Headline metric is over_caution_rate (refusal + caution-driven not-useful),
    which isolates the safety-misfire question the proposal is about. The
    other_not_useful_rate (capability failures) is carried alongside only as a
    check: if it is roughly flat across life_science and control, that validates
    attributing any over_caution_rate gap to biological content rather than to
    life-science questions simply being harder.

    Requires an is_control column merged onto the scored data (join on query_id
    with the query metadata if not already present).
    """
    if "is_control" not in df.columns:
        raise KeyError("is_control not in dataframe; merge query metadata on query_id first.")
    grp = df.assign(
        set=lambda d: d["is_control"].map(lambda x: "control" if bool(x) else "life_science")
    )
    keys = ["model_name", "set"]
    oc = over_caution_rate(grp, group_cols=keys)
    other = other_not_useful_rate(grp, group_cols=keys)
    n = grp.groupby(keys).size().reset_index(name="n")
    return oc.merge(other, on=keys).merge(n, on=keys)


def judge_expert_agreement(judge_labels, expert_labels):
    """Cohen's kappa and percent agreement between judge and expert primary labels.

    Pass two aligned sequences of labels for the same responses. Returns a dict.
    """
    from sklearn.metrics import cohen_kappa_score

    judge_labels = list(judge_labels)
    expert_labels = list(expert_labels)
    if len(judge_labels) != len(expert_labels):
        raise ValueError("judge_labels and expert_labels must be the same length.")
    kappa = cohen_kappa_score(expert_labels, judge_labels)
    agree = sum(a == b for a, b in zip(judge_labels, expert_labels)) / len(judge_labels)
    return {"cohen_kappa": kappa, "percent_agreement": agree, "n": len(judge_labels)}


# --- Simple plotting helpers (matplotlib) ---------------------------------

def plot_rates_by_model(summary_df, out_path="results/figures/rates_by_model.png"):
    """Grouped bar chart of false-refusal and not-useful rate per model."""
    import matplotlib.pyplot as plt

    models = summary_df["model_name"].tolist()
    x = range(len(models))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar([i - width / 2 for i in x], summary_df["false_refusal_rate"], width,
           label="False-refusal rate")
    ax.bar([i + width / 2 for i in x], summary_df["not_useful_rate"], width,
           label="Not-useful rate")
    ax.set_xticks(list(x))
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("Rate on safe queries")
    ax.set_title("Over-refusal and not-useful responses by model")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_decomposition_stacked(decomp_df, out_path="results/figures/usefulness_decomposition.png"):
    """Stacked bar per model: useful / refusal / not-useful-caution / not-useful-other.

    Reads the output of decomposition_by_model. The useful segment height is the
    net-usefulness ranking; the three failure segments show whether a model's
    shortfall is over-caution or capability.
    """
    import matplotlib.pyplot as plt

    d = decomp_df.set_index("model_name")
    segments = [
        ("useful", "Useful", "#2e7d32"),
        ("refusal", "Refusal (over-caution)", "#b71c1c"),
        ("not_useful_caution", "Not-useful: caution-driven", "#ef6c00"),
        ("not_useful_other", "Not-useful: other (capability)", "#9e9e9e"),
    ]
    models = d.index.tolist()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bottom = [0.0] * len(models)
    for col, label, color in segments:
        vals = d[col].tolist()
        ax.bar(models, vals, bottom=bottom, label=label, color=color)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_ylabel("Share of responses")
    ax.set_ylim(0, 1)
    ax.set_title("Response decomposition by model (safe queries)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
