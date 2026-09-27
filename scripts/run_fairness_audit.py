"""Run the subgroup fairness audit on the held-out test set.

Reads protected attributes from the isolated source and joins them to predictions
*for auditing only*. This is the one place in the codebase that touches them; they
are never features (ETHICS.md).

Usage::

    python scripts/run_fairness_audit.py
    python scripts/run_fairness_audit.py --budget 0.10
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from dropout_ews.config.settings import (
    DATA_DIR,
    REPORTS_DIR,
    load_feature_config,
    load_threshold_config,
)
from dropout_ews.data.checkpoints import build_checkpoint_rows
from dropout_ews.data.loaders import load_student_info
from dropout_ews.evaluation.fairness import (
    MIN_POSITIVES_FOR_RECALL,
    PROTECTED_ATTRIBUTES,
    audit,
    disparities,
    inconclusive_groups,
)
from dropout_ews.evaluation.metrics import metrics_at_threshold
from dropout_ews.evaluation.splits import make_temporal_split
from dropout_ews.models.registry import load_model

KEY_COLUMNS = ["code_module", "code_presentation", "id_student"]
DROP_COLUMNS = ("label", "date_unregistration", "split", "row_id")
OUT_DIR = REPORTS_DIR / "fairness"
REPORT_PATH = REPORTS_DIR / "fairness_audit.md"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--budget",
        type=float,
        default=None,
        help="alert budget defining the operating point; defaults to features.yaml",
    )
    args = parser.parse_args()

    config = load_feature_config()
    budget = args.budget if args.budget is not None else config.evaluation.alert_budget
    thresholds = load_threshold_config()

    frame = pd.read_parquet(DATA_DIR / "processed" / "features.parquet")
    population, _ = build_checkpoint_rows()
    frame.index = population.index
    frame["code_presentation"] = population["code_presentation"].to_numpy()
    split = make_temporal_split(population)

    model, metadata = load_model()
    test = frame.loc[split.test].copy()
    features = test.drop(columns=[c for c in DROP_COLUMNS if c in test.columns])
    test["probability"] = model.predict_proba(features)[:, 1]

    # The operating point staff actually use: the top `budget` share of the
    # cohort, taken positionally so ties do not inflate it (see budget_sweep).
    order = np.argsort(-test["probability"].to_numpy(), kind="mergesort")
    k = max(1, round(budget * len(test)))
    threshold = float(test["probability"].to_numpy()[order[k - 1]])

    overall = metrics_at_threshold(
        test["label"].to_numpy(), test["probability"].to_numpy(), threshold
    )

    # Protected attributes, read only here.
    demographics = load_student_info()[[*KEY_COLUMNS, *PROTECTED_ATTRIBUTES]]
    audited = test.merge(demographics, on=KEY_COLUMNS, how="left", validate="many_to_one")

    print(f"Model {metadata.model_version}")
    print(f"Operating point: top {budget:.0%} of the cohort, cutoff {threshold:.5f}")
    print(
        f"Overall: recall {overall.recall:.3f}, precision {overall.precision:.4f}, "
        f"FNR {1 - overall.recall:.3f}, flagged {overall.flagged_share:.1%}"
    )

    print("\n[1/3] Per-subgroup error rates...")
    audit_frame = audit(audited, threshold)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    audit_frame.to_csv(OUT_DIR / "subgroup_metrics.csv", index=False)
    display = audit_frame[
        [
            "attribute",
            "value",
            "rows",
            "positives",
            "base_rate",
            "alert_rate",
            "recall",
            "false_negative_rate",
            "conclusive",
        ]
    ]
    print(display.to_string(index=False))

    print("\n[2/3] Largest gaps, with an overlap check...")
    fnr_gaps = disparities(audit_frame, "false_negative_rate")
    fpr_gaps = disparities(audit_frame, "false_positive_rate")
    fnr_gaps.to_csv(OUT_DIR / "fnr_disparities.csv", index=False)
    fpr_gaps.to_csv(OUT_DIR / "fpr_disparities.csv", index=False)
    print("False-negative rate (a missed at-risk student):")
    print(fnr_gaps.to_string(index=False) if len(fnr_gaps) else "  none assessable")

    print("\n[3/3] Groups too small to assess...")
    inconclusive = inconclusive_groups(audit_frame)
    inconclusive.to_csv(OUT_DIR / "inconclusive_groups.csv", index=False)
    print(inconclusive.to_string(index=False) if len(inconclusive) else "  none")

    _write_report(
        metadata.model_version,
        budget,
        threshold,
        overall,
        audit_frame,
        fnr_gaps,
        fpr_gaps,
        inconclusive,
        thresholds.calibrated,
    )
    print(f"\nReport: {REPORT_PATH}")
    return 0


def _write_report(
    model_version: str,
    budget: float,
    threshold: float,
    overall: object,
    audit_frame: pd.DataFrame,
    fnr_gaps: pd.DataFrame,
    fpr_gaps: pd.DataFrame,
    inconclusive: pd.DataFrame,
    calibrated: bool,
) -> None:
    conclusive_fnr = fnr_gaps[fnr_gaps["distinguishable_from_noise"]] if len(fnr_gaps) else fnr_gaps
    conclusive_fpr = fpr_gaps[fpr_gaps["distinguishable_from_noise"]] if len(fpr_gaps) else fpr_gaps

    lines = [
        "# Fairness audit",
        "",
        f"Model `{model_version}`, held-out test presentation. Generated by "
        "`scripts/run_fairness_audit.py`.",
        "",
        "## How to read this",
        "",
        "**Alert-rate differences are expected and are not the finding.** A "
        "well-calibrated model flags higher-base-rate groups more often, because "
        "those students really do withdraw more often. Phase 3 recorded those base "
        "rates before any model existed, precisely so this audit could not mistake "
        "them for model behaviour. Demographic parity is the wrong test here, and "
        "chasing it would mean withholding support from the group that needs it most.",
        "",
        "**The question is whether errors differ**, and specifically the "
        "false-negative rate: an at-risk student the model misses receives no offer "
        "of help. A higher false-positive rate mostly costs staff time, which "
        "matters but is not the same kind of harm.",
        "",
        "**A gap whose confidence intervals overlap is not a finding.** Reporting "
        "one anyway is how an audit manufactures conclusions, so overlap is checked "
        "and flagged.",
        "",
        "## Operating point",
        "",
        f"- Alert budget: **{budget:.0%}** of the cohort",
        f"- Cutoff probability: **{threshold:.5f}**",
        f"- Overall recall: **{overall.recall:.1%}**, precision "  # type: ignore[attr-defined]
        f"**{overall.precision:.1%}**, false-negative rate "  # type: ignore[attr-defined]
        f"**{1 - overall.recall:.1%}**",  # type: ignore[attr-defined]
        f"- Risk bands calibrated: **{calibrated}**",
        "",
        "The overall false-negative rate is the context for everything below: the "
        "model already misses most students who withdraw, in every group. Subgroup "
        "gaps are differences in how that failure is distributed, not the difference "
        "between working and not working.",
        "",
        "## Per-subgroup error rates",
        "",
        audit_frame[
            [
                "attribute",
                "value",
                "rows",
                "positives",
                "base_rate",
                "alert_rate",
                "recall",
                "false_negative_rate",
                "fnr_ci_low",
                "fnr_ci_high",
                "precision",
                "calibration_gap",
                "conclusive",
            ]
        ].to_markdown(index=False),
        "",
        "## Largest false-negative gaps",
        "",
        fnr_gaps.to_markdown(index=False)
        if len(fnr_gaps)
        else "_No attribute had two assessable groups._",
        "",
    ]

    if len(conclusive_fnr):
        lines += [
            "**Gaps that survive the overlap check:**",
            "",
            *(
                f"- `{row['attribute']}`: the false-negative rate is "
                f"{row['worst_rate']:.1%} for **{row['worst_group']}** against "
                f"{row['best_rate']:.1%} for **{row['best_group']}**, a gap of "
                f"{row['gap']:.1%}."
                for _, row in conclusive_fnr.iterrows()
            ),
            "",
        ]
    else:
        lines += [
            "**No false-negative gap survives the overlap check.** Every difference "
            "measured here is within sampling noise at this sample size. That is not "
            "the same as evidence of fairness — it is an absence of evidence of "
            "disparity, and the distinction matters.",
            "",
        ]

    lines += [
        "## Largest false-positive gaps",
        "",
        fpr_gaps.to_markdown(index=False) if len(fpr_gaps) else "_Not assessable._",
        "",
    ]
    if len(conclusive_fpr):
        lines += [
            "A higher false-positive rate means more unnecessary outreach for that "
            "group. It costs staff time and can feel intrusive to a student who is "
            "not struggling, but it does not deny anyone support.",
            "",
        ]

    lines += [
        "## Groups too small to assess",
        "",
        inconclusive.to_markdown(index=False)
        if len(inconclusive)
        else "_All groups met the minimum size._",
        "",
        f"Recall is not reported as a finding below {MIN_POSITIVES_FOR_RECALL} "
        "positives, because the interval is too wide to support a claim. These groups "
        "are listed rather than omitted: an audit that silently drops what it could "
        "not assess reads as though it assessed everything.",
        "",
        "## What this audit does not establish",
        "",
        "- **Excluding protected attributes from the features does not make the model "
        "fair.** Engagement correlates with employment, caring responsibilities and "
        "connectivity, so proxies exist whatever the feature list says. That is why "
        "error rates are measured rather than assumed.",
        "- **One institution, one cohort, 2013-2014.** These results do not transfer.",
        "- **The label under-counts disengagement.** It records formal "
        "de-registration, and students who stop engaging without withdrawing are "
        "labelled negative. That under-counting is unlikely to be uniform across "
        "groups, which would bias every rate here in a direction this audit cannot "
        "measure.",
        "- **Intersections are not tested.** Each attribute is assessed marginally. "
        "A disparity affecting a combination of attributes would not appear, and the "
        "cohort is not large enough to test intersections at a 3% positive rate.",
        "",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
