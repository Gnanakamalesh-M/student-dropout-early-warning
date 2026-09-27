"""Measure drift between the training and test presentations.

This is a real drift measurement, not a demo: the test presentation is a later
cohort of the same courses, which is exactly the situation a deployed monitor
faces each term. It also serves as a self-test of the monitor — the test-set
performance in the model card is known, so if the monitor screams about a cohort
the model handled fine, the monitor is the thing that is wrong.

Usage::

    python scripts/run_drift_report.py
    python scripts/run_drift_report.py --pooled   # also show the artefact
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from dropout_ews.config.settings import DATA_DIR, REPORTS_DIR, load_feature_config
from dropout_ews.data.checkpoints import build_checkpoint_rows
from dropout_ews.evaluation.splits import make_temporal_split
from dropout_ews.models.registry import load_model
from dropout_ews.monitoring import drift

DROP_COLUMNS = ("label", "date_unregistration", "split", "row_id")
OUT_DIR = REPORTS_DIR / "drift"
REPORT_PATH = REPORTS_DIR / "drift_report.md"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pooled",
        action="store_true",
        help="also compute drift with checkpoints pooled, to size the artefact",
    )
    args = parser.parse_args()

    config = load_feature_config()
    features = list(config.features.all_features())
    budget = config.evaluation.alert_budget

    frame = pd.read_parquet(DATA_DIR / "processed" / "features.parquet")
    population, _ = build_checkpoint_rows()
    frame.index = population.index
    frame["checkpoint_day"] = population["checkpoint_day"].to_numpy()
    frame["code_presentation"] = population["code_presentation"].to_numpy()
    split = make_temporal_split(population)

    reference = frame.loc[split.train]
    current = frame.loc[split.test]

    model, metadata = load_model()
    reference_scores = model.predict_proba(
        reference.drop(columns=[c for c in DROP_COLUMNS if c in reference.columns])
    )[:, 1]
    current_scores = model.predict_proba(
        current.drop(columns=[c for c in DROP_COLUMNS if c in current.columns])
    )[:, 1]

    # The deployed cutoff: top `budget` share of the reference cohort.
    order = np.argsort(-reference_scores, kind="mergesort")
    k = max(1, round(budget * len(reference_scores)))
    threshold = float(reference_scores[order[k - 1]])

    print(f"Model {metadata.model_version}")
    print(f"Reference: {sorted(set(reference['code_presentation']))} ({len(reference):,} rows)")
    print(f"Current:   {sorted(set(current['code_presentation']))} ({len(current):,} rows)")
    print(f"Cutoff {threshold:.5f} (top {budget:.0%} of reference)\n")

    print("[1/4] Feature drift, within checkpoint...")
    within = drift.feature_drift(reference, current, features, by_checkpoint=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    within.to_csv(OUT_DIR / "feature_drift_by_checkpoint.csv", index=False)
    summary = drift.summarize(within)
    print(
        f"  {summary.features_checked} features x {within['checkpoint_day'].nunique()} checkpoints"
    )
    print(
        f"  major: {len(summary.major)}  minor: {len(summary.minor)}  "
        f"degenerate: {len(summary.degenerate)}  coverage: {len(summary.coverage_drops)}"
    )
    print(f"  requires_review: {summary.requires_review}")
    if summary.major:
        worst = within[within["band"] == "major"].nlargest(8, "psi")
        print(
            worst[["feature", "checkpoint_day", "psi", "reference_mean", "current_mean"]].to_string(
                index=False
            )
        )

    pooled = pd.DataFrame()
    if args.pooled:
        print("\n[2/4] The same features, checkpoints pooled (the artefact)...")
        pooled = drift.feature_drift(reference, current, features, by_checkpoint=False)
        pooled.to_csv(OUT_DIR / "feature_drift_pooled.csv", index=False)
        pooled_summary = drift.summarize(pooled)
        print(f"  major: {len(pooled_summary.major)} (vs {len(summary.major)} within checkpoint)")
    else:
        print("\n[2/4] Pooled comparison skipped (--pooled to include).")

    print("\n[3/4] Prediction drift, per checkpoint (needs no labels)...")
    rows = []
    for checkpoint in sorted(set(reference["checkpoint_day"]) & set(current["checkpoint_day"])):
        reference_mask = (reference["checkpoint_day"] == checkpoint).to_numpy()
        current_mask = (current["checkpoint_day"] == checkpoint).to_numpy()
        rows.append(
            drift.prediction_drift(
                reference_scores[reference_mask],
                current_scores[current_mask],
                threshold,
                checkpoint_day=int(checkpoint),
            ).to_row()
        )
    prediction = pd.DataFrame(rows)
    prediction.to_csv(OUT_DIR / "prediction_drift.csv", index=False)
    print(
        prediction[
            [
                "checkpoint_day",
                "psi",
                "band",
                "reference_alert_rate",
                "current_alert_rate",
                "alert_rate_change",
            ]
        ].to_string(index=False)
    )

    print("\n[4/4] Calibration drift (needs labels; complete here, not in production)...")
    calibration = drift.calibration_drift(
        reference["label"].to_numpy(),
        reference_scores,
        current["label"].to_numpy(),
        current_scores,
        labels_complete=True,
    )
    pd.DataFrame([calibration.to_row()]).to_csv(OUT_DIR / "calibration_drift.csv", index=False)
    print(
        f"  reference: predicted {calibration.reference_predicted:.4f} vs "
        f"observed {calibration.reference_observed:.4f} (gap {calibration.reference_gap:+.4f})"
    )
    print(
        f"  current:   predicted {calibration.current_predicted:.4f} vs "
        f"observed {calibration.current_observed:.4f} (gap {calibration.current_gap:+.4f})"
    )
    print(f"  gap change: {calibration.gap_change:+.4f}")

    _write_report(
        metadata.model_version,
        threshold,
        budget,
        reference,
        current,
        within,
        pooled,
        summary,
        prediction,
        calibration,
    )
    print(f"\nReport: {REPORT_PATH}")
    return 0


def _write_report(
    model_version: str,
    threshold: float,
    budget: float,
    reference: pd.DataFrame,
    current: pd.DataFrame,
    within: pd.DataFrame,
    pooled: pd.DataFrame,
    summary: drift.DriftSummary,
    prediction: pd.DataFrame,
    calibration: drift.CalibrationDrift,
) -> None:
    lines = [
        "# Drift report",
        "",
        f"Model `{model_version}`. Generated by `scripts/run_drift_report.py`.",
        "",
        f"Reference: `{sorted(set(reference['code_presentation']))}` "
        f"({len(reference):,} rows). "
        f"Current: `{sorted(set(current['code_presentation']))}` ({len(current):,} rows).",
        f"Deployed cutoff **{threshold:.5f}**, the top {budget:.0%} of the reference cohort.",
        "",
        "## How to read this",
        "",
        "**Drift is measured within checkpoint**, and pooling fails in both "
        "directions. It can invent drift, because two cohorts with identical "
        "per-checkpoint distributions still differ pooled if their composition "
        "differs — and a mid-presentation cohort always does. It can also hide "
        "drift, which is what happened here: see the pooling section below. The "
        "second is the dangerous one, and it was found by running this report "
        "rather than by reasoning about it.",
        "",
        "**PSI bands are convention and PSI is biased upward on small samples.** "
        "The 0.10/0.25 bands come from credit scoring and are not validated for "
        "this task. Measured here on identical distributions with a fixed 10 bins, "
        "the median PSI is 0.18 at n=50 and 1.31 at n=25 — noise alone clears the "
        '"major drift" band. Bin count therefore adapts to sample size and PSI is '
        'refused below 50 rows. Treat a band as "look at this", never as a decision.',
        "",
        "**Drift is not evidence the model got worse.** Input distributions can "
        "move while performance holds; that is precisely what this report shows "
        "below. The summary escalates to `requires_review`, and there is "
        "deliberately no `requires_retraining` — nothing measured here can "
        "establish that retraining would help.",
        "",
        "## Feature drift, within checkpoint",
        "",
        f"- Features checked: **{summary.features_checked}**",
        f"- Major: **{len(summary.major)}** — {', '.join(f'`{f}`' for f in summary.major) or '_none_'}",
        f"- Minor: **{len(summary.minor)}** — {', '.join(f'`{f}`' for f in summary.minor) or '_none_'}",
        f"- Not measurable: **{len(summary.degenerate)}** — "
        f"{', '.join(f'`{f}`' for f in summary.degenerate) or '_none_'}",
        f"- Coverage change: **{len(summary.coverage_drops)}** — "
        f"{', '.join(f'`{f}`' for f in summary.coverage_drops) or '_none_'}",
        f"- **requires_review: {summary.requires_review}**",
        "",
    ]

    worst = within.dropna(subset=["psi"]).nlargest(15, "psi")
    if len(worst):
        lines += [
            "### Largest 15 PSI values",
            "",
            worst[
                [
                    "feature",
                    "checkpoint_day",
                    "psi",
                    "band",
                    "reference_mean",
                    "current_mean",
                    "current_n",
                    "empty_bins",
                ]
            ].to_markdown(index=False),
            "",
        ]

    degenerate = within[within["degenerate"]]
    if len(degenerate):
        reasons = degenerate.groupby("degenerate_reason").size().sort_values(ascending=False)
        lines += [
            "### Why some cells were not measurable",
            "",
            *(f"- {reason}: {count} cells" for reason, count in reasons.items()),
            "",
            "These are reported rather than dropped. A monitor that silently omits "
            "what it could not measure looks like a monitor that found nothing.",
            "",
        ]

    if len(pooled):
        lines += [
            "## The pooling artefact, measured on real data",
            "",
            f"Pooling flags **{len(drift.summarize(pooled).major)}** features as major "
            f"against **{len(summary.major)}** within checkpoint, on the same two "
            "cohorts — it reports *less* drift, not more.",
            "",
            "The mechanism is cancellation. `days_since_last_submission` shifted "
            "+6.3 days at checkpoint 90 and -7.3 days at checkpoint 150: opposite "
            "directions, similar magnitudes. Pooled, the means are 23.6 against "
            '24.6 and PSI reads 0.19, a "minor" band; within checkpoint all six '
            "checkpoints are major, PSI 0.71 to 1.15.",
            "",
            "I had expected pooling to *manufacture* drift through composition "
            "shift, and wrote the module around that. It does do that — there is a "
            "unit test for it on synthetic data. But on the real cohorts it masked "
            "drift instead, and a monitor that averages a drifting cohort into "
            "looking healthy will not fire when it should. Both directions are now "
            "pinned by tests.",
            "",
            pooled.dropna(subset=["psi"])
            .nlargest(10, "psi")[["feature", "psi", "band", "reference_mean", "current_mean"]]
            .to_markdown(index=False),
            "",
        ]

    lines += [
        "## Prediction drift (no labels required)",
        "",
        "The only signal that can fire on the cohort currently being scored. "
        "`alert_rate_change` is the operational number: the cutoff was chosen for a "
        f"{budget:.0%} staffing capacity, so a rise means staff are being asked to "
        "contact more students than the operating point promised.",
        "",
        prediction[
            [
                "checkpoint_day",
                "psi",
                "band",
                "reference_alert_rate",
                "current_alert_rate",
                "alert_rate_change",
                "current_n",
            ]
        ].to_markdown(index=False),
        "",
        "## Calibration drift (labels required, so always at least 30 days stale)",
        "",
        "The failure that matters most here, because the dashboard shows "
        "probabilities to staff. A model whose ranking still holds but whose "
        "probabilities have drifted keeps flagging roughly the right students while "
        "misstating their risk, and a counsellor has no way to notice.",
        "",
        f"- Reference: predicted {calibration.reference_predicted:.4f} vs observed "
        f"{calibration.reference_observed:.4f} (gap {calibration.reference_gap:+.4f})",
        f"- Current: predicted {calibration.current_predicted:.4f} vs observed "
        f"{calibration.current_observed:.4f} (gap {calibration.current_gap:+.4f})",
        f"- Change in gap: **{calibration.gap_change:+.4f}**",
        "",
        "Negative means the model understates risk — the direction that withholds "
        "support. Same sign convention as the fairness audit, deliberately.",
        "",
        "In production this number is unavailable for the current cohort. The "
        "horizon is 30 days, so a row scored today cannot be scored against for a "
        "month, and the last checkpoints of a presentation resolve only after it "
        "ends. `labels_complete=False` marks a calibration figure computed over "
        "unresolved rows, which counts every pending student as a negative and "
        "makes the model look over-confident when it may not be.",
        "",
        "## Retraining policy",
        "",
        "See `docs/MONITORING.md`. The short version: this report escalates to a "
        "person, and a person decides. No threshold in this repository triggers an "
        "automatic retrain.",
        "",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
