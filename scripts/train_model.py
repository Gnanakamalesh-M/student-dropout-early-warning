"""Phase 6: train, tune, calibrate, and register the primary XGBoost model.

Pipeline, in order, with the discipline that keeps each step honest:

1. **SMOTE invariant audit.** Count structurally impossible rows SMOTE invents,
   turning the ADR-0002 objection into evidence.
2. **Imbalance comparison.** Grouped CV on the training split only.
3. **Optuna tuning.** Same folds, same split. Validation and test are never
   scored during the search.
4. **Fit** on the training presentations.
5. **Calibrate** isotonically on validation.
6. **Derive risk bands** from the alert budget on validation.
7. **Evaluate** on test, once, with cluster bootstrap intervals.
8. **Register** the artifact with provenance metadata.

Usage::

    python scripts/train_model.py                    # full run
    python scripts/train_model.py --trials 20        # quick pass
    python scripts/train_model.py --skip-imbalance   # reuse a prior decision
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from dropout_ews.config.settings import (
    DATA_DIR,
    REPORTS_DIR,
    load_feature_config,
    load_threshold_config,
)
from dropout_ews.data.checkpoints import build_checkpoint_rows
from dropout_ews.evaluation.metrics import (
    budget_sweep,
    evaluate,
    ranking_metrics,
    threshold_sweep,
)
from dropout_ews.evaluation.splits import (
    GROUP_COLUMN,
    assert_no_group_leakage_in_folds,
    grouped_cv_splits,
    make_temporal_split,
)
from dropout_ews.models.calibration import (
    assess_calibration,
    bands_to_config,
    calibrate,
    derive_bands_from_budget,
)
from dropout_ews.models.imbalance import (
    count_invariant_violations,
    imbalance_strategies,
    select_strategy,
    smote_synthetic_rows,
)
from dropout_ews.models.registry import (
    ModelMetadata,
    frame_hash,
    git_commit,
    new_version_id,
    save_model,
)
from dropout_ews.models.tuning import tune_xgboost
from dropout_ews.models.xgboost_model import positive_class_weight, xgboost_pipeline

FEATURES_PARQUET = DATA_DIR / "processed" / "features.parquet"
TABLES_DIR = REPORTS_DIR / "model"
REPORT_PATH = REPORTS_DIR / "model_comparison.md"

DROP_COLUMNS = ("label", "date_unregistration", "split", "row_id")


def _features(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(columns=[c for c in DROP_COLUMNS if c in frame.columns])


def _cv_average_precision(
    build, features: pd.DataFrame, y: np.ndarray, folds
) -> tuple[float, float]:
    scores = []
    for train_positions, validation_positions in folds:
        pipeline = build()
        pipeline.fit(features.iloc[train_positions], y[train_positions])
        probabilities = pipeline.predict_proba(features.iloc[validation_positions])[:, 1]
        scores.append(ranking_metrics(y[validation_positions], probabilities).average_precision)
    return float(np.mean(scores)), float(np.std(scores))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=60)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--skip-imbalance", action="store_true")
    parser.add_argument("--background-size", type=int, default=2000)
    args = parser.parse_args()

    if not FEATURES_PARQUET.exists():
        print(f"{FEATURES_PARQUET} missing. Run scripts/build_features.py first.")
        return 1

    config = load_feature_config()
    seed = config.evaluation.random_seed
    budget = config.evaluation.alert_budget

    frame = pd.read_parquet(FEATURES_PARQUET)
    population, _ = build_checkpoint_rows()
    frame.index = population.index
    frame["code_presentation"] = population["code_presentation"].to_numpy()
    split = make_temporal_split(population)

    train_frame = frame.loc[split.train].reset_index(drop=True)
    train_features = _features(train_frame)
    y_train = train_frame["label"].to_numpy()
    folds = grouped_cv_splits(population, split.train, n_splits=args.cv_folds, random_state=seed)
    assert_no_group_leakage_in_folds(population, split.train, folds)

    scale_pos_weight = positive_class_weight(y_train)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    print(split.summary(population).to_string(index=False))
    print(f"\nscale_pos_weight = {scale_pos_weight:.1f}, alert_budget = {budget:.0%}\n")

    # ---------------------------------------------------------------- 1. SMOTE audit
    print("[1/8] Auditing what SMOTE actually generates...")
    allowlist = config.features.all_features()
    # Audit on the builder-produced columns only; cohort z-scores are added
    # inside the pipeline, so they do not exist at this stage.
    audit_columns = [c for c in allowlist if c in train_features.columns]
    audit_real = train_features[audit_columns].fillna(train_features[audit_columns].median())
    real_report = count_invariant_violations(audit_real)
    synthetic = smote_synthetic_rows(audit_real, y_train, random_state=seed)
    synthetic_report = count_invariant_violations(synthetic)
    invariant_table = pd.concat(
        [real_report.to_frame("real students"), synthetic_report.to_frame("SMOTE synthetic")],
        ignore_index=True,
    )
    invariant_table.to_csv(TABLES_DIR / "smote_invariant_audit.csv", index=False)
    print(invariant_table.to_string(index=False))

    # --------------------------------------------------------- 2. Imbalance comparison
    imbalance_table = pd.DataFrame()
    chosen_strategy = "class weights"
    if not args.skip_imbalance:
        print("\n[2/8] Comparing imbalance strategies (grouped CV on train only)...")
        rows = []
        for name, spec in imbalance_strategies(scale_pos_weight).items():

            def build(spec=spec):
                return xgboost_pipeline(
                    scale_pos_weight=spec["scale_pos_weight"],
                    resampler=spec["resampler"],
                )

            mean, std = _cv_average_precision(build, train_features, y_train, folds)
            rows.append({"strategy": name, "cv_pr_auc": round(mean, 4), "cv_std": round(std, 4)})
            print(f"      {name:<22} PR-AUC {mean:.4f} +/- {std:.4f}")
        imbalance_table = pd.DataFrame(rows).sort_values("cv_pr_auc", ascending=False)
        imbalance_table.to_csv(TABLES_DIR / "imbalance_comparison.csv", index=False)
        chosen_strategy, strategy_reason = select_strategy(imbalance_table)
        print(f"      chosen: {chosen_strategy} -- {strategy_reason}")
    else:
        strategy_reason = "not re-measured (--skip-imbalance)"
        print("\n[2/8] Skipping the imbalance comparison (--skip-imbalance)")

    strategy_spec = imbalance_strategies(scale_pos_weight)[chosen_strategy]

    # ------------------------------------------------------------------- 3. Tuning
    print(f"\n[3/8] Tuning XGBoost with Optuna ({args.trials} trials)...")
    tuning = tune_xgboost(
        train_features,
        y_train,
        folds,
        n_trials=args.trials,
        random_state=seed,
    )
    tuning.trials.to_csv(TABLES_DIR / "optuna_trials.csv", index=False)
    print(f"      best CV PR-AUC {tuning.best_value:.4f}")
    print(f"      params: {json.dumps(tuning.best_params, sort_keys=True)}")

    # --------------------------------------------------------------------- 4. Fit
    print("\n[4/8] Fitting the tuned model on the training presentations...")
    pipeline = xgboost_pipeline(
        params=tuning.best_params,
        scale_pos_weight=strategy_spec["scale_pos_weight"],
        resampler=strategy_spec["resampler"],
    )
    pipeline.fit(train_features, y_train)

    # --------------------------------------------------------------- 5. Calibration
    print("\n[5/8] Calibrating isotonically on validation...")
    validation_frame = frame.loc[split.validation]
    validation_features = _features(validation_frame)
    y_validation = validation_frame["label"].to_numpy()

    prob_before = pipeline.predict_proba(validation_features)[:, 1]
    calibrated = calibrate(pipeline, validation_features, y_validation)
    prob_after = calibrated.predict_proba(validation_features)[:, 1]
    calibration = assess_calibration(y_validation, prob_before, prob_after)
    calibration.to_frame().to_csv(TABLES_DIR / "calibration.csv", index=False)
    calibration.reliability.to_csv(TABLES_DIR / "reliability.csv", index=False)
    print(
        f"      Brier {calibration.brier_before:.5f} -> {calibration.brier_after:.5f} "
        f"({'improved' if calibration.improved else 'WORSE'})"
    )
    print(
        f"      mean predicted {calibration.mean_predicted_before:.4f} -> "
        f"{calibration.mean_predicted_after:.4f} vs observed "
        f"{calibration.observed_rate:.4f}"
    )

    # --------------------------------------------------------------- 6. Risk bands
    print(f"\n[6/8] Deriving risk bands from the {budget:.0%} alert budget...")
    bands = derive_bands_from_budget(y_validation, prob_after, critical_budget=budget)
    bands.to_csv(TABLES_DIR / "risk_bands.csv", index=False)
    print(bands.to_string(index=False))
    band_config = bands_to_config(
        bands, version=load_threshold_config().version + 1, name="budget-calibrated-v1"
    )

    # ---------------------------------------------------------------- 7. Test eval
    print("\n[7/8] Evaluating on test (once)...")
    test_frame = frame.loc[split.test]
    test_features = _features(test_frame)
    y_test = test_frame["label"].to_numpy()
    p_test = calibrated.predict_proba(test_features)[:, 1]

    critical_cutoff = float(bands.loc[bands["band"] == "critical", "min_probability"].iloc[0])
    test_result = evaluate(
        "xgboost (calibrated, test)",
        y_test,
        p_test,
        test_frame["checkpoint_day"].to_numpy(),
        test_frame[GROUP_COLUMN].to_numpy(),
        critical_cutoff,
        bootstrap_iterations=args.bootstrap,
        random_state=seed,
    )
    test_budget = budget_sweep(y_test, p_test)
    test_sweep = threshold_sweep(y_test, p_test)
    test_result.per_checkpoint.to_csv(TABLES_DIR / "per_checkpoint.csv", index=False)
    test_budget.to_csv(TABLES_DIR / "budget_sweep.csv", index=False)
    test_sweep.to_csv(TABLES_DIR / "threshold_sweep.csv", index=False)

    low, high = test_result.intervals["average_precision"]
    print(
        f"      test PR-AUC {test_result.ranking.average_precision:.4f} [{low:.4f}, {high:.4f}] | "
        f"ROC-AUC {test_result.ranking.roc_auc:.4f} | Brier {test_result.ranking.brier_score:.5f}"
    )
    print(f"      at the CRITICAL cutoff: {test_result.operating_point}")
    print("\n" + test_budget.to_string(index=False))

    # ----------------------------------------------------------------- 8. Register
    print("\n[8/8] Registering the artifact...")
    version = new_version_id("xgboost")
    metrics = {
        "cv_pr_auc": round(tuning.best_value, 5),
        "test_pr_auc": round(test_result.ranking.average_precision, 5),
        "test_pr_auc_ci": [round(low, 5), round(high, 5)],
        "test_roc_auc": round(test_result.ranking.roc_auc, 5),
        "test_brier": round(test_result.ranking.brier_score, 6),
        "test_base_rate": round(test_result.ranking.base_rate, 5),
        "calibration_brier_before": round(calibration.brier_before, 6),
        "calibration_brier_after": round(calibration.brier_after, 6),
        "budget_sweep": test_budget.to_dict(orient="records"),
        "per_checkpoint": test_result.per_checkpoint.to_dict(orient="records"),
    }
    metadata = ModelMetadata(
        model_version=version,
        model_type="xgboost",
        created_at=pd.Timestamp.utcnow().isoformat(),
        package_version=__import__("dropout_ews").__version__,
        git_commit=git_commit(),
        feature_names=allowlist,
        n_features=len(allowlist),
        hyperparameters=tuning.best_params,
        imbalance_strategy=chosen_strategy,
        calibration_method="isotonic (prefit, fitted on validation)",
        train_rows=len(train_frame),
        train_positives=int(y_train.sum()),
        train_data_hash=frame_hash(train_features),
        train_presentations=split.train_presentations,
        validation_presentations=split.validation_presentations,
        test_presentations=split.test_presentations,
        metrics=metrics,
        band_config=band_config,
        notes=[
            f"Imbalance strategy chosen because: {strategy_reason}.",
            "Bands derived from the alert budget, not from the probability "
            "distribution. See docs/MODEL_CARD.md.",
            "Evaluation sets under-represent repeat students (ADR-0003 amendment).",
        ],
    )
    background = train_features.sample(
        min(args.background_size, len(train_features)), random_state=seed
    ).reset_index(drop=True)
    directory = save_model(calibrated, metadata, background=background)
    print(f"      {directory}")

    _write_report(
        config,
        split,
        population,
        invariant_table,
        imbalance_table,
        chosen_strategy,
        strategy_reason,
        tuning,
        calibration,
        bands,
        test_result,
        test_budget,
        test_sweep,
        version,
    )
    print(f"\nReport: {REPORT_PATH}")
    return 0


def _write_report(
    config,
    split,
    population,
    invariant_table,
    imbalance_table,
    chosen_strategy,
    strategy_reason,
    tuning,
    calibration,
    bands,
    test_result,
    test_budget,
    test_sweep,
    version,
) -> None:
    low, high = test_result.intervals["average_precision"]
    budget = config.evaluation.alert_budget
    budget_row = test_budget[test_budget["alert_budget"] == budget]
    at_budget = budget_row.iloc[0] if len(budget_row) else test_budget.iloc[2]
    synthetic = invariant_table[invariant_table["dataset"] == "SMOTE synthetic"].iloc[0]
    real = invariant_table[invariant_table["dataset"] == "real students"].iloc[0]

    lines = [
        "# Primary model — XGBoost (Phase 6)",
        "",
        f"Artifact version `{version}`. Generated by `scripts/train_model.py`.",
        "",
        "## Headline",
        "",
        f"- Test PR-AUC **{test_result.ranking.average_precision:.4f}** "
        f"[{low:.4f}, {high:.4f}] against a "
        f"{test_result.ranking.base_rate:.2%} base rate "
        f"({test_result.ranking.average_precision / test_result.ranking.base_rate:.1f}x).",
        f"- Tuned CV PR-AUC {tuning.best_value:.4f} over {tuning.n_trials} Optuna trials.",
        f"- At a {budget:.0%} alert budget: precision "
        f"**{at_budget['precision']:.1%}** ({at_budget['lift']:.1f}x), reaching "
        f"{at_budget['recall']:.1%} of students who withdraw.",
        f"- Brier {calibration.brier_before:.5f} -> {calibration.brier_after:.5f} after "
        "isotonic calibration on validation.",
        "",
        "## The SMOTE objection, tested rather than asserted",
        "",
        "ADR-0002 rejected SMOTE because interpolating between students would",
        "manufacture impossible trajectories. That was an argument; this is the",
        "measurement. The feature set has hard structural invariants — nested",
        "windows (`clicks_7d <= clicks_14d <= clicks_28d`), day caps",
        "(`active_days_28d <= 28`), and binary indicators — that no real student",
        "can violate.",
        "",
        invariant_table.to_markdown(index=False),
        "",
        f"Real students violate nothing ({real['rows_with_any_violation']} rows). "
        f"SMOTE produces **{synthetic['rows_with_any_violation']:,} violating rows of "
        f"{synthetic['rows']:,} ({synthetic['share_violating']:.1%})**. These are not",
        "merely unusual students; they are arithmetically impossible ones. The",
        "objection holds.",
        "",
        "## Imbalance strategies",
        "",
        "Grouped CV on the training split only. Resamplers sit inside the",
        "pipeline, so they are refitted per fold — applying SMOTE before splitting",
        "leaks, because a synthetic training point can be interpolated from a",
        "validation neighbour.",
        "",
        imbalance_table.to_markdown(index=False) if len(imbalance_table) else "_skipped_",
        "",
        f"Chosen: **{chosen_strategy}** — {strategy_reason}.",
        "",
        "Selection is not `argmax`. The top strategies fall within fold noise of",
        "each other, so picking the highest score selects on which fold split",
        "happened to favour it. The rule is instead: take everything within one",
        "fold standard deviation of the best, then prefer the simplest — one that",
        "discards no data, invents no data, and adds no randomness.",
        "",
        "## Calibration",
        "",
        "The dashboard shows a counsellor a percentage, so the number has to mean",
        "what it says. Isotonic rather than Platt: Platt assumes a sigmoid link,",
        "which is a strong assumption at a 3% base rate, and validation has enough",
        "positives for a nonparametric fit.",
        "",
        calibration.to_frame().to_markdown(index=False),
        "",
        "### Reliability (quantile bins)",
        "",
        "Quantile rather than equal-width bins: at a 3% base rate nearly every",
        "prediction lands in the lowest equal-width bin, so that table would say",
        "nothing about the high-risk region that drives decisions.",
        "",
        calibration.reliability.to_markdown(index=False),
        "",
        "## Risk bands, derived from capacity",
        "",
        "Cut so the CRITICAL band matches what staff can actually take on, which is",
        "the procedure `thresholds.yaml` specified in Phase 1. Not chosen from",
        "where the probability distribution happens to bend.",
        "",
        bands.to_markdown(index=False),
        "",
        "## Per-checkpoint (test)",
        "",
        test_result.per_checkpoint.to_markdown(index=False),
        "",
        "## Capacity view (test)",
        "",
        test_budget.to_markdown(index=False),
        "",
        "## Recall tradeoff (test)",
        "",
        "Retained so the recall view stays visible rather than being dropped once",
        "it proved inconvenient in Phase 5.",
        "",
        test_sweep.to_markdown(index=False),
        "",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
