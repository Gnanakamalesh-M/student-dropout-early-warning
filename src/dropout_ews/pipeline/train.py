"""Baseline training and evaluation run.

Order of operations, and why:

1. Fit each pipeline on the **training** presentations only.
2. Select the operating-point threshold on **validation**, never on test. The
   threshold is a fitted quantity like any other; choosing it on test would
   turn the test set into a tuning set.
3. Report on **test** (the chronologically last presentation) with cluster
   bootstrap intervals over students.
4. Score the Phase 3 trivial rules on the same test rows, so the comparison is
   like-for-like rather than against numbers remembered from another split.

Cross-validated training scores are also reported. They are the honest place to
compare models, because the test set is a single cohort and its intervals are
wide.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline

from dropout_ews.config.settings import load_feature_config
from dropout_ews.evaluation.metrics import (
    EvaluationResult,
    budget_sweep,
    evaluate,
    ranking_metrics,
    threshold_for_recall,
    threshold_sweep,
    trivial_rule_scores,
)
from dropout_ews.evaluation.splits import (
    GROUP_COLUMN,
    TemporalSplit,
    assert_no_group_leakage_in_folds,
    grouped_cv_splits,
)

PipelineFactory = Callable[[], Pipeline]


@dataclass
class TrainedBaseline:
    name: str
    pipeline: Pipeline
    cv_average_precision_mean: float
    cv_average_precision_std: float
    validation_threshold: float
    validation: EvaluationResult
    test: EvaluationResult
    test_sweep: pd.DataFrame
    test_budget_sweep: pd.DataFrame


def _feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """The pipeline input: cohort keys plus builder features.

    ``label`` and ``date_unregistration`` are dropped here. They travel with the
    processed parquet because the trainer needs the label, but the pipeline must
    never see them (ADR-0003 control #2).
    """
    return frame.drop(
        columns=[
            column
            for column in ("label", "date_unregistration", "split", "row_id")
            if column in frame.columns
        ]
    )


def cross_validate_average_precision(
    pipeline_factory: PipelineFactory,
    frame: pd.DataFrame,
    population: pd.DataFrame,
    train_index: pd.Index,
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[float, float]:
    """Grouped CV average precision on the training split."""
    folds = grouped_cv_splits(population, train_index, n_splits=n_splits, random_state=random_state)
    assert_no_group_leakage_in_folds(population, train_index, folds)

    train_frame = frame.loc[train_index].reset_index(drop=True)
    features = _feature_frame(train_frame)
    y = train_frame["label"].to_numpy()

    scores = []
    for train_positions, validation_positions in folds:
        pipeline = pipeline_factory()
        pipeline.fit(features.iloc[train_positions], y[train_positions])
        probabilities = pipeline.predict_proba(features.iloc[validation_positions])[:, 1]
        scores.append(ranking_metrics(y[validation_positions], probabilities).average_precision)
    return float(np.mean(scores)), float(np.std(scores))


def train_baseline(
    name: str,
    pipeline_factory: PipelineFactory,
    frame: pd.DataFrame,
    population: pd.DataFrame,
    split: TemporalSplit,
    n_splits: int = 5,
    bootstrap_iterations: int = 1000,
) -> TrainedBaseline:
    """Fit, choose a threshold on validation, evaluate on test."""
    config = load_feature_config()
    target_recall = config.evaluation.target_recall
    seed = config.evaluation.random_seed

    cv_mean, cv_std = cross_validate_average_precision(
        pipeline_factory, frame, population, split.train, n_splits, seed
    )

    train_frame = frame.loc[split.train]
    pipeline = pipeline_factory()
    pipeline.fit(_feature_frame(train_frame), train_frame["label"].to_numpy())

    def _score(index: pd.Index) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        subset = frame.loc[index]
        probabilities = pipeline.predict_proba(_feature_frame(subset))[:, 1]
        return subset, subset["label"].to_numpy(), probabilities

    validation_frame, y_validation, p_validation = _score(split.validation)
    # THE THRESHOLD IS CHOSEN HERE, on validation, and never revisited on test.
    threshold = threshold_for_recall(y_validation, p_validation, target_recall)

    validation_result = evaluate(
        f"{name} (validation)",
        y_validation,
        p_validation,
        validation_frame["checkpoint_day"].to_numpy(),
        validation_frame[GROUP_COLUMN].to_numpy(),
        threshold,
        bootstrap_iterations=bootstrap_iterations,
        random_state=seed,
    )

    test_frame, y_test, p_test = _score(split.test)
    test_result = evaluate(
        f"{name} (test)",
        y_test,
        p_test,
        test_frame["checkpoint_day"].to_numpy(),
        test_frame[GROUP_COLUMN].to_numpy(),
        threshold,
        bootstrap_iterations=bootstrap_iterations,
        random_state=seed,
    )

    return TrainedBaseline(
        name=name,
        pipeline=pipeline,
        cv_average_precision_mean=cv_mean,
        cv_average_precision_std=cv_std,
        validation_threshold=threshold,
        validation=validation_result,
        test=test_result,
        test_sweep=threshold_sweep(y_test, p_test),
        test_budget_sweep=budget_sweep(y_test, p_test),
    )


def evaluate_trivial_rules(
    frame: pd.DataFrame, split: TemporalSplit, bootstrap_iterations: int = 400
) -> list[EvaluationResult]:
    """Score the Phase 3 rules on the same test rows as the models."""
    test_frame = frame.loc[split.test]
    y = test_frame["label"].to_numpy()
    results = []
    for name, scores in trivial_rule_scores(test_frame).items():
        # A rule has no probability, so the "threshold" is its own positive
        # level. For the binary rules that means flagging exactly the rule; for
        # the continuous ones it is the recall-matched cut.
        threshold = threshold_for_recall(y, scores, 0.8)
        results.append(
            evaluate(
                name,
                y,
                scores,
                test_frame["checkpoint_day"].to_numpy(),
                test_frame[GROUP_COLUMN].to_numpy(),
                threshold,
                bootstrap_iterations=bootstrap_iterations,
            )
        )
    return results
