"""Tests for SHAP attribution on the real artifact.

Marked ``ml``, and skipped when no model has been trained, because these need a
registered artifact. The additivity test is the one that matters: if SHAP values
do not reconstruct the model output, every explanation shown to a counsellor is
wrong in a way nothing else would reveal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dropout_ews.config.settings import MODELS_DIR
from dropout_ews.explainability.narratives import render_explanation
from dropout_ews.explainability.shap_explainer import (
    NEGLIGIBLE_CONTRIBUTION,
    Explanation,
    FeatureContribution,
    RiskExplainer,
    attribution_stability,
    transform_to_model_matrix,
    unwrap_pipeline,
)

pytestmark = [
    pytest.mark.ml,
    pytest.mark.skipif(
        not (MODELS_DIR / "LATEST").is_file(),
        reason="no registered model; run scripts/train_model.py",
    ),
]

DROP_COLUMNS = ("label", "date_unregistration", "split", "row_id")


@pytest.fixture(scope="module")
def artifacts():
    """The registered model, an explainer, and a slice of test rows."""
    from dropout_ews.data.checkpoints import build_checkpoint_rows
    from dropout_ews.evaluation.splits import make_temporal_split
    from dropout_ews.models.registry import load_background, load_model

    frame = pd.read_parquet("data/processed/features.parquet")
    population, _ = build_checkpoint_rows()
    frame.index = population.index
    frame["code_presentation"] = population["code_presentation"].to_numpy()
    split = make_temporal_split(population)

    model, metadata = load_model()
    explainer = RiskExplainer(model, background=load_background())
    test = frame.loc[split.test]
    features = test.drop(columns=[c for c in DROP_COLUMNS if c in test.columns])
    return model, metadata, explainer, features


# ---------------------------------------------------------------------------
# Additivity — the correctness guarantee
# ---------------------------------------------------------------------------


def test_shap_values_reconstruct_the_model_margin(artifacts) -> None:
    """base value + sum(contributions) must equal the booster's raw output.

    This is the property that makes an attribution an explanation rather than a
    plausible-looking number. Verified against the margin, not the calibrated
    probability, because attributions are in pre-calibration log-odds.
    """
    _, _, explainer, features = artifacts
    sample = features.head(200)
    values, base = explainer.shap_values(sample)
    matrix = transform_to_model_matrix(explainer.pipeline, sample)
    margin = explainer.classifier.predict(matrix, output_margin=True)
    np.testing.assert_allclose(base + values.sum(axis=1), margin, atol=1e-4)


def test_explanation_raw_score_matches_the_margin(artifacts) -> None:
    _, _, explainer, features = artifacts
    sample = features.head(20)
    matrix = transform_to_model_matrix(explainer.pipeline, sample)
    margin = explainer.classifier.predict(matrix, output_margin=True)
    for position, explanation in enumerate(explainer.explain(sample)):
        assert explanation.raw_score == pytest.approx(float(margin[position]), abs=1e-4)


def test_base_value_is_constant_across_rows(artifacts) -> None:
    """XGBoost returns the bias as the final column; a non-constant value would
    mean the output layout changed and the split into values/bias is wrong."""
    _, _, explainer, features = artifacts
    _, base_a = explainer.shap_values(features.head(50))
    _, base_b = explainer.shap_values(features.tail(50))
    assert base_a == pytest.approx(base_b)


def test_attribution_count_matches_the_model_matrix_width(artifacts) -> None:
    """37 allowlisted features plus the imputer's missingness indicators. A
    mismatch would misalign every attribution to the wrong feature name."""
    _, metadata, explainer, features = artifacts
    explanation = explainer.explain_one(features.head(1))
    assert len(explanation.contributions) == len(explainer.feature_names)
    assert len(explanation.contributions) >= metadata.n_features


def test_feature_names_are_unique_and_ordered(artifacts) -> None:
    _, _, explainer, _ = artifacts
    names = explainer.feature_names
    assert len(names) == len(set(names))


def test_explain_one_rejects_a_multi_row_frame(artifacts) -> None:
    _, _, explainer, features = artifacts
    with pytest.raises(ValueError, match="exactly one row"):
        explainer.explain_one(features.head(3))


def test_unwrap_pipeline_reaches_through_the_calibrator(artifacts) -> None:
    """The registered artifact is a calibrated wrapper; TreeSHAP needs the
    pipeline inside it."""
    model, _, _, _ = artifacts
    pipeline = unwrap_pipeline(model)
    assert "classifier" in pipeline.named_steps
    assert "cohort" in pipeline.named_steps


def test_unwrap_pipeline_rejects_an_unexpected_type() -> None:
    with pytest.raises(TypeError, match="cannot unwrap"):
        unwrap_pipeline("not a model")


# ---------------------------------------------------------------------------
# Attribution stability — the reason this was checked before wiring to a UI
# ---------------------------------------------------------------------------


def test_top_factors_are_stable_among_near_identical_students(artifacts) -> None:
    """At a 2.7% positive rate with 37 correlated features, attributions could be
    small and unstable. If two near-identical students received different "main
    reasons", showing those reasons as *the* explanation would mislead staff even
    with a well-calibrated probability.

    Measured on high-risk rows, since those are the ones staff actually see. The
    bound is deliberately modest: the claim being defended is "usually the same
    factors", not "always".
    """
    model, _, explainer, features = artifacts
    probabilities = model.predict_proba(features)[:, 1]
    high_risk = features[probabilities >= np.quantile(probabilities, 0.95)].head(300)

    stability = attribution_stability(explainer, high_risk, k=3, n_neighbours=5)
    mean_overlap = stability["mean_top_k_overlap"].mean()
    assert mean_overlap > 0.6, (
        f"top-3 factors agree with near neighbours only {mean_overlap:.0%} of the time; "
        "presenting them as the explanation would be misleading"
    )


def test_stability_is_at_least_as_good_for_high_risk_students(artifacts) -> None:
    """The population staff act on should not be the least explainable one."""
    model, _, explainer, features = artifacts
    probabilities = model.predict_proba(features)[:, 1]
    high_risk = features[probabilities >= np.quantile(probabilities, 0.95)].head(250)
    everyone = features.head(250)

    high_mean = attribution_stability(explainer, high_risk, k=3)["mean_top_k_overlap"].mean()
    all_mean = attribution_stability(explainer, everyone, k=3)["mean_top_k_overlap"].mean()
    assert high_mean >= all_mean - 0.05


# ---------------------------------------------------------------------------
# Global importance
# ---------------------------------------------------------------------------


def test_global_importance_is_ordered_and_covers_every_feature(artifacts) -> None:
    _, _, explainer, features = artifacts
    importance = explainer.global_importance(features.head(1000))
    assert list(importance["feature"]) != sorted(importance["feature"])  # ordered by value
    assert importance["mean_abs_shap"].is_monotonic_decreasing
    assert set(importance["feature"]) == set(explainer.feature_names)
    assert (importance["mean_abs_shap"] >= 0).all()


def test_global_importance_uses_the_background_when_no_frame_is_given(artifacts) -> None:
    """The background travels with the artifact so a published global summary
    stays comparable across runs."""
    _, _, explainer, _ = artifacts
    importance = explainer.global_importance()
    assert len(importance) == len(explainer.feature_names)


def test_global_importance_without_a_frame_or_background_fails_loudly(artifacts) -> None:
    model, _, _, _ = artifacts
    bare = RiskExplainer(model, background=None)
    with pytest.raises(ValueError, match="no frame supplied"):
        bare.global_importance()


# ---------------------------------------------------------------------------
# Rendering the real explanations
# ---------------------------------------------------------------------------


def test_real_explanations_render_without_unmapped_features(artifacts) -> None:
    """Every feature the model uses must have narrative metadata, including the
    imputer's indicator columns. An unmapped one would surface to staff as a raw
    column name."""
    _, _, explainer, features = artifacts
    for explanation in explainer.explain(features.head(40)):
        rendered = render_explanation(explanation)
        for factor in (
            rendered.risk_factors + rendered.protective_factors + rendered.context_factors
        ):
            assert factor.label
            assert "_" not in factor.label or factor.label.startswith("No recorded data")


def test_rendered_explanations_always_carry_the_disclaimer(artifacts) -> None:
    _, _, explainer, features = artifacts
    for explanation in explainer.explain(features.head(20)):
        assert render_explanation(explanation).disclaimer


def test_high_risk_students_get_actionable_factors(artifacts) -> None:
    """A counsellor looking at a flagged student needs something they can act on,
    not only context such as how far through the course the student is."""
    model, _, explainer, features = artifacts
    probabilities = model.predict_proba(features)[:, 1]
    high_risk = features[probabilities >= np.quantile(probabilities, 0.98)].head(50)

    with_actionable = sum(
        1
        for explanation in explainer.explain(high_risk)
        if render_explanation(explanation).risk_factors
    )
    assert with_actionable / len(high_risk) > 0.8, (
        "most flagged students should have at least one actionable risk factor"
    )


def test_matched_factors_feed_the_intervention_engine(artifacts) -> None:
    from dropout_ews.interventions.engine import recommend

    model, _, explainer, features = artifacts
    probabilities = model.predict_proba(features)[:, 1]
    high_risk = features[probabilities >= np.quantile(probabilities, 0.98)].head(30)

    for explanation in explainer.explain(high_risk):
        rendered = render_explanation(explanation)
        recommendations, _ = recommend(rendered.matched_factors, "critical")
        assert recommendations, "a critical-band student must receive an offer of support"


# ---------------------------------------------------------------------------
# Negligible contributions
# ---------------------------------------------------------------------------


def test_negligible_contributions_are_excluded_from_top_factors() -> None:
    """TreeSHAP returns tiny non-zero values for features that played no real
    part; presenting one as a reason would be noise dressed as explanation."""
    explanation = Explanation(
        contributions=[
            FeatureContribution("clicks_7d", 10.0, 0.5),
            FeatureContribution("mean_score", 60.0, NEGLIGIBLE_CONTRIBUTION / 10),
        ],
        base_value=-3.0,
        raw_score=-2.5,
    )
    top = explanation.top(k=5)
    assert [c.feature for c in top] == ["clicks_7d"]
    assert explanation.top(k=5, drop_negligible=False)[1].feature == "mean_score"
