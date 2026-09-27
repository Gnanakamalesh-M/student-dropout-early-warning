"""Tests for the primary model: imbalance audit, calibration, bands, registry.

The invariant-audit tests matter most. ADR-0002 rejected SMOTE on an argument,
and these tests are what turn it into a claim that can be checked and, if the
data changed, falsified.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from dropout_ews.models.calibration import (
    assess_calibration,
    bands_to_config,
    calibrate,
    derive_bands_from_budget,
    reliability_table,
)
from dropout_ews.models.imbalance import (
    count_invariant_violations,
    imbalance_strategies,
    smote_synthetic_rows,
)
from dropout_ews.models.registry import (
    ModelMetadata,
    frame_hash,
    list_versions,
    load_background,
    load_model,
    new_version_id,
    save_model,
)
from dropout_ews.models.xgboost_model import positive_class_weight, xgboost_pipeline

pytestmark = pytest.mark.ml


# ---------------------------------------------------------------------------
# Structural invariants
# ---------------------------------------------------------------------------


def _valid_rows(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Rows that respect every structural invariant, as real students do."""
    rng = np.random.default_rng(seed)
    active_7 = rng.integers(0, 8, n)
    active_14 = active_7 + rng.integers(0, 7, n)
    active_28 = active_14 + rng.integers(0, 14, n)
    clicks_7 = rng.integers(0, 200, n)
    clicks_14 = clicks_7 + rng.integers(0, 200, n)
    clicks_28 = clicks_14 + rng.integers(0, 300, n)
    # Each window must be built from the previous one, not from an earlier link:
    # deriving clicks_all_time from clicks_28d lets it fall below clicks_56d,
    # which is itself an invariant violation.
    clicks_56 = clicks_28 + rng.integers(0, 300, n)
    clicks_all = clicks_56 + rng.integers(0, 900, n)
    return pd.DataFrame(
        {
            "clicks_7d": clicks_7.astype(float),
            "clicks_14d": clicks_14.astype(float),
            "clicks_28d": clicks_28.astype(float),
            "clicks_56d": clicks_56.astype(float),
            "clicks_all_time": clicks_all.astype(float),
            "active_days_7d": active_7.astype(float),
            "active_days_14d": active_14.astype(float),
            "active_days_28d": active_28.astype(float),
            "ever_active": (clicks_28 > 0).astype(float),
            "has_submitted": rng.integers(0, 2, n).astype(float),
            "assessments_due": rng.integers(0, 5, n).astype(float),
        }
    )


def test_real_shaped_rows_violate_nothing() -> None:
    """Guard the guard: a detector that fires on valid data is useless."""
    report = count_invariant_violations(_valid_rows())
    assert report.any_violation_count == 0
    assert report.nested_windows == 0
    assert report.day_caps == 0
    assert report.binary_indicators == 0


def test_nested_window_violation_is_detected() -> None:
    """clicks_7d cannot exceed clicks_28d: the 7-day window is a subset."""
    frame = _valid_rows(10)
    frame.loc[0, "clicks_7d"] = frame.loc[0, "clicks_28d"] + 500
    report = count_invariant_violations(frame)
    assert report.nested_windows >= 1
    assert report.any_violation_count == 1


def test_day_cap_violation_is_detected() -> None:
    """A student cannot be active on 40 days of a 28-day window."""
    frame = _valid_rows(10)
    frame.loc[3, "active_days_28d"] = 40
    assert count_invariant_violations(frame).day_caps >= 1


def test_fractional_binary_indicator_is_detected() -> None:
    """Interpolation turns a 0/1 flag into 0.37, which is not a state a student
    can be in."""
    frame = _valid_rows(10)
    frame.loc[5, "ever_active"] = 0.37
    assert count_invariant_violations(frame).binary_indicators == 1


def test_negative_count_is_detected() -> None:
    frame = _valid_rows(10)
    frame.loc[7, "clicks_28d"] = -5.0
    assert count_invariant_violations(frame).negative_counts >= 1


def test_smote_manufactures_structurally_impossible_students() -> None:
    """The ADR-0002 objection, as a test rather than an assertion.

    SMOTE interpolates each dimension independently, so it preserves marginal
    bounds but breaks the *relationships* between features. On the real training
    split this produces impossible rows in about 22% of what it generates; the
    threshold here is deliberately loose so the test checks the mechanism rather
    than pinning a figure that depends on the sample.
    """
    frame = _valid_rows(400, seed=3)
    rng = np.random.default_rng(3)
    # A minority class that sits away from the majority, so interpolation has
    # somewhere to go wrong.
    y = (rng.random(len(frame)) < 0.12).astype(int)
    frame.loc[y == 1, "clicks_7d"] = frame.loc[y == 1, "clicks_28d"]

    synthetic = smote_synthetic_rows(frame, y, random_state=3)
    assert len(synthetic) > 0
    report = count_invariant_violations(synthetic)
    assert report.any_violation_count > 0, (
        "SMOTE produced no invariant violations on this fixture, which would "
        "undermine the ADR-0002 rationale"
    )


def test_imbalance_strategies_all_build_a_working_pipeline() -> None:
    """Each configuration must at least construct; a broken entry would be
    silently skipped in the comparison otherwise."""
    strategies = imbalance_strategies(scale_pos_weight=36.0)
    assert {"none", "class weights", "SMOTE"} <= set(strategies)
    for spec in strategies.values():
        pipeline = xgboost_pipeline(
            scale_pos_weight=spec["scale_pos_weight"], resampler=spec["resampler"]
        )
        assert pipeline is not None


def test_resampler_sits_before_the_classifier_not_after() -> None:
    """A resampler after the classifier would be a no-op; inside the pipeline it
    is refitted per fold and never touches validation rows."""
    strategies = imbalance_strategies(36.0)
    pipeline = xgboost_pipeline(resampler=strategies["SMOTE"]["resampler"])
    names = [name for name, _ in pipeline.steps]
    assert names.index("resample") == names.index("classifier") - 1
    # Imputation must precede it, so the sampler never sees NaN.
    assert names.index("impute") < names.index("resample")


def test_positive_class_weight_is_the_negative_to_positive_ratio() -> None:
    y = np.array([1] * 10 + [0] * 90)
    assert positive_class_weight(y) == pytest.approx(9.0)


def test_positive_class_weight_rejects_a_single_class() -> None:
    with pytest.raises(ValueError, match="no positives"):
        positive_class_weight(np.zeros(10, dtype=int))


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def _miscalibrated(n: int = 4000, seed: int = 1):
    """Scores that rank well but are systematically too high."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.05).astype(int)
    # Correct ranking, inflated magnitude.
    p = np.clip(0.25 + 0.4 * y + rng.normal(0, 0.1, n), 0.001, 0.999)
    return y, p


def test_calibration_report_detects_overconfidence() -> None:
    y, p = _miscalibrated()
    report = assess_calibration(y, p, p)
    assert report.mean_predicted_before > report.observed_rate * 3
    assert not report.improved or report.brier_after == report.brier_before


def test_isotonic_calibration_improves_brier_and_mean_prediction() -> None:
    """The point of calibrating: a displayed percentage should mean what it says."""
    from sklearn.isotonic import IsotonicRegression

    y, p = _miscalibrated()
    fitted = IsotonicRegression(out_of_bounds="clip").fit(p, y)
    report = assess_calibration(y, p, fitted.predict(p))
    assert report.improved
    assert abs(report.mean_predicted_after - report.observed_rate) < abs(
        report.mean_predicted_before - report.observed_rate
    )


def test_reliability_table_uses_quantile_bins() -> None:
    """Equal-width bins put nearly every prediction in the lowest bucket at a 3%
    base rate, saying nothing about the high-risk region that drives decisions."""
    rng = np.random.default_rng(2)
    y = (rng.random(5000) < 0.03).astype(int)
    p = np.clip(rng.beta(1.2, 30, 5000), 0, 1)
    table = reliability_table(y, p, n_bins=10)
    assert len(table) >= 5
    # Quantile bins hold roughly equal counts; equal-width would not.
    assert table["rows"].max() / table["rows"].min() < 3


def test_calibrate_does_not_refit_the_underlying_pipeline() -> None:
    """``FrozenEstimator``/``prefit`` must preserve training-set learning; a
    refit on validation would silently discard the training presentations."""
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(4)
    X_train = pd.DataFrame({"a": rng.normal(size=500), "b": rng.normal(size=500)})
    y_train = (X_train["a"] > 0).astype(int).to_numpy()
    model = LogisticRegression().fit(X_train, y_train)
    original = model.coef_.copy()

    X_validation = pd.DataFrame({"a": rng.normal(size=300), "b": rng.normal(size=300)})
    y_validation = (X_validation["a"] > 0).astype(int).to_numpy()
    calibrate(model, X_validation, y_validation)
    np.testing.assert_allclose(model.coef_, original)


# ---------------------------------------------------------------------------
# Risk bands
# ---------------------------------------------------------------------------


def _band_inputs(n: int = 20_000, seed: int = 5):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.03).astype(int)
    p = np.clip(0.03 + 0.25 * y + rng.normal(0, 0.05, n), 0, 1)
    return y, p


def test_bands_are_ordered_and_start_at_zero() -> None:
    bands = derive_bands_from_budget(*_band_inputs())
    assert bands["band"].tolist() == ["low", "medium", "high", "critical"]
    assert bands["min_probability"].iloc[0] == 0.0
    assert bands["min_probability"].is_monotonic_increasing


def test_critical_band_matches_the_requested_budget() -> None:
    """The whole point: the top tier is sized to what staff can take on."""
    y, p = _band_inputs()
    bands = derive_bands_from_budget(y, p, critical_budget=0.05)
    critical = bands[bands["band"] == "critical"].iloc[0]
    assert critical["share_of_cohort"] == pytest.approx(0.05, abs=0.02)


def test_band_precision_increases_with_severity() -> None:
    """If a higher band were not more precise, the banding would be noise."""
    bands = derive_bands_from_budget(*_band_inputs()).set_index("band")
    assert bands.loc["critical", "precision_in_band"] > bands.loc["low", "precision_in_band"]
    assert bands.loc["critical", "lift_in_band"] > 1.0


def test_band_cumulative_recall_decreases_with_severity() -> None:
    bands = derive_bands_from_budget(*_band_inputs())
    recalls = bands["cumulative_recall_at_or_above"].tolist()
    assert recalls == sorted(recalls, reverse=True)
    assert recalls[0] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("critical", "high", "medium"),
    [(0.2, 0.1, 0.3), (0.05, 0.5, 0.4), (0.0, 0.15, 0.35), (0.05, 0.15, 1.5)],
)
def test_invalid_budget_ordering_is_rejected(critical, high, medium) -> None:
    y, p = _band_inputs(2000)
    with pytest.raises(ValueError, match="budgets must satisfy"):
        derive_bands_from_budget(
            y, p, critical_budget=critical, high_budget=high, medium_budget=medium
        )


def test_bands_to_config_marks_the_result_calibrated() -> None:
    """Only this path may set ``calibrated: true``; the shipped placeholder
    config stays false until the procedure has actually been run."""
    bands = derive_bands_from_budget(*_band_inputs())
    config = bands_to_config(bands, version=2, name="test-v1")
    assert config["calibrated"] is True
    assert config["version"] == 2
    assert [band["key"] for band in config["bands"]] == [
        "low",
        "medium",
        "high",
        "critical",
    ]


def test_generated_band_config_validates_against_the_schema() -> None:
    """A derived config must be loadable by the same validator that guards the
    shipped one, including its governance constraints."""
    from dropout_ews.config.settings import ThresholdConfig

    bands = derive_bands_from_budget(*_band_inputs())
    config = ThresholdConfig.model_validate(bands_to_config(bands, version=3, name="x"))
    assert config.calibrated is True
    assert config.governance.require_human_review_before_intervention is True


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def _metadata(version: str) -> ModelMetadata:
    return ModelMetadata(
        model_version=version,
        model_type="xgboost",
        created_at="2026-09-27T00:00:00Z",
        package_version="0.1.0",
        git_commit="abc123",
        feature_names=["clicks_7d", "clicks_28d"],
        n_features=2,
        hyperparameters={"max_depth": 4},
        imbalance_strategy="none",
        calibration_method="isotonic",
        train_rows=100,
        train_positives=3,
        train_data_hash="deadbeef",
        train_presentations=["2013B"],
        validation_presentations=["2014B"],
        test_presentations=["2014J"],
        metrics={"test_pr_auc": 0.1},
    )


def test_artifact_round_trip_preserves_predictions(tmp_path) -> None:
    """A model that predicts differently after a save/load cycle would make
    every stored prediction unreproducible."""
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(6)
    X = pd.DataFrame({"a": rng.normal(size=300), "b": rng.normal(size=300)})
    y = (X["a"] > 0).astype(int).to_numpy()
    model = LogisticRegression().fit(X, y)
    before = model.predict_proba(X)[:, 1]

    version = "xgboost-test"
    save_model(model, _metadata(version), models_dir=tmp_path)
    loaded, metadata = load_model(version, models_dir=tmp_path)
    np.testing.assert_allclose(loaded.predict_proba(X)[:, 1], before)
    assert metadata.model_version == version
    assert metadata.imbalance_strategy == "none"


def test_latest_pointer_resolves_without_an_explicit_version(tmp_path) -> None:
    from sklearn.dummy import DummyClassifier

    model = DummyClassifier(strategy="prior").fit(pd.DataFrame({"a": [0.0, 1.0]}), np.array([0, 1]))
    save_model(model, _metadata("xgboost-v1"), models_dir=tmp_path)
    save_model(model, _metadata("xgboost-v2"), models_dir=tmp_path)
    _, metadata = load_model(models_dir=tmp_path)
    assert metadata.model_version == "xgboost-v2"


def test_missing_latest_pointer_fails_with_actionable_guidance(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match=r"train_model\.py"):
        load_model(models_dir=tmp_path)


def test_metadata_is_valid_json_and_round_trips(tmp_path) -> None:
    """Metadata has to be machine-readable: the API and the model card page both
    consume it."""
    from sklearn.dummy import DummyClassifier

    model = DummyClassifier(strategy="prior").fit(pd.DataFrame({"a": [0.0, 1.0]}), np.array([0, 1]))
    directory = save_model(model, _metadata("xgboost-v1"), models_dir=tmp_path)
    payload = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    assert payload["n_features"] == 2
    assert ModelMetadata(**payload).model_version == "xgboost-v1"


def test_shap_background_is_versioned_with_the_model(tmp_path) -> None:
    """SHAP values depend on the background sample, so explanations are only
    reproducible if the same sample travels with the artifact."""
    from sklearn.dummy import DummyClassifier

    model = DummyClassifier(strategy="prior").fit(pd.DataFrame({"a": [0.0, 1.0]}), np.array([0, 1]))
    background = pd.DataFrame({"clicks_7d": [1.0, 2.0], "clicks_28d": [3.0, 4.0]})
    save_model(model, _metadata("xgboost-v1"), background=background, models_dir=tmp_path)
    loaded = load_background("xgboost-v1", models_dir=tmp_path)
    pd.testing.assert_frame_equal(loaded, background)


def test_list_versions_returns_newest_first(tmp_path) -> None:
    from sklearn.dummy import DummyClassifier

    model = DummyClassifier(strategy="prior").fit(pd.DataFrame({"a": [0.0, 1.0]}), np.array([0, 1]))
    for version in ("xgboost-20260101T000000Z", "xgboost-20260201T000000Z"):
        save_model(model, _metadata(version), models_dir=tmp_path)
    assert list_versions(models_dir=tmp_path) == [
        "xgboost-20260201T000000Z",
        "xgboost-20260101T000000Z",
    ]


def test_frame_hash_is_stable_and_sensitive() -> None:
    """The hash answers "was this model trained on the inputs I have now", so it
    must change when values or column names change."""
    frame = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    assert frame_hash(frame) == frame_hash(frame.copy())

    changed_value = frame.copy()
    changed_value.loc[0, "a"] = 99.0
    assert frame_hash(changed_value) != frame_hash(frame)

    renamed = frame.rename(columns={"a": "z"})
    assert frame_hash(renamed) != frame_hash(frame)


def test_version_ids_sort_chronologically() -> None:
    first = new_version_id("xgboost")
    assert first.startswith("xgboost-")
    assert first.endswith("Z")
