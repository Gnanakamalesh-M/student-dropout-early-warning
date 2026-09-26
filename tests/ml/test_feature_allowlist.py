"""Bind the allowlist in ``features.yaml`` to what the code actually produces.

Without these tests the config and the code drift apart silently: a typo in the
allowlist becomes a missing feature, and a renamed builder output becomes a
KeyError deep in training — or worse, a quietly smaller feature set that still
trains.

They also enforce ADR-0003 control #2: nothing outcome-adjacent and no
protected attribute may appear in the matrix.
"""

from __future__ import annotations

import fnmatch

import numpy as np
import pandas as pd
import pytest

from dropout_ews.config.settings import load_feature_config
from dropout_ews.features.builder import FeatureBuilder
from dropout_ews.preprocessing.cohort import CohortZScorer

pytestmark = pytest.mark.ml

MODULE, PRESENTATION = "AAA", "2013J"
COHORT_SIZE = 40


def _fixture_frames() -> tuple[pd.DataFrame, ...]:
    """A cohort large enough for the z-scorer to fit real statistics."""
    rng = np.random.default_rng(0)
    students = list(range(1, COHORT_SIZE + 1))

    population = pd.DataFrame(
        {
            "code_module": MODULE,
            "code_presentation": PRESENTATION,
            "id_student": students,
            "checkpoint_day": 90,
            "module_presentation_length": 269,
        }
    )
    engagement = pd.DataFrame(
        [
            {
                "code_module": MODULE,
                "code_presentation": PRESENTATION,
                "id_student": student,
                "course_day": day,
                "clicks": int(rng.integers(1, 60)),
                "distinct_resources": int(rng.integers(1, 5)),
            }
            for student in students
            for day in range(5, 90, 7)
        ]
    )
    assessments = pd.DataFrame(
        {
            "code_module": MODULE,
            "code_presentation": PRESENTATION,
            "id_assessment": [1, 2, 3],
            "assessment_type": "TMA",
            "date": [20.0, 50.0, 80.0],
            "weight": [30.0, 30.0, 40.0],
        }
    )
    submissions = pd.DataFrame(
        [
            {
                "id_assessment": assessment,
                "id_student": student,
                "date_submitted": due + int(rng.integers(-3, 4)),
                "is_banked": 0,
                "score": float(rng.integers(20, 95)),
            }
            for student in students
            for assessment, due in ((1, 20), (2, 50), (3, 80))
        ]
    )
    static = pd.DataFrame(
        {
            "code_module": MODULE,
            "code_presentation": PRESENTATION,
            "id_student": students,
            "num_of_prev_attempts": rng.integers(0, 3, len(students)),
            "studied_credits": rng.choice([30, 60, 120], len(students)),
            "date_registration": -rng.integers(1, 200, len(students)).astype(float),
        }
    )
    return population, engagement, assessments, submissions, static


@pytest.fixture(scope="module")
def produced() -> pd.DataFrame:
    """Every column the full feature stack produces, builder plus transformer."""
    population, engagement, assessments, submissions, static = _fixture_frames()
    result = FeatureBuilder().build(
        population,
        engagement=engagement,
        assessments=assessments,
        student_assessment=submissions,
        static=static,
    )
    frame = result.frame.copy()
    # The z-scorer needs the cohort keys, which travel on the population.
    frame["code_presentation"] = PRESENTATION
    return CohortZScorer(min_cohort_size=10).fit(frame).transform(frame)


def test_every_allowlisted_feature_is_actually_produced(produced) -> None:
    """A typo in features.yaml becomes a silently missing feature otherwise."""
    allowlisted = load_feature_config().features.all_features()
    missing = [name for name in allowlisted if name not in produced.columns]
    assert not missing, f"allowlisted but never produced: {missing}"


def test_allowlist_has_no_duplicates() -> None:
    allowlisted = load_feature_config().features.all_features()
    duplicates = {name for name in allowlisted if allowlisted.count(name) > 1}
    assert not duplicates, f"duplicated in the allowlist: {sorted(duplicates)}"


def test_allowlist_size_matches_the_recorded_decision() -> None:
    """37 features: the Phase 4 ablation set of 36, plus ``checkpoint_day``.

    Pinning the count makes an accidental addition or removal visible in review.
    It also caught a real bug: ``FeatureGroups`` originally lacked the
    ``assessment`` and ``static`` fields, so Pydantic silently discarded 15
    features and the allowlist reported 22 instead of 37.
    """
    assert len(load_feature_config().features.all_features()) == 37


def test_group_sizes_match_the_ablation_record() -> None:
    """Per-group counts, so a feature moving between groups is also visible."""
    groups = load_feature_config().features
    assert len(groups.level) == 11
    assert len(groups.recency) == 3
    assert len(groups.trend) == 3
    assert len(groups.assessment) == 11
    assert len(groups.relative) == 5
    assert len(groups.static) == 4
    assert groups.volatility == []
    assert groups.composite == []


def test_unknown_feature_group_is_rejected() -> None:
    """A group in features.yaml that the schema does not declare must fail, not
    vanish. This is the guard for the bug described above."""
    from pydantic import ValidationError

    from dropout_ews.config.settings import FeatureGroups

    with pytest.raises(ValidationError):
        FeatureGroups.model_validate({"level": ["a"], "mystery_group": ["b"]})


def test_features_excluded_by_the_ablation_stay_excluded(produced) -> None:
    """These are still computed by the builder, but must not be modelled.

    Each was tested and failed to earn its place; see features.yaml for the
    per-feature evidence.
    """
    config = load_feature_config()
    allowlisted = set(config.features.all_features())
    assert config.excluded_by_ablation, "the exclusion list should not be empty"
    for excluded in config.excluded_by_ablation:
        assert excluded in produced.columns, f"builder should still compute {excluded}"
        assert excluded not in allowlisted, f"{excluded} was excluded by ablation"


def test_every_builder_output_is_accounted_for(produced) -> None:
    """The completeness check.

    Every column the builder produces must be either allowlisted or explicitly
    excluded. Without this, dropping a feature from the allowlist by accident is
    invisible — which is exactly how ``assessments_banked`` went untested until
    a count mismatch exposed it.
    """
    from dropout_ews.features.builder import KEY_COLUMNS

    config = load_feature_config()
    accounted = (
        set(config.features.all_features())
        | set(config.excluded_by_ablation)
        | {*KEY_COLUMNS, "row_id", "code_presentation"}
    )
    unaccounted = [
        column
        for column in produced.columns
        if column not in accounted and not column.endswith("_cohort_z")
    ]
    assert not unaccounted, (
        f"builder outputs neither allowlisted nor excluded: {sorted(unaccounted)}. "
        "Add each to features.yaml under `features:` or `excluded_by_ablation:`."
    )


def test_volatility_group_is_empty_on_purpose() -> None:
    """Recorded as an empty group rather than deleted, so the exclusion stays
    visible instead of being silently forgotten."""
    groups = load_feature_config().features
    assert groups.volatility == []


def test_no_forbidden_column_is_allowlisted() -> None:
    """ADR-0003 control #2, including protected attributes (ETHICS.md)."""
    config = load_feature_config()
    allowlisted = set(config.features.all_features())

    for column in config.forbidden.exact:
        assert column not in allowlisted
    for pattern in config.forbidden.patterns:
        matched = {name for name in allowlisted if fnmatch.fnmatch(name, pattern)}
        assert not matched, f"{pattern} matched allowlisted features {sorted(matched)}"


def test_protected_attributes_are_in_the_forbidden_list() -> None:
    """Data minimisation is structural, not aspirational: the config must name
    these so the allowlist validator can reject them."""
    forbidden = set(load_feature_config().forbidden.exact)
    for attribute in ("gender", "age_band", "imd_band", "disability", "region"):
        assert attribute in forbidden


def test_no_protected_attribute_is_produced_by_the_feature_stack(produced) -> None:
    """The stronger claim: these never even reach the frame the model sees."""
    for attribute in ("gender", "age_band", "imd_band", "disability", "region"):
        assert attribute not in produced.columns


def test_outcome_columns_are_never_produced(produced) -> None:
    for column in ("date_unregistration", "final_result", "label"):
        assert column not in produced.columns


def test_allowlisted_features_are_numeric(produced) -> None:
    """Non-numeric columns would fail at fit time with a confusing error."""
    allowlisted = load_feature_config().features.all_features()
    non_numeric = [
        name for name in allowlisted if not pd.api.types.is_numeric_dtype(produced[name])
    ]
    assert not non_numeric, f"non-numeric allowlisted features: {non_numeric}"


def test_allowlisted_features_contain_no_infinities(produced) -> None:
    """Ratio features use +1 smoothing; an infinity breaks tree splits and every
    downstream metric."""
    allowlisted = load_feature_config().features.all_features()
    values = produced[allowlisted].to_numpy(dtype="float64")
    assert not np.isinf(values).any()


def test_cohort_z_features_come_from_the_transformer_not_the_builder() -> None:
    """Fitted state must not live in the stateless builder (ADR-0003 #4)."""
    population, engagement, assessments, submissions, static = _fixture_frames()
    built = FeatureBuilder().build(
        population,
        engagement=engagement,
        assessments=assessments,
        student_assessment=submissions,
        static=static,
    )
    relative = load_feature_config().features.relative
    assert relative, "the relative group should not be empty"
    for name in relative:
        assert name not in built.feature_names
