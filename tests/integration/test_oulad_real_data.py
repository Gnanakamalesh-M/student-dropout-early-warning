"""Tests against the real OULAD extraction.

Marked ``integration`` because they need ``data/raw/oulad/``, which is not
committed. Run after ``python scripts/download_data.py``:

    pytest tests/integration -m integration

These assert the *verified* figures recorded in docs/DATA_CARD.md. They are
regression tests on the data snapshot: if a mirror ever serves a different
version of OULAD, these fail rather than letting every downstream number shift
silently.
"""

from __future__ import annotations

import pytest

from dropout_ews.data.checkpoints import build_checkpoint_rows
from dropout_ews.data.loaders import (
    OULAD_DIR,
    load_courses,
    load_student_assessment,
    load_student_info,
    load_student_registration,
    presentation_sort_key,
    presentations_in_order,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (OULAD_DIR / "studentInfo.csv").is_file(),
        reason="OULAD not downloaded; run scripts/download_data.py",
    ),
]


# ---------------------------------------------------------------------------
# Snapshot shape
# ---------------------------------------------------------------------------


def test_row_counts_match_the_data_card() -> None:
    assert len(load_courses()) == 22
    assert len(load_student_info()) == 32_593
    assert len(load_student_registration()) == 32_593
    assert len(load_student_assessment()) == 173_912


def test_seven_modules_across_four_presentation_codes() -> None:
    """OULAD has 7 modules and 4 presentation codes, giving 22
    module-presentations — not 7 presentations. Phase 1 documentation had this
    wrong and it is corrected in the data card."""
    courses = load_courses()
    assert courses["code_module"].nunique() == 7
    assert courses["code_presentation"].nunique() == 4
    assert len(courses) == 22


def test_presentations_sort_chronologically() -> None:
    """The forward-in-time split depends on this ordering (ADR-0003)."""
    assert presentations_in_order() == ["2013B", "2013J", "2014B", "2014J"]
    assert presentation_sort_key("2013B") < presentation_sort_key("2013J")
    assert presentation_sort_key("2013J") < presentation_sort_key("2014B")


# ---------------------------------------------------------------------------
# The null-token quirk — the most dangerous one in this dataset
# ---------------------------------------------------------------------------


def test_missing_values_are_parsed_from_the_question_mark_token() -> None:
    """OULAD encodes missing as the literal string '?'. Read without
    na_values, date_unregistration parses as object and every null check
    silently reports 'not null' for all 32,593 rows, which produces a
    plausible-looking but completely wrong label set."""
    registration = load_student_registration()
    assert registration["date_unregistration"].dtype.kind == "f"
    n_missing = int(registration["date_unregistration"].isna().sum())
    assert n_missing == 22_521, "expected 22,521 students with no de-registration day"
    assert not (registration["date_unregistration"] == "?").any()


def test_imd_band_category_is_normalised() -> None:
    """The raw data has '10-20' without the percent sign that the other nine
    bands carry, which makes a ten-band variable look like eleven."""
    bands = set(load_student_info()["imd_band"].dropna().unique())
    assert "10-20" not in bands
    assert "10-20%" in bands
    assert len(bands) == 10


# ---------------------------------------------------------------------------
# Label consistency between the event time and the administrative outcome
# ---------------------------------------------------------------------------


def test_label_source_disagreements_match_recorded_counts() -> None:
    """Two documented inconsistencies exist between final_result and
    date_unregistration. Both are handled explicitly, not silently."""
    merged = load_student_registration().merge(
        load_student_info()[["code_module", "code_presentation", "id_student", "final_result"]],
        on=["code_module", "code_presentation", "id_student"],
        validate="one_to_one",
    )
    withdrawn = merged["final_result"] == "Withdrawn"
    has_event = merged["date_unregistration"].notna()

    # Withdrawn but no event time -> excluded as label-ambiguous.
    assert int((withdrawn & ~has_event).sum()) == 93
    # Event time but not marked Withdrawn -> event time is authoritative.
    assert int((~withdrawn & has_event).sum()) == 9


# ---------------------------------------------------------------------------
# The constructed population
# ---------------------------------------------------------------------------


def test_population_matches_the_verified_report() -> None:
    population, report = build_checkpoint_rows()

    assert len(population) == 150_938
    assert report.total_positives == 4_559
    assert report.positive_rate == pytest.approx(0.0302, abs=5e-5)
    assert report.withdrawn_without_event_time == 93
    assert report.withdrew_before_first_checkpoint == 5_127


def test_horizon_30_satisfies_the_precommitted_decision_rule() -> None:
    """ADR-0001 committed in advance to keeping H=30 only if the day-30
    positive rate is at least 1.5%. It is 4.04%, so H=30 stands. This test
    pins the number the decision rested on."""
    _, report = build_checkpoint_rows()
    day30_rate = report.positives_by_checkpoint[30] / report.rows_by_checkpoint[30]
    assert day30_rate == pytest.approx(0.0404, abs=5e-4)
    assert day30_rate >= 0.015


def test_horizon_30_causes_no_censoring() -> None:
    """The shortest presentation is 234 days and the last checkpoint plus the
    horizon is 210, so no row is lost to censoring. This is why H=30 preserves
    the full sample while H=60 and H=90 do not."""
    _, report = build_checkpoint_rows()
    assert sum(report.excluded_censored.values()) == 0


def test_no_row_exists_at_or_after_a_withdrawal() -> None:
    """The core leakage guarantee for the population layer."""
    population, _ = build_checkpoint_rows()
    event = population["date_unregistration"]
    assert not (event.notna() & (event <= population["checkpoint_day"])).any()


def test_every_positive_lies_strictly_inside_its_horizon() -> None:
    population, _ = build_checkpoint_rows()
    positives = population[population["label"] == 1]
    offset = positives["date_unregistration"] - positives["checkpoint_day"]
    assert offset.between(1, 30, inclusive="both").all()


def test_population_carries_provenance() -> None:
    population, _ = build_checkpoint_rows()
    assert (population["data_source"] == "real_oulad").all()
