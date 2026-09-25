"""Tests for the checkpoint population construction.

These run against hand-built fixtures rather than the real dataset, so they
execute fast and in CI without the 46 MB download. Tests that need the real
data live in ``tests/integration`` and are marked.

The cases below are chosen to cover every inclusion rule in docs/TASK_SPEC.md
and each real data quirk found on 2026-09-25.
"""

from __future__ import annotations

import pandas as pd
import pytest

from dropout_ews.data import checkpoints as cp
from dropout_ews.data.checkpoints import (
    PopulationReport,
    _assert_population_invariants,
    build_checkpoint_rows,
)

CHECKPOINTS = [30, 60, 90]
HORIZON = 30


def _student(
    sid: int,
    unregistration: float | None,
    result: str = "Pass",
    length: int = 240,
    module: str = "AAA",
    presentation: str = "2013J",
) -> dict[str, object]:
    return {
        "code_module": module,
        "code_presentation": presentation,
        "id_student": sid,
        "date_registration": -50.0,
        "date_unregistration": unregistration,
        "module_presentation_length": length,
        "final_result": result,
    }


@pytest.fixture
def patched_base(monkeypatch: pytest.MonkeyPatch):
    """Replace the real loaders with an in-memory cohort."""

    def _install(rows: list[dict[str, object]]) -> None:
        frame = pd.DataFrame(rows)
        registration = frame[
            [
                "code_module",
                "code_presentation",
                "id_student",
                "date_registration",
                "date_unregistration",
            ]
        ]
        courses = (
            frame[["code_module", "code_presentation", "module_presentation_length"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        info = frame[["code_module", "code_presentation", "id_student", "final_result"]]
        monkeypatch.setattr(cp, "load_student_registration", lambda: registration)
        monkeypatch.setattr(cp, "load_courses", lambda: courses)
        monkeypatch.setattr(cp, "load_student_info", lambda: info)

    return _install


def _build(rows, **kwargs):
    return build_checkpoint_rows(checkpoints=CHECKPOINTS, horizon_days=HORIZON, **kwargs)


# ---------------------------------------------------------------------------
# Label correctness
# ---------------------------------------------------------------------------


def test_completer_is_negative_at_every_checkpoint(patched_base) -> None:
    patched_base([_student(1, None)])
    population, _ = _build(None)
    assert len(population) == len(CHECKPOINTS)
    assert population["label"].tolist() == [0, 0, 0]


def test_withdrawal_is_positive_only_in_the_horizon_window(patched_base) -> None:
    """A student leaving on day 75 is positive at checkpoint 60 (75 is within
    60+30) and negative at 30 (75 > 60). No row exists at 90 — already gone."""
    patched_base([_student(1, 75.0, result="Withdrawn")])
    population, _ = _build(None)
    labels = dict(zip(population["checkpoint_day"], population["label"], strict=True))
    assert labels == {30: 0, 60: 1}


def test_horizon_boundary_is_inclusive_at_the_upper_edge(patched_base) -> None:
    """Withdrawal exactly at t + H counts as positive: the window is (t, t+H]."""
    patched_base([_student(1, 60.0, result="Withdrawn")])
    population, _ = _build(None)
    labels = dict(zip(population["checkpoint_day"], population["label"], strict=True))
    assert labels[30] == 1  # 60 == 30 + 30


def test_no_row_is_emitted_at_or_after_withdrawal(patched_base) -> None:
    """The boundary case: withdrawal exactly on a checkpoint day. That student
    is no longer enrolled at t, so no row may exist — emitting one would leak
    an outcome that has already happened."""
    patched_base([_student(1, 60.0, result="Withdrawn")])
    population, _ = _build(None)
    assert 60 not in population["checkpoint_day"].tolist()


# ---------------------------------------------------------------------------
# Real data quirks
# ---------------------------------------------------------------------------


def test_withdrawn_without_event_time_is_excluded_and_counted(patched_base) -> None:
    """93 real students are Withdrawn with no de-registration day. They cannot
    be labelled either way, so they are dropped and reported."""
    patched_base([_student(1, None, result="Withdrawn"), _student(2, None)])
    population, report = _build(None)
    assert report.withdrawn_without_event_time == 1
    assert population["id_student"].unique().tolist() == [2]


def test_event_time_wins_over_final_result(patched_base) -> None:
    """9 real students have a de-registration day but final_result == 'Fail'.
    A dated event is a concrete fact; final_result is an administrative
    summary. The event time is authoritative."""
    patched_base([_student(1, 45.0, result="Fail")])
    population, _ = _build(None)
    labels = dict(zip(population["checkpoint_day"], population["label"], strict=True))
    assert labels[30] == 1


def test_pre_course_withdrawals_are_never_scored(patched_base) -> None:
    """5,127 real students withdraw at or before the first checkpoint. No
    in-course behaviour exists to observe, so no engagement model could flag
    them; they are out of scope and must not appear."""
    patched_base([_student(1, -10.0, result="Withdrawn"), _student(2, 0.0, result="Withdrawn")])
    population, report = _build(None)
    assert population.empty or population["label"].sum() == 0
    assert len(population) == 0
    assert report.withdrew_before_first_checkpoint == 2


def test_late_administrative_withdrawal_stays_negative(patched_base) -> None:
    """Real de-registration days reach 444, beyond the longest presentation.
    Such a student was enrolled throughout, so every row is negative."""
    patched_base([_student(1, 444.0, result="Withdrawn", length=269)])
    population, _ = _build(None)
    assert len(population) == len(CHECKPOINTS)
    assert population["label"].sum() == 0


# ---------------------------------------------------------------------------
# Censoring
# ---------------------------------------------------------------------------


def test_censored_rows_are_dropped_not_labelled_negative(patched_base) -> None:
    """A presentation of 100 days cannot support checkpoint 90 with H=30,
    because day 120 was never observed. Calling it negative would assert an
    unobserved fact and bias the model toward optimism."""
    patched_base([_student(1, None, length=100)])
    population, report = _build(None)
    assert population["checkpoint_day"].tolist() == [30, 60]
    assert report.excluded_censored[90] == 1


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------


def test_invariant_rejects_a_post_event_row() -> None:
    """The guard must fire if construction ever emits a row at/after the event."""
    bad = pd.DataFrame(
        {
            "code_module": ["AAA"],
            "code_presentation": ["2013J"],
            "id_student": [1],
            "date_unregistration": [20.0],
            "checkpoint_day": [30],
            "label": [0],
        }
    )
    with pytest.raises(ValueError, match="at or after the student's withdrawal"):
        _assert_population_invariants(bad, HORIZON)


def test_invariant_rejects_a_positive_outside_its_horizon() -> None:
    bad = pd.DataFrame(
        {
            "code_module": ["AAA"],
            "code_presentation": ["2013J"],
            "id_student": [1],
            "date_unregistration": [200.0],
            "checkpoint_day": [30],
            "label": [1],
        }
    )
    with pytest.raises(ValueError, match="outside its horizon"):
        _assert_population_invariants(bad, HORIZON)


def test_invariant_rejects_duplicate_student_checkpoint_rows() -> None:
    bad = pd.DataFrame(
        {
            "code_module": ["AAA", "AAA"],
            "code_presentation": ["2013J", "2013J"],
            "id_student": [1, 1],
            "date_unregistration": [None, None],
            "checkpoint_day": [30, 30],
            "label": [0, 0],
        }
    )
    with pytest.raises(ValueError, match="duplicate"):
        _assert_population_invariants(bad, HORIZON)


def test_same_student_id_across_modules_is_not_a_duplicate(patched_base) -> None:
    """OULAD reuses id_student across modules: a student can appear in several
    module-presentations. The key is (module, presentation, student), and
    treating id_student alone as unique would wrongly collapse these rows.
    It also means grouped CV must group on id_student to avoid leakage."""
    patched_base(
        [
            _student(1, None, module="AAA"),
            _student(1, None, module="BBB"),
        ]
    )
    population, _ = _build(None)
    assert len(population) == 2 * len(CHECKPOINTS)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def test_report_totals_and_markdown() -> None:
    report = PopulationReport(
        source_rows=100,
        rows_by_checkpoint={30: 80, 60: 70},
        positives_by_checkpoint={30: 8, 60: 7},
    )
    assert report.total_rows == 150
    assert report.total_positives == 15
    assert report.positive_rate == pytest.approx(0.1)
    rendered = report.to_markdown()
    assert "| 30 | 80 | 8 | 10.00% |" in rendered
