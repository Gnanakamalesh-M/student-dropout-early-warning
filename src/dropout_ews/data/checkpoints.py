"""Construct the modelling population: one row per (student, checkpoint).

This module implements docs/TASK_SPEC.md. It produces the *labelled skeleton*
only — identifiers, the checkpoint day, and the target. Features are attached
later by :mod:`dropout_ews.features.builder`, which is deliberately separate so
that the population definition can be tested without any feature logic.

Getting this wrong invalidates everything downstream, so each inclusion rule is
implemented as its own named, tested step rather than as one large boolean
expression.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from dropout_ews.config.settings import load_feature_config
from dropout_ews.data.loaders import (
    load_courses,
    load_student_info,
    load_student_registration,
)

KEY_COLUMNS = ["code_module", "code_presentation", "id_student"]

ExclusionReason = Literal[
    "withdrawn_without_event_time",
    "already_withdrawn_at_checkpoint",
    "label_horizon_censored",
]


@dataclass
class PopulationReport:
    """Auditable counts for every row the construction discarded.

    Published in docs/DATA_CARD.md. Exclusions that are only implied by code
    are exclusions nobody notices, and silently dropping students is how a
    dataset quietly stops representing the population it claims to.
    """

    source_rows: int = 0
    withdrawn_without_event_time: int = 0
    withdrew_before_first_checkpoint: int = 0
    rows_by_checkpoint: dict[int, int] = field(default_factory=dict)
    positives_by_checkpoint: dict[int, int] = field(default_factory=dict)
    excluded_already_withdrawn: dict[int, int] = field(default_factory=dict)
    excluded_censored: dict[int, int] = field(default_factory=dict)

    @property
    def total_rows(self) -> int:
        return sum(self.rows_by_checkpoint.values())

    @property
    def total_positives(self) -> int:
        return sum(self.positives_by_checkpoint.values())

    @property
    def positive_rate(self) -> float:
        return self.total_positives / self.total_rows if self.total_rows else float("nan")

    def to_markdown(self) -> str:
        lines = [
            f"- Source student-presentation rows: **{self.source_rows:,}**",
            "- Excluded, withdrawn with no recorded event time: "
            f"**{self.withdrawn_without_event_time:,}**",
            "- Withdrew at or before the first checkpoint (never scored): "
            f"**{self.withdrew_before_first_checkpoint:,}**",
            "",
            "| Checkpoint | Scored rows | Positives | Positive rate | Excluded: already withdrawn | Excluded: censored |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
        for t in sorted(self.rows_by_checkpoint):
            n = self.rows_by_checkpoint[t]
            p = self.positives_by_checkpoint[t]
            lines.append(
                f"| {t} | {n:,} | {p:,} | {100 * p / n:.2f}% | "
                f"{self.excluded_already_withdrawn.get(t, 0):,} | "
                f"{self.excluded_censored.get(t, 0):,} |"
            )
        lines.append(
            f"| **Total** | **{self.total_rows:,}** | **{self.total_positives:,}** | "
            f"**{100 * self.positive_rate:.2f}%** | | |"
        )
        return "\n".join(lines)


def _base_frame() -> tuple[pd.DataFrame, int]:
    """Join registration, course length, and outcome into one row per student.

    Returns the frame and the number of label-ambiguous rows removed.
    """
    registration = load_student_registration()
    courses = load_courses()
    info = load_student_info()[[*KEY_COLUMNS, "final_result"]]

    frame = registration.merge(
        courses, on=["code_module", "code_presentation"], how="left", validate="many_to_one"
    ).merge(info, on=KEY_COLUMNS, how="left", validate="one_to_one")

    if frame["module_presentation_length"].isna().any():
        raise ValueError("a student-presentation has no matching course length")

    # 93 students are recorded as Withdrawn but carry no de-registration day,
    # so their event time is unknown. They cannot be labelled either way:
    # calling them negative asserts they stayed, calling them positive invents
    # a date. They are excluded and counted.
    ambiguous = (frame["final_result"] == "Withdrawn") & frame["date_unregistration"].isna()
    n_ambiguous = int(ambiguous.sum())

    # The converse case (9 students with a de-registration day but a final
    # result of "Fail") is NOT excluded: a recorded event time is a concrete,
    # dated fact, whereas `final_result` is an administrative summary. The
    # event time is treated as authoritative, which is also the only choice
    # consistent with using it to define the label everywhere else.
    return frame.loc[~ambiguous].reset_index(drop=True), n_ambiguous


def build_checkpoint_rows(
    checkpoints: list[int] | None = None,
    horizon_days: int | None = None,
) -> tuple[pd.DataFrame, PopulationReport]:
    """Build the labelled (student, checkpoint) population.

    Args:
        checkpoints: course-relative days to score at. Defaults to
            ``features.yaml``.
        horizon_days: label horizon ``H``. Defaults to ``features.yaml``.

    Returns:
        The population frame and an auditable :class:`PopulationReport`.

    The returned frame carries ``date_unregistration`` so that downstream
    validation can assert it was not used as a feature, and so the batch
    scorer can reconstruct labels. It is in the forbidden list
    (``features.yaml``) and the feature allowlist prevents it reaching a model.
    """
    config = load_feature_config()
    checkpoints = checkpoints if checkpoints is not None else config.task.checkpoints
    horizon = horizon_days if horizon_days is not None else config.task.horizon_days
    drop_censored = config.task.drop_censored_rows

    base, n_ambiguous = _base_frame()
    report = PopulationReport(
        source_rows=len(base) + n_ambiguous,
        withdrawn_without_event_time=n_ambiguous,
    )

    unregistration = base["date_unregistration"]
    first_checkpoint = min(checkpoints)
    # Withdrawals at or before day 0 are a large group (3,089 of 10,063) and
    # are out of scope by construction: no in-course behaviour exists to
    # observe, so no engagement-based model could ever flag them. They are
    # excluded implicitly by the eligibility rule below, but counted
    # explicitly here so the data card can state the scope honestly.
    report.withdrew_before_first_checkpoint = int((unregistration <= first_checkpoint).sum())

    parts: list[pd.DataFrame] = []
    for t in checkpoints:
        # Rule 1 — still enrolled at t. Rows at or after withdrawal are never
        # emitted: they would leak the outcome, being trivially separable.
        eligible = base.loc[unregistration.isna() | (unregistration > t)]
        report.excluded_already_withdrawn[t] = len(base) - len(eligible)

        # Rule 2 — the label window must be fully observed. A horizon running
        # past the end of the presentation is administratively censored, and
        # such rows are dropped rather than assumed negative (TASK_SPEC.md).
        if drop_censored:
            observed = eligible.loc[eligible["module_presentation_length"] >= t + horizon]
        else:
            observed = eligible
        report.excluded_censored[t] = len(eligible) - len(observed)

        rows = observed[[*KEY_COLUMNS, "date_unregistration", "module_presentation_length"]].copy()
        rows["checkpoint_day"] = t
        rows["horizon_days"] = horizon
        # Rule 3 — the label: did the student withdraw inside (t, t + H]?
        rows["label"] = (
            rows["date_unregistration"].gt(t) & rows["date_unregistration"].le(t + horizon)
        ).astype("int8")
        # Chronological ordering key for the forward-in-time split (ADR-0003).
        rows["data_source"] = "real_oulad"

        report.rows_by_checkpoint[t] = len(rows)
        report.positives_by_checkpoint[t] = int(rows["label"].sum())
        parts.append(rows)

    population = pd.concat(parts, ignore_index=True)
    _assert_population_invariants(population, horizon)
    return population, report


def _assert_population_invariants(population: pd.DataFrame, horizon: int) -> None:
    """Fail loudly if the constructed population violates the task spec.

    These run on every build rather than only in tests: the cost is
    negligible against the cost of training on a silently malformed
    population.
    """
    # An empty population is not an invariant violation: a cohort consisting
    # only of pre-course withdrawals legitimately yields no scorable rows, and
    # every check below holds vacuously. Callers that require a non-empty
    # population assert that themselves — see scripts/build_dataset.py.
    if population.empty:
        return

    duplicated = population.duplicated([*KEY_COLUMNS, "checkpoint_day"]).sum()
    if duplicated:
        raise ValueError(f"{duplicated} duplicate (student, checkpoint) rows")

    # No row may exist at or after the student's withdrawal.
    unreg = population["date_unregistration"]
    post_event = (unreg.notna() & (unreg <= population["checkpoint_day"])).sum()
    if post_event:
        raise ValueError(f"{post_event} rows are at or after the student's withdrawal")

    # A positive label must correspond to an event strictly inside the horizon.
    positives = population.loc[population["label"] == 1, ["date_unregistration", "checkpoint_day"]]
    if positives["date_unregistration"].isna().any():
        raise ValueError("a positive row has no event time")
    offset = positives["date_unregistration"] - positives["checkpoint_day"]
    if not offset.between(1, horizon, inclusive="both").all():
        raise ValueError("a positive label lies outside its horizon window")

    if not population["label"].isin([0, 1]).all():
        raise ValueError("labels must be 0 or 1")
