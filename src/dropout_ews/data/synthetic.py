"""Synthetic longitudinal cohort generator.

OULAD carries no semester GPA, backlog count, or subject-wise attendance, and
those dimensions are central to the product this project describes. Rather
than fabricate them onto real students, this module generates a separate,
clearly-labelled synthetic cohort from an explicit causal model.

READ THIS BEFORE QUOTING ANY METRIC COMPUTED ON THIS DATA
---------------------------------------------------------

Metrics on synthetic data measure **pipeline correctness, not predictive
performance**. The data-generating process below deliberately encodes the same
relationships the feature engineering is designed to detect, so a model will
score well on it almost by construction. That circularity is unavoidable in
synthetic data; publishing the generating process is what keeps it honest
rather than hidden. Headline metrics come from OULAD alone (ADR-0002).

The causal model
----------------

Per-student latent traits, drawn once::

    propensity       ~ N(0, 1)   engagement disposition
    prior_attainment ~ N(0, 1)   academic preparedness on entry
    resilience       ~ N(0, 1)   tendency to persist through difficulty

Per-semester generative chain::

    shock_t          ~ Bernoulli(p_shock)         financial / health / caring
    effort_t         = AR(1) in propensity, drift, and accumulated shock
    attendance_t     <- effort_t                  + noise
    subject_marks_t  <- attendance_t, prior_attainment, subject difficulty
    gpa_t            <- subject_marks_t
    backlogs_t       <- backlogs_{t-1} + failed subjects this semester
    lms_activity_t   <- effort_t                  + noise
    dropout hazard_t <- attendance decline, gpa, backlogs, shock, -resilience

The drift term is what produces genuine declining trajectories (the
86% -> 78% -> 63% pattern the system is meant to catch) rather than students
who are simply low-performing throughout. A fixed share of students receive a
negative drift.

The hazard intercept is **calibrated by bisection** so the cohort hits a
configured overall dropout rate, instead of being a hand-tuned constant that
silently changes meaning whenever another coefficient is touched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

DATA_SOURCE = "synthetic"

# Subject difficulty offsets, in GPA points. Fixed rather than random so the
# same subject is comparably hard across cohorts and semesters, which is what
# makes cohort-relative features meaningful.
SUBJECTS: dict[str, float] = {
    "MATH": -0.45,
    "PHYS": -0.30,
    "PROG": -0.10,
    "DBMS": 0.00,
    "COMM": 0.35,
    "ELEC": 0.15,
}

PASS_MARK = 40.0


@dataclass(frozen=True)
class SyntheticConfig:
    """Generator parameters. Every default is documented and seeded."""

    n_students: int = 4_000
    n_semesters: int = 8
    seed: int = 20260925

    # Target share of the cohort that drops out at some point. The hazard
    # intercept is calibrated to hit this, so changing other coefficients does
    # not silently move the base rate.
    target_dropout_rate: float = 0.22

    # Share of students given a negative effort drift, producing declining
    # trajectories rather than uniformly weak performance.
    declining_share: float = 0.30
    declining_drift: float = -0.34
    improving_share: float = 0.15
    improving_drift: float = 0.30

    # AR(1) persistence of effort between semesters.
    effort_persistence: float = 0.72
    effort_noise: float = 0.42

    # Per-semester probability of an external shock, and its effort penalty.
    shock_probability: float = 0.08
    shock_effort_penalty: float = 0.85
    shock_persistence: float = 0.55

    # Attendance: mapped from effort onto a realistic percentage range.
    attendance_mean: float = 78.0
    attendance_effort_scale: float = 11.0
    attendance_noise: float = 5.5
    subject_attendance_noise: float = 6.0

    # Marks: out of 100.
    mark_mean: float = 58.0
    mark_attendance_scale: float = 0.42
    mark_attainment_scale: float = 7.5
    mark_noise: float = 9.0
    subject_difficulty_scale: float = 6.0

    # Engagement.
    lms_sessions_mean: float = 24.0
    lms_effort_scale: float = 9.0

    # Hazard coefficients (log-odds). Signs encode the direction the model is
    # expected to learn; magnitudes are chosen so no single factor dominates.
    hazard_attendance_decline: float = 1.15
    hazard_low_gpa: float = 1.05
    hazard_backlogs: float = 0.42
    hazard_shock: float = 0.95
    hazard_resilience: float = -0.55
    hazard_semester_decay: float = -0.12

    # Missingness, applied after generation. Real institutional data has gaps;
    # a generator without them produces features that never exercise the
    # imputation path.
    missing_attendance_rate: float = 0.03
    missing_lms_rate: float = 0.05

    subjects: dict[str, float] = field(default_factory=lambda: dict(SUBJECTS))

    def __post_init__(self) -> None:
        if not 0.0 < self.target_dropout_rate < 1.0:
            raise ValueError("target_dropout_rate must be in (0, 1)")
        if self.declining_share + self.improving_share > 1.0:
            raise ValueError("declining_share + improving_share must not exceed 1")
        if self.n_semesters < 3:
            raise ValueError("need at least 3 semesters for trend features")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    result: np.ndarray = 1.0 / (1.0 + np.exp(-x))
    return result


@dataclass
class SyntheticCohort:
    """The generated cohort, in the shape a real institution would hold it."""

    students: pd.DataFrame
    """One row per student: program, entry cohort, dropout semester."""

    semester_records: pd.DataFrame
    """One row per (student, semester): attendance, GPA, backlogs, engagement."""

    subject_attendance: pd.DataFrame
    """One row per (student, semester, subject)."""

    calibration: dict[str, Any]
    """What the hazard calibration actually achieved, for the data card."""


class SyntheticCohortGenerator:
    """Generate a seeded synthetic longitudinal cohort."""

    def __init__(self, config: SyntheticConfig | None = None) -> None:
        self.config = config or SyntheticConfig()

    # -- latent traits -----------------------------------------------------

    def _draw_traits(self, rng: np.random.Generator) -> pd.DataFrame:
        config = self.config
        n = config.n_students

        propensity = rng.normal(0, 1, n)
        # Prior attainment correlates with propensity: students arriving
        # better prepared also tend to engage more. Independent draws would
        # make the two features artificially orthogonal.
        prior_attainment = 0.35 * propensity + np.sqrt(1 - 0.35**2) * rng.normal(0, 1, n)
        resilience = rng.normal(0, 1, n)

        # Assign trajectory shapes. Drift is what separates "declining" from
        # "consistently weak", and the system is meant to detect the former.
        trajectory = np.full(n, "stable", dtype=object)
        shuffled = rng.permutation(n)
        n_declining = int(config.declining_share * n)
        n_improving = int(config.improving_share * n)
        trajectory[shuffled[:n_declining]] = "declining"
        trajectory[shuffled[n_declining : n_declining + n_improving]] = "improving"

        drift = np.where(
            trajectory == "declining",
            config.declining_drift,
            np.where(trajectory == "improving", config.improving_drift, 0.0),
        )

        return pd.DataFrame(
            {
                "student_id": np.arange(1, n + 1),
                "propensity": propensity,
                "prior_attainment": prior_attainment,
                "resilience": resilience,
                "trajectory_shape": trajectory,
                "effort_drift": drift,
            }
        )

    # -- per-semester simulation -------------------------------------------

    def _simulate(
        self, traits: pd.DataFrame, hazard_intercept: float, rng: np.random.Generator
    ) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
        """Run the generative chain. Returns semester rows, subject rows, and
        each student's dropout semester (0 = retained)."""
        config = self.config
        n = len(traits)
        subjects = list(config.subjects)

        propensity = traits["propensity"].to_numpy()
        attainment = traits["prior_attainment"].to_numpy()
        resilience = traits["resilience"].to_numpy()
        drift = traits["effort_drift"].to_numpy()

        effort = propensity.copy()
        # Initialise the shock carry-over at its stationary mean rather than
        # zero. Starting from zero makes it ramp up over the first few
        # semesters, which imposes a spurious cohort-wide downward drift on
        # attendance and GPA — an artifact of the burn-in, not of the causal
        # model. That artifact would then be picked up by exactly the trend
        # features this project is testing, making them look informative for
        # the wrong reason.
        stationary_shock = (
            config.shock_probability
            * config.shock_effort_penalty
            / (1.0 - config.shock_persistence)
        )
        shock_carry = np.full(n, stationary_shock)
        backlogs = np.zeros(n, dtype=int)
        active = np.ones(n, dtype=bool)
        dropout_semester = np.zeros(n, dtype=int)
        first_attendance = np.full(n, np.nan)

        semester_rows: list[pd.DataFrame] = []
        subject_rows: list[pd.DataFrame] = []

        for semester in range(1, config.n_semesters + 1):
            # --- effort: AR(1) with drift and decaying shock ---
            shock = rng.random(n) < config.shock_probability
            shock_carry = (
                config.shock_persistence * shock_carry + shock * config.shock_effort_penalty
            )
            effort = (
                config.effort_persistence * effort
                + (1 - config.effort_persistence) * propensity
                + drift
                - shock_carry
                + rng.normal(0, config.effort_noise, n)
            )

            # --- attendance ---
            attendance = (
                config.attendance_mean
                + config.attendance_effort_scale * effort
                + rng.normal(0, config.attendance_noise, n)
            ).clip(0, 100)

            # --- subject marks, GPA, backlogs ---
            marks = np.empty((n, len(subjects)))
            for index, subject in enumerate(subjects):
                difficulty = config.subjects[subject] * config.subject_difficulty_scale
                marks[:, index] = (
                    config.mark_mean
                    + config.mark_attendance_scale * (attendance - config.attendance_mean)
                    + config.mark_attainment_scale * attainment
                    + difficulty
                    + rng.normal(0, config.mark_noise, n)
                ).clip(0, 100)

            failed = (marks < PASS_MARK).sum(axis=1)
            backlogs = backlogs + failed
            # GPA on a 10-point scale, the convention in Indian universities.
            gpa = (marks.mean(axis=1) / 10.0).clip(0, 10)

            # --- engagement ---
            lms_sessions = np.maximum(
                0,
                rng.poisson(
                    np.maximum(0.5, config.lms_sessions_mean + config.lms_effort_scale * effort)
                ),
            )
            assignments_set = 6
            submission_probability = _sigmoid(0.8 * effort + 0.9)
            assignments_submitted = rng.binomial(assignments_set, submission_probability)

            # Decline against the student's own first-semester baseline. NaN in
            # semester 1 (no baseline yet) is expected and handled below, so
            # the invalid-operation warning is suppressed rather than masked.
            with np.errstate(invalid="ignore"):
                attendance_vs_baseline = attendance - first_attendance

            frame = pd.DataFrame(
                {
                    "student_id": traits["student_id"].to_numpy(),
                    "semester": semester,
                    "attendance_pct": attendance.round(2),
                    "gpa": gpa.round(3),
                    "subjects_attempted": len(subjects),
                    "subjects_failed": failed,
                    "backlogs_active": backlogs,
                    "assignments_set": assignments_set,
                    "assignments_submitted": assignments_submitted,
                    "lms_sessions": lms_sessions,
                    "lms_active_days": np.minimum(
                        90, rng.poisson(np.maximum(0.5, 0.55 * lms_sessions))
                    ),
                    "external_shock": shock.astype("int8"),
                }
            )
            semester_rows.append(frame.loc[active].copy())

            subject_frame = pd.DataFrame(
                {
                    "student_id": np.repeat(traits["student_id"].to_numpy(), len(subjects)),
                    "semester": semester,
                    "subject_code": np.tile(subjects, n),
                    "mark": marks.reshape(-1).round(2),
                    "attendance_pct": (
                        np.repeat(attendance, len(subjects))
                        + rng.normal(0, config.subject_attendance_noise, n * len(subjects))
                    )
                    .clip(0, 100)
                    .round(2),
                }
            )
            subject_frame = subject_frame.loc[np.repeat(active, len(subjects))]
            subject_rows.append(subject_frame)

            # --- dropout hazard, evaluated at the end of the semester ---
            # Decline is measured against the student's own baseline, which is
            # the signal the system is built to detect.
            decline = np.nan_to_num(-attendance_vs_baseline / 20.0, nan=0.0).clip(0, None)
            gpa_shortfall = ((5.5 - gpa) / 2.0).clip(0, None)

            logit = (
                hazard_intercept
                + config.hazard_attendance_decline * decline
                + config.hazard_low_gpa * gpa_shortfall
                + config.hazard_backlogs * np.minimum(backlogs, 8)
                + config.hazard_shock * shock
                + config.hazard_resilience * resilience
                + config.hazard_semester_decay * semester
            )
            hazard = _sigmoid(logit)

            # The final semester has no "next semester" to fail to return to,
            # so no dropout event is generated there.
            if semester < config.n_semesters:
                leaves = active & (rng.random(n) < hazard)
                dropout_semester[leaves] = semester
                active = active & ~leaves

            first_attendance = np.where(np.isnan(first_attendance), attendance, first_attendance)

        return (
            pd.concat(semester_rows, ignore_index=True),
            pd.concat(subject_rows, ignore_index=True),
            dropout_semester,
        )

    # -- hazard calibration ------------------------------------------------

    def _calibrate_intercept(self, traits: pd.DataFrame) -> tuple[float, dict[str, Any]]:
        """Bisect the hazard intercept to hit ``target_dropout_rate``.

        Calibrating rather than hand-tuning means the configured dropout rate
        keeps its meaning when any other coefficient changes.
        """
        config = self.config
        low, high = -12.0, 4.0
        achieved = float("nan")
        intercept = low

        for _ in range(40):
            intercept = 0.5 * (low + high)
            # A fresh generator per trial so the bisection is deterministic
            # and monotone; reusing one would make each evaluation depend on
            # the sequence of previous trials.
            rng = np.random.default_rng(config.seed + 9_000)
            _, _, dropout_semester = self._simulate(traits, intercept, rng)
            achieved = float((dropout_semester > 0).mean())
            if abs(achieved - config.target_dropout_rate) < 0.002:
                break
            if achieved > config.target_dropout_rate:
                high = intercept
            else:
                low = intercept

        return intercept, {
            "hazard_intercept": round(intercept, 4),
            "target_dropout_rate": config.target_dropout_rate,
            "achieved_dropout_rate": round(achieved, 4),
        }

    # -- missingness -------------------------------------------------------

    def _apply_missingness(
        self, semester_records: pd.DataFrame, rng: np.random.Generator
    ) -> pd.DataFrame:
        config = self.config
        records = semester_records.copy()
        n = len(records)
        # Missing completely at random. Real gaps are rarely MCAR, but
        # generating a specific non-random mechanism would bake an assumption
        # about it into the data; MCAR keeps the imputation path exercised
        # without asserting a pattern we cannot justify.
        records.loc[rng.random(n) < config.missing_attendance_rate, "attendance_pct"] = np.nan
        missing_lms = rng.random(n) < config.missing_lms_rate
        records.loc[missing_lms, ["lms_sessions", "lms_active_days"]] = np.nan
        return records

    # -- public API --------------------------------------------------------

    def generate(self) -> SyntheticCohort:
        config = self.config
        traits = self._draw_traits(np.random.default_rng(config.seed))
        intercept, calibration = self._calibrate_intercept(traits)

        rng = np.random.default_rng(config.seed + 9_000)
        semester_records, subject_attendance, dropout_semester = self._simulate(
            traits, intercept, rng
        )
        semester_records = self._apply_missingness(
            semester_records, np.random.default_rng(config.seed + 1)
        )

        students = traits[["student_id", "trajectory_shape", "prior_attainment"]].copy()
        students["dropout_semester"] = np.where(dropout_semester > 0, dropout_semester, np.nan)
        students["program"] = "BTech-CSE"
        students["entry_cohort"] = 2024
        students["data_source"] = DATA_SOURCE
        semester_records["data_source"] = DATA_SOURCE
        subject_attendance["data_source"] = DATA_SOURCE

        calibration["n_students"] = config.n_students
        calibration["n_semesters"] = config.n_semesters
        calibration["seed"] = config.seed
        calibration["semester_rows"] = len(semester_records)
        calibration["subject_rows"] = len(subject_attendance)

        return SyntheticCohort(
            students=students,
            semester_records=semester_records,
            subject_attendance=subject_attendance,
            calibration=calibration,
        )


def build_synthetic_checkpoints(
    cohort: SyntheticCohort, horizon_semesters: int = 1
) -> pd.DataFrame:
    """Build the labelled (student, semester) population for the cohort.

    Mirrors :func:`dropout_ews.data.checkpoints.build_checkpoint_rows` and
    obeys the same rules from docs/TASK_SPEC.md:

    * a row exists only for a semester the student actually attended;
    * the label is 1 if the student leaves within the next
      ``horizon_semesters``;
    * rows whose label window extends past the final semester are dropped
      rather than assumed negative.
    """
    if horizon_semesters < 1:
        raise ValueError("horizon_semesters must be at least 1")

    records = cohort.semester_records[["student_id", "semester"]].copy()
    dropout = cohort.students.set_index("student_id")["dropout_semester"]
    n_semesters = int(cohort.semester_records["semester"].max())

    records["dropout_semester"] = records["student_id"].map(dropout)
    # The label window is (t, t + H] in semesters, so it must end no later
    # than the final observed semester.
    observed = records.loc[records["semester"] + horizon_semesters <= n_semesters].copy()

    event = observed["dropout_semester"]
    observed["label"] = (
        event.notna()
        & event.ge(observed["semester"])
        & event.le(observed["semester"] + horizon_semesters - 1)
    ).astype("int8")
    observed["horizon_semesters"] = horizon_semesters
    observed["data_source"] = DATA_SOURCE

    # A student who left at the end of semester t attended semester t, so a
    # row at t is legitimate and carries the positive label. No row may exist
    # after that, which the semester_records join already guarantees.
    stale = observed["dropout_semester"].notna() & observed["semester"].gt(
        observed["dropout_semester"]
    )
    if stale.any():
        raise ValueError(f"{int(stale.sum())} rows exist after the student left")

    return observed.reset_index(drop=True)
