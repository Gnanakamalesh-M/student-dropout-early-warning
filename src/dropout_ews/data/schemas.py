"""Pandera schemas for the raw OULAD tables.

These are contracts on data we do not control. They exist to fail loudly and
early: a silently renamed column or an unexpected category would otherwise
surface much later as a confusing modelling bug, or worse, as a plausible
result computed from the wrong thing.

Table and column names follow OULAD as published (camelCase filenames,
snake_case columns); they are not renamed here so that the raw layer stays a
faithful mirror of the source. Renaming happens in the loaders.

Reference: Kuzilek, Hlosta & Zdrahal (2017), "Open University Learning
Analytics dataset", *Scientific Data* 4:170171.
"""

from __future__ import annotations

import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema

# ---------------------------------------------------------------------------
# Shared domains
# ---------------------------------------------------------------------------

# Presentation codes: a year followed by B (February start) or J (October
# start). The trailing letter encodes the month, which is what lets us order
# presentations chronologically for the forward-in-time split (ADR-0003).
PRESENTATION_PATTERN = r"^\d{4}[BJ]$"

FINAL_RESULTS = ["Pass", "Fail", "Withdrawn", "Distinction"]
ASSESSMENT_TYPES = ["TMA", "CMA", "Exam"]

# `date` columns are course-relative day offsets from the presentation start.
# Negative values are legitimate and meaningful: registration and some VLE
# activity happen before day 0. Any lower bound here must therefore be
# generous, and code must never assume dates are non-negative.
#
# Observed ranges in the published snapshot (measured 2026-09-25):
#
#   studentRegistration.date_registration    -322 ..  167
#   studentRegistration.date_unregistration  -365 ..  444
#   studentAssessment.date_submitted          -11 ..  608
#   assessments.date                           12 ..  261
#   studentVle.date                           -25 ..  269
#
# Both extremes exceed the longest presentation (269 days): de-registration
# and assessment submission are sometimes recorded administratively long after
# a course ends. These bounds are set wider than observed on purpose — they
# exist to catch a unit change or a parsing failure, not to assert the exact
# empirical range, which would make the schema brittle against a data refresh.
MIN_COURSE_DAY = -500
MAX_COURSE_DAY = 800


# ---------------------------------------------------------------------------
# Table schemas
# ---------------------------------------------------------------------------

COURSES_SCHEMA = DataFrameSchema(
    name="courses",
    description="One row per module presentation, with its length in days.",
    columns={
        "code_module": Column(str, nullable=False),
        "code_presentation": Column(str, pa.Check.str_matches(PRESENTATION_PATTERN)),
        # Presentation length bounds the checkpoints we can score: a checkpoint
        # beyond this, or whose horizon extends beyond it, is censored.
        "module_presentation_length": Column(int, pa.Check.in_range(1, 400)),
    },
    unique=["code_module", "code_presentation"],
    strict=False,
    coerce=True,
)

STUDENT_INFO_SCHEMA = DataFrameSchema(
    name="studentInfo",
    description="One row per student per presentation, with the final outcome.",
    columns={
        "code_module": Column(str, nullable=False),
        "code_presentation": Column(str, pa.Check.str_matches(PRESENTATION_PATTERN)),
        "id_student": Column(int, nullable=False),
        # Demographics below are validated but routed to an isolated table and
        # excluded from the feature matrix; only the fairness audit reads them.
        # See docs/ETHICS.md.
        "gender": Column(str, nullable=True),
        "region": Column(str, nullable=True),
        "highest_education": Column(str, nullable=True),
        # imd_band (area deprivation) has genuinely missing values in OULAD.
        "imd_band": Column(str, nullable=True),
        "age_band": Column(str, nullable=True),
        "num_of_prev_attempts": Column(int, pa.Check.ge(0)),
        "studied_credits": Column(int, pa.Check.ge(0)),
        "disability": Column(str, nullable=True),
        # The outcome. Never a feature (ADR-0003); used only to derive labels.
        "final_result": Column(str, pa.Check.isin(FINAL_RESULTS)),
    },
    unique=["code_module", "code_presentation", "id_student"],
    strict=False,
    coerce=True,
)

STUDENT_REGISTRATION_SCHEMA = DataFrameSchema(
    name="studentRegistration",
    description="Registration and, where applicable, de-registration day.",
    columns={
        "code_module": Column(str, nullable=False),
        "code_presentation": Column(str, pa.Check.str_matches(PRESENTATION_PATTERN)),
        "id_student": Column(int, nullable=False),
        # Usually negative: students register before the presentation starts.
        "date_registration": Column(
            float, pa.Check.in_range(MIN_COURSE_DAY, MAX_COURSE_DAY), nullable=True
        ),
        # THE EVENT TIME. Null means the student did not de-register.
        # Strictly forbidden as a feature; it defines the label and nothing
        # else. See docs/TASK_SPEC.md.
        "date_unregistration": Column(
            float, pa.Check.in_range(MIN_COURSE_DAY, MAX_COURSE_DAY), nullable=True
        ),
    },
    unique=["code_module", "code_presentation", "id_student"],
    strict=False,
    coerce=True,
)

ASSESSMENTS_SCHEMA = DataFrameSchema(
    name="assessments",
    description="Assessment definitions per presentation.",
    columns={
        "code_module": Column(str, nullable=False),
        "code_presentation": Column(str, pa.Check.str_matches(PRESENTATION_PATTERN)),
        "id_assessment": Column(int, nullable=False),
        "assessment_type": Column(str, pa.Check.isin(ASSESSMENT_TYPES)),
        # Due day, course-relative. Null for some final exams in OULAD, where
        # the due date is taken to be the end of the presentation.
        "date": Column(float, pa.Check.in_range(MIN_COURSE_DAY, MAX_COURSE_DAY), nullable=True),
        "weight": Column(float, pa.Check.in_range(0, 100)),
    },
    unique=["id_assessment"],
    strict=False,
    coerce=True,
)

STUDENT_ASSESSMENT_SCHEMA = DataFrameSchema(
    name="studentAssessment",
    description="Submitted assessments with scores.",
    columns={
        "id_assessment": Column(int, nullable=False),
        "id_student": Column(int, nullable=False),
        # Submission day, course-relative. This is the timestamp that decides
        # whether a submission is visible at a given checkpoint.
        "date_submitted": Column(int, pa.Check.in_range(MIN_COURSE_DAY, MAX_COURSE_DAY)),
        # 1 if the result was transferred from a previous attempt.
        "is_banked": Column(int, pa.Check.isin([0, 1])),
        "score": Column(float, pa.Check.in_range(0, 100), nullable=True),
    },
    strict=False,
    coerce=True,
)

VLE_SCHEMA = DataFrameSchema(
    name="vle",
    description="VLE material metadata (one row per site/resource).",
    columns={
        "id_site": Column(int, nullable=False),
        "code_module": Column(str, nullable=False),
        "code_presentation": Column(str, pa.Check.str_matches(PRESENTATION_PATTERN)),
        "activity_type": Column(str, nullable=False),
        # Mostly null in OULAD; not relied upon.
        "week_from": Column(float, nullable=True),
        "week_to": Column(float, nullable=True),
    },
    unique=["id_site"],
    strict=False,
    coerce=True,
)

# The large one: roughly 10.7 million rows. Validating it in full is slow, so
# loaders validate a sample by default and the full frame only on request.
STUDENT_VLE_SCHEMA = DataFrameSchema(
    name="studentVle",
    description="Daily per-resource click counts. ~10.7M rows.",
    columns={
        "code_module": Column(str, nullable=False),
        "code_presentation": Column(str, pa.Check.str_matches(PRESENTATION_PATTERN)),
        "id_student": Column(int, nullable=False),
        "id_site": Column(int, nullable=False),
        # Course-relative day of the activity. Can be negative (pre-start).
        "date": Column(int, pa.Check.in_range(MIN_COURSE_DAY, MAX_COURSE_DAY)),
        "sum_click": Column(int, pa.Check.gt(0)),
    },
    strict=False,
    coerce=True,
)


RAW_SCHEMAS: dict[str, DataFrameSchema] = {
    "courses": COURSES_SCHEMA,
    "assessments": ASSESSMENTS_SCHEMA,
    "vle": VLE_SCHEMA,
    "studentInfo": STUDENT_INFO_SCHEMA,
    "studentRegistration": STUDENT_REGISTRATION_SCHEMA,
    "studentAssessment": STUDENT_ASSESSMENT_SCHEMA,
    "studentVle": STUDENT_VLE_SCHEMA,
}
"""Raw table name (matching the CSV stem) to its schema."""
