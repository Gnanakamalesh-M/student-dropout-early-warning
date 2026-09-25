"""Loaders for the raw OULAD CSVs.

Every raw-data quirk is handled here, once, so that nothing downstream has to
know about them. The quirks below were found by inspecting the actual files on
2026-09-25, not assumed:

1. **Missing values are the literal string ``"?"``**, not an empty field. This
   is the most dangerous quirk in the dataset: read without ``na_values``,
   ``date_unregistration`` parses as ``object`` and every null check silently
   returns "not null" for all 32,593 rows. Label construction then produces
   nonsense that still looks plausible.
2. **``imd_band`` has an inconsistent category**: ``"10-20"`` is missing the
   percent sign that all nine other bands carry, so a naive ``value_counts``
   reports eleven categories for a ten-band variable.
3. **``date_registration`` and ``date_unregistration`` are course-relative day
   offsets and are frequently negative** — students register before the
   presentation starts, and 3,089 of the 10,063 recorded withdrawals happen at
   or before day 0.
"""

from __future__ import annotations

import functools

import pandas as pd

from dropout_ews.config.settings import DATA_DIR
from dropout_ews.data.schemas import RAW_SCHEMAS

OULAD_DIR = DATA_DIR / "raw" / "oulad"

# OULAD encodes missing values as a literal question mark. See module docstring.
NA_VALUES = ["?"]

# Presentation codes sort chronologically under this key: the year, then B
# (February start) before J (October start). Used for the forward-in-time
# split in ADR-0003. Plain string sort happens to agree, but relying on that
# coincidence would break if a code like "2014A" ever appeared.
_MONTH_ORDER = {"B": 0, "J": 1}


def presentation_sort_key(code: str) -> tuple[int, int]:
    """Return a chronological sort key for an OULAD presentation code."""
    year, month = int(code[:4]), code[4]
    if month not in _MONTH_ORDER:
        raise ValueError(f"unrecognised presentation month letter in {code!r}")
    return year, _MONTH_ORDER[month]


def _read(name: str, validate: bool = True, **kwargs: object) -> pd.DataFrame:
    """Read one raw CSV and validate it against its schema."""
    path = OULAD_DIR / f"{name}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. Run: python scripts/download_data.py")
    frame = pd.read_csv(path, na_values=NA_VALUES, **kwargs)  # type: ignore[arg-type]
    if validate:
        frame = RAW_SCHEMAS[name].validate(frame, lazy=True)
    return frame


def _normalise_imd_band(series: pd.Series) -> pd.Series:
    """Repair the ``10-20`` category so all bands carry a percent sign."""
    return series.replace({"10-20": "10-20%"})


@functools.lru_cache(maxsize=1)
def load_courses() -> pd.DataFrame:
    """Module presentations and their length in days (22 rows)."""
    return _read("courses")


@functools.lru_cache(maxsize=1)
def load_student_info() -> pd.DataFrame:
    """One row per student-presentation, including ``final_result``.

    ``final_result`` is an outcome and must never become a feature; it is used
    only to cross-check labels and to report cohort composition.
    """
    frame = _read("studentInfo")
    frame["imd_band"] = _normalise_imd_band(frame["imd_band"])
    return frame


@functools.lru_cache(maxsize=1)
def load_student_registration() -> pd.DataFrame:
    """Registration and de-registration days.

    ``date_unregistration`` is the event time that defines the label. It is
    forbidden as a feature (docs/TASK_SPEC.md).
    """
    return _read("studentRegistration")


@functools.lru_cache(maxsize=1)
def load_assessments() -> pd.DataFrame:
    """Assessment definitions (206 rows)."""
    return _read("assessments")


@functools.lru_cache(maxsize=1)
def load_student_assessment() -> pd.DataFrame:
    """Submitted assessments with scores (173,912 rows)."""
    return _read("studentAssessment")


@functools.lru_cache(maxsize=1)
def load_vle() -> pd.DataFrame:
    """VLE resource metadata (6,364 rows)."""
    return _read("vle")


def load_student_vle(validate_sample: int | None = 100_000) -> pd.DataFrame:
    """Load the daily clickstream (~10.7M rows, ~454 MB as CSV).

    Not cached — it is large enough that holding it alongside derived frames is
    wasteful. Prefer the DuckDB aggregation in
    :mod:`dropout_ews.data.clickstream`, which never materialises the full
    frame in pandas.

    Args:
        validate_sample: validate a random sample of this many rows rather
            than all of them. Full validation of 10.7M rows is slow and adds
            little: the schema is checking column presence and value domains,
            which a large sample establishes just as well. Pass ``None`` to
            validate everything.
    """
    frame = pd.read_csv(OULAD_DIR / "studentVle.csv", na_values=NA_VALUES)
    schema = RAW_SCHEMAS["studentVle"]
    if validate_sample is None or validate_sample >= len(frame):
        schema.validate(frame, lazy=True)
    else:
        schema.validate(frame.sample(validate_sample, random_state=0), lazy=True)
    return frame


def presentations_in_order() -> list[str]:
    """Distinct presentation codes, chronologically ordered.

    Returns ``["2013B", "2013J", "2014B", "2014J"]`` for OULAD as published.
    Note these are *presentation codes*, not module-presentations: the dataset
    has 7 modules across these 4 codes, giving 22 module-presentations.
    """
    codes = load_courses()["code_presentation"].unique().tolist()
    return sorted(codes, key=presentation_sort_key)
