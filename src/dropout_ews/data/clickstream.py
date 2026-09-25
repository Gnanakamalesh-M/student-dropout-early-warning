"""Pre-aggregate the OULAD clickstream with DuckDB.

``studentVle.csv`` is 10,655,280 rows and ~454 MB of CSV: one row per
(student, resource, day). Feature construction needs trailing-window
aggregates at six checkpoints for ~27,000 students, and doing that with a
pandas groupby per checkpoint over the resource-level frame is the scale risk
flagged in ADR-0002.

The fix is to aggregate once to **student-day** granularity — summing over
resources — and persist it as Parquet. That is the finest granularity any
as-of feature needs, so it is lossless for our purposes while being roughly a
third of the rows and a small fraction of the bytes.

DuckDB streams the CSV and does the aggregation out of core, so peak memory
stays low regardless of the file size. Nothing here loads the full clickstream
into pandas.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from dropout_ews.config.settings import DATA_DIR

RAW_STUDENT_VLE = DATA_DIR / "raw" / "oulad" / "studentVle.csv"
STUDENT_DAY_PARQUET = DATA_DIR / "interim" / "engagement_student_day.parquet"

# Aggregate to one row per student-presentation-day.
#
# `sum_click` is summed and resources are counted distinctly, because "20
# clicks on one page" and "20 clicks across eight resources" are different
# engagement signals and the second is worth keeping.
#
# The `?` null token (see loaders) cannot appear in the columns selected here:
# studentVle has no missing values in `date` or `sum_click`. It is still
# declared so a future change to the source cannot silently coerce.
#
# `COPY ... TO` does not accept a bound parameter for its destination, so the
# output path is interpolated. Single quotes are doubled to close the only
# injection route, and the value is always an internally constructed path
# rather than user input.
_AGGREGATE_SQL = """
COPY (
    SELECT
        code_module,
        code_presentation,
        id_student,
        CAST(date AS INTEGER)            AS course_day,
        SUM(sum_click)::INTEGER          AS clicks,
        COUNT(DISTINCT id_site)::INTEGER AS distinct_resources
    FROM read_csv(
        ?,
        header = true,
        nullstr = '?',
        columns = {
            'code_module': 'VARCHAR',
            'code_presentation': 'VARCHAR',
            'id_student': 'INTEGER',
            'id_site': 'INTEGER',
            'date': 'INTEGER',
            'sum_click': 'INTEGER'
        }
    )
    GROUP BY 1, 2, 3, 4
    ORDER BY 1, 2, 3, 4
) TO '$DESTINATION' (FORMAT PARQUET, COMPRESSION ZSTD)
"""


def _sql_literal(path: Path) -> str:
    """Escape a path for use as a SQL string literal."""
    return str(path).replace("'", "''")


def build_student_day_engagement(
    source: Path | None = None,
    destination: Path | None = None,
    force: bool = False,
) -> Path:
    """Aggregate the clickstream to student-day Parquet.

    Idempotent: returns the existing file unless ``force`` is set.
    """
    source = source or RAW_STUDENT_VLE
    destination = destination or STUDENT_DAY_PARQUET

    if destination.exists() and not force:
        return destination
    if not source.is_file():
        raise FileNotFoundError(f"{source} not found. Run: python scripts/download_data.py")

    destination.parent.mkdir(parents=True, exist_ok=True)
    # A partial Parquet file left behind by an interrupted run would be
    # indistinguishable from a complete one, so write to a temporary path and
    # move it into place only on success.
    temporary = destination.with_suffix(".parquet.tmp")
    temporary.unlink(missing_ok=True)

    connection = duckdb.connect()
    try:
        connection.execute(
            # A sentinel replace rather than str.format: the SQL contains a
            # `columns = { ... }` struct whose braces format() would misparse.
            _AGGREGATE_SQL.replace("$DESTINATION", _sql_literal(temporary)),
            [str(source)],
        )
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        connection.close()

    temporary.replace(destination)
    return destination


def load_student_day_engagement(path: Path | None = None) -> pd.DataFrame:
    """Load the aggregated student-day engagement frame.

    Builds it first if it is missing.
    """
    path = path or STUDENT_DAY_PARQUET
    if not path.exists():
        build_student_day_engagement(destination=path)
    return pd.read_parquet(path)


def engagement_summary(path: Path | None = None) -> dict[str, object]:
    """Report row counts and coverage, for the data card.

    Read via DuckDB rather than pandas so this stays cheap on a large file.
    """
    path = path or STUDENT_DAY_PARQUET
    if not path.exists():
        build_student_day_engagement(destination=path)

    connection = duckdb.connect()
    try:
        row = connection.execute(
            """
            SELECT
                COUNT(*)                        AS rows,
                COUNT(DISTINCT id_student)      AS students,
                MIN(course_day)                 AS min_day,
                MAX(course_day)                 AS max_day,
                SUM(clicks)                     AS total_clicks,
                SUM(CASE WHEN course_day < 0 THEN 1 ELSE 0 END) AS pre_start_rows
            FROM read_parquet(?)
            """,
            [str(path)],
        ).fetchone()
    finally:
        connection.close()

    assert row is not None
    return {
        "rows": row[0],
        "students": row[1],
        "min_course_day": row[2],
        "max_course_day": row[3],
        "total_clicks": row[4],
        "pre_start_rows": row[5],
        "file_mb": round(path.stat().st_size / 1e6, 1),
    }
