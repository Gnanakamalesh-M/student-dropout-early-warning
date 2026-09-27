"""Seed the database from the processed data and the registered model artifact.

Loads, in order: the active model version and threshold config, students,
demographics (into their isolated table), engagement and assessment events, and
demo users.

Scope is deliberate. The full clickstream is 1.8 million student-day rows, and
seeding all of it into Postgres for a portfolio demo buys nothing over seeding the
cohort the API actually serves. ``--cohort`` controls which presentations are
loaded and defaults to the held-out test presentation, so the demo shows the model
scoring students it never trained on. ``--all-cohorts`` loads everything and
reports how long it took rather than pretending it is free.

Usage::

    python scripts/seed_db.py                        # test cohort
    python scripts/seed_db.py --all-cohorts          # everything
    python scripts/seed_db.py --reset                # drop and recreate first
"""

from __future__ import annotations

import argparse
import time

import pandas as pd
from backend.app.db import models
from backend.app.db.session import get_engine, reset_engine, session_scope, sync_database_url
from backend.app.repository import student_code
from backend.app.security import Role, hash_password
from sqlalchemy import delete, select

from dropout_ews.config.settings import (
    load_feature_config,
    load_threshold_config,
)
from dropout_ews.data.checkpoints import build_checkpoint_rows
from dropout_ews.data.clickstream import load_student_day_engagement
from dropout_ews.data.loaders import load_assessments, load_student_assessment, load_student_info
from dropout_ews.evaluation.splits import make_temporal_split
from dropout_ews.models.registry import load_model

DEMO_USERS = (
    ("admin", Role.ADMIN, "admin-demo-password"),
    ("counsellor", Role.COUNSELLOR, "counsellor-demo-password"),
    ("analyst", Role.ANALYST, "analyst-demo-password"),
)


def _reset_schema() -> None:
    engine = get_engine()
    with engine.begin() as connection:
        connection.exec_driver_sql(models.DROP_RISK_HISTORY_VIEW_SQL)
    models.Base.metadata.drop_all(engine)
    models.Base.metadata.create_all(engine)
    with engine.begin() as connection:
        connection.exec_driver_sql(models.RISK_HISTORY_VIEW_SQL)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--reset", action="store_true", help="drop and recreate the schema")
    parser.add_argument(
        "--all-cohorts", action="store_true", help="seed every presentation, not just test"
    )
    args = parser.parse_args()

    if args.database_url:
        reset_engine()
        get_engine(args.database_url)
    print(f"Database: {sync_database_url(args.database_url)}")

    if args.reset:
        print("Resetting schema...")
        _reset_schema()

    started = time.perf_counter()
    config = load_feature_config()
    thresholds = load_threshold_config()
    _, metadata = load_model()

    population, _ = build_checkpoint_rows()
    split = make_temporal_split(population)
    if args.all_cohorts:
        wanted = set(population["code_presentation"].unique())
        cohort = population[population["code_presentation"].isin(wanted)]
    else:
        # Scope to the test SPLIT, not the test presentation. Those differ: the
        # split drops 1,588 students from 2014J because they also appear in an
        # earlier presentation (ADR-0003 amendment). Seeding by presentation
        # alone would put students the model trained on into the demo cohort and
        # quietly undermine the claim that it shows deployment behaviour.
        wanted = set(split.test_presentations)
        cohort = population.loc[split.test]
    print(f"Cohorts: {sorted(wanted)} ({len(cohort):,} checkpoint rows)")
    enrolments = cohort.drop_duplicates(subset=["code_module", "code_presentation", "id_student"])[
        ["code_module", "code_presentation", "id_student", "module_presentation_length"]
    ]
    print(f"Students: {len(enrolments):,}")

    with session_scope(args.database_url) as session:
        # -- reference data ------------------------------------------------
        model_row = session.scalar(
            select(models.ModelVersion).where(models.ModelVersion.version == metadata.model_version)
        )
        if model_row is None:
            model_row = models.ModelVersion(
                version=metadata.model_version,
                model_type=metadata.model_type,
                trained_at=str(metadata.created_at),
                git_commit=metadata.git_commit,
                package_version=metadata.package_version,
                n_features=metadata.n_features,
                train_data_hash=metadata.train_data_hash,
                imbalance_strategy=metadata.imbalance_strategy,
                calibration_method=metadata.calibration_method,
                metrics=metadata.metrics,
                feature_names=metadata.feature_names,
                is_active=True,
            )
            session.add(model_row)
        # Exactly one active version at a time, so "which model is serving" has a
        # single answer.
        session.query(models.ModelVersion).filter(
            models.ModelVersion.version != metadata.model_version
        ).update({"is_active": False})

        threshold_row = session.scalar(
            select(models.ThresholdConfig).where(models.ThresholdConfig.name == thresholds.name)
        )
        if threshold_row is None:
            threshold_row = models.ThresholdConfig(
                version=thresholds.version,
                name=thresholds.name,
                calibrated=thresholds.calibrated,
                alert_budget=config.evaluation.alert_budget,
                bands=[band.model_dump() for band in thresholds.bands],
                is_active=True,
            )
            session.add(threshold_row)
        session.query(models.ThresholdConfig).filter(
            models.ThresholdConfig.name != thresholds.name
        ).update({"is_active": False})
        session.flush()
        print(f"Model version: {model_row.version} | thresholds: {threshold_row.name}")

        # -- users ---------------------------------------------------------
        for username, role, password in DEMO_USERS:
            if session.scalar(select(models.User).where(models.User.username == username)) is None:
                session.add(
                    models.User(
                        username=username, role=role.value, hashed_password=hash_password(password)
                    )
                )
        session.flush()

        # -- students and demographics -------------------------------------
        session.execute(delete(models.Student))
        session.flush()

        info = load_student_info().set_index(["code_module", "code_presentation", "id_student"])
        student_ids: dict[tuple[str, str, int], int] = {}
        student_rows = []
        demographic_rows = []
        for record in enrolments.itertuples(index=False):
            key = (record.code_module, record.code_presentation, int(record.id_student))
            row = models.Student(
                student_code=student_code(*key),
                code_module=key[0],
                code_presentation=key[1],
                source_student_id=key[2],
                data_source="real_oulad",
                module_presentation_length=int(record.module_presentation_length),
            )
            student_rows.append(row)
        session.add_all(student_rows)
        session.flush()
        for row in student_rows:
            student_ids[(row.code_module, row.code_presentation, row.source_student_id)] = row.id

        # Protected attributes go into their own table and nowhere else.
        for key, student_id in student_ids.items():
            if key not in info.index:
                continue
            record = info.loc[key]
            demographic_rows.append(
                models.StudentDemographics(
                    student_id=student_id,
                    gender=_maybe(record.get("gender")),
                    age_band=_maybe(record.get("age_band")),
                    imd_band=_maybe(record.get("imd_band")),
                    disability=_maybe(record.get("disability")),
                    region=_maybe(record.get("region")),
                )
            )
        session.add_all(demographic_rows)
        session.flush()
        print(f"  students {len(student_rows):,} | demographics {len(demographic_rows):,}")

        # -- engagement ----------------------------------------------------
        engagement = load_student_day_engagement()
        engagement = engagement[engagement["code_presentation"].isin(wanted)]
        engagement["student_id"] = [
            student_ids.get((m, p, int(s)))
            for m, p, s in zip(
                engagement["code_module"],
                engagement["code_presentation"],
                engagement["id_student"],
                strict=True,
            )
        ]
        engagement = engagement.dropna(subset=["student_id"])
        payload = engagement[["student_id", "course_day", "clicks", "distinct_resources"]].copy()
        payload["student_id"] = payload["student_id"].astype(int)
        # bulk_insert_mappings rather than ORM objects: ~700k rows through the
        # unit of work would be needlessly slow and memory-hungry.
        session.bulk_insert_mappings(models.EngagementRecord, payload.to_dict(orient="records"))
        print(f"  engagement rows {len(payload):,}")

        # -- assessments ---------------------------------------------------
        assessments = load_assessments()
        submissions = load_student_assessment()
        merged = assessments.merge(submissions, on="id_assessment", how="left")
        merged = merged[merged["code_presentation"].isin(wanted)]
        merged["student_id"] = [
            student_ids.get((m, p, int(s)) if pd.notna(s) else None)
            for m, p, s in zip(
                merged["code_module"],
                merged["code_presentation"],
                merged["id_student"],
                strict=True,
            )
        ]
        merged = merged.dropna(subset=["student_id"])
        lengths = dict(
            zip(
                enrolments["code_presentation"],
                enrolments["module_presentation_length"],
                strict=True,
            )
        )
        assessment_payload = [
            {
                "student_id": int(record.student_id),
                "assessment_id": int(record.id_assessment),
                "assessment_type": record.assessment_type,
                # A NULL due date means a final exam; treat it as the end of the
                # presentation, matching the feature builder.
                "due_day": int(
                    record.date
                    if pd.notna(record.date)
                    else lengths.get(record.code_presentation, 269)
                ),
                "weight": float(record.weight),
                "date_submitted": (
                    int(record.date_submitted) if pd.notna(record.date_submitted) else None
                ),
                "score": float(record.score) if pd.notna(record.score) else None,
                "is_banked": bool(record.is_banked) if pd.notna(record.is_banked) else False,
            }
            for record in merged.itertuples(index=False)
        ]
        session.bulk_insert_mappings(models.AssessmentRecord, assessment_payload)
        print(f"  assessment rows {len(assessment_payload):,}")

    elapsed = time.perf_counter() - started
    print(f"\nSeeded in {elapsed:.1f}s. Next: python scripts/score_cohort.py")
    return 0


def _maybe(value: object) -> str | None:
    return None if value is None or pd.isna(value) else str(value)


if __name__ == "__main__":
    raise SystemExit(main())
