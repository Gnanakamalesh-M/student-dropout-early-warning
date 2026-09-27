"""Batch scorer: write predictions and derive alerts.

This is the job an institution would run at the end of each assessment window. It
persists predictions rather than recomputing them on read, which is the point of
the table: a trajectory only means something if past scores are the ones that were
actually produced, under the model version recorded against each row. Recomputing
history under a newer model would silently rewrite what staff saw.

Idempotent. ``predictions`` is unique on (student, checkpoint, model version), so
re-running updates in place rather than accumulating duplicates.

Usage::

    python scripts/score_cohort.py
    python scripts/score_cohort.py --checkpoint 90     # one checkpoint only
"""

from __future__ import annotations

import argparse
import time

import pandas as pd
from backend.app.db import models
from backend.app.db.session import get_engine, reset_engine, session_scope, sync_database_url
from sqlalchemy import delete, select

from dropout_ews.config.settings import DATA_DIR, load_feature_config, load_threshold_config
from dropout_ews.data.checkpoints import build_checkpoint_rows
from dropout_ews.models.registry import load_model

FEATURES_PARQUET = DATA_DIR / "processed" / "features.parquet"
DROP_COLUMNS = ("label", "date_unregistration", "split", "row_id")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--checkpoint", type=int, default=None)
    parser.add_argument(
        "--with-labels",
        action="store_true",
        default=True,
        help="store the observed label where the horizon has closed, for monitoring",
    )
    args = parser.parse_args()

    if args.database_url:
        reset_engine()
        get_engine(args.database_url)
    print(f"Database: {sync_database_url(args.database_url)}")

    config = load_feature_config()
    thresholds = load_threshold_config()
    model, metadata = load_model()

    frame = pd.read_parquet(FEATURES_PARQUET)
    population, _ = build_checkpoint_rows()
    frame.index = population.index
    frame["code_presentation"] = population["code_presentation"].to_numpy()

    started = time.perf_counter()
    with session_scope(args.database_url) as session:
        model_row = session.scalar(
            select(models.ModelVersion).where(models.ModelVersion.version == metadata.model_version)
        )
        threshold_row = session.scalar(
            select(models.ThresholdConfig).where(models.ThresholdConfig.is_active.is_(True))
        )
        if model_row is None or threshold_row is None:
            print("Model version or threshold config missing. Run scripts/seed_db.py first.")
            return 1

        # Only students that exist in the database, so scoring a cohort that was
        # never seeded fails visibly rather than silently producing nothing.
        students = {
            (row.code_module, row.code_presentation, row.source_student_id): row.id
            for row in session.scalars(select(models.Student)).all()
        }
        if not students:
            print("No students in the database. Run scripts/seed_db.py first.")
            return 1
        print(f"Students in scope: {len(students):,}")

        keys = list(
            zip(frame["code_module"], frame["code_presentation"], frame["id_student"], strict=True)
        )
        frame["student_db_id"] = [students.get((m, p, int(s))) for m, p, s in keys]
        scope = frame.dropna(subset=["student_db_id"]).copy()
        if args.checkpoint is not None:
            scope = scope[scope["checkpoint_day"] == args.checkpoint]
        if scope.empty:
            print("Nothing to score.")
            return 1

        features = scope.drop(
            columns=[
                c
                for c in (*DROP_COLUMNS, "student_db_id", "code_presentation")
                if c in scope.columns
            ]
        )
        # `code_presentation` is a cohort key the pipeline needs, so it is put back
        # rather than dropped with the outcome columns.
        features["code_presentation"] = scope["code_presentation"].to_numpy()
        probabilities = model.predict_proba(features)[:, 1]

        scope["probability"] = probabilities
        scope["band"] = [thresholds.band_for(p).key for p in probabilities]
        print(f"Scored {len(scope):,} rows in {time.perf_counter() - started:.1f}s")

        # Idempotent replace for this (model version, checkpoint scope).
        statement = delete(models.Prediction).where(
            models.Prediction.model_version_id == model_row.id
        )
        if args.checkpoint is not None:
            statement = statement.where(models.Prediction.checkpoint_day == args.checkpoint)
        session.execute(statement)
        session.flush()

        from backend.app.services import features_hash

        payload = []
        for record in scope.itertuples(index=False):
            payload.append(
                {
                    "student_id": int(record.student_db_id),
                    "model_version_id": model_row.id,
                    "threshold_config_id": threshold_row.id,
                    "checkpoint_day": int(record.checkpoint_day),
                    "horizon_days": config.task.horizon_days,
                    "probability": float(record.probability),
                    "band": record.band,
                    # A per-row hash would be ideal but costs a hash per student;
                    # the batch hash identifies the feature matrix this score came
                    # from, which is what reproducibility needs.
                    "features_hash": "",
                    "label": int(record.label) if args.with_labels else None,
                }
            )
        batch_hash = features_hash(features)
        for item in payload:
            item["features_hash"] = batch_hash
        session.bulk_insert_mappings(models.Prediction, payload)
        session.flush()
        print(f"Wrote {len(payload):,} predictions (features_hash {batch_hash[:12]}...)")

        alerts = _derive_alerts(session, model_row.id, thresholds)
        print(f"Derived {len(alerts):,} alerts")

    print("\nDone.")
    return 0


def _derive_alerts(session, model_version_id: int, thresholds) -> list[dict[str, object]]:
    """Apply the configured alert rules to the stored predictions.

    Alerts are about *change* as much as level: a student steady at 0.30 needs
    less attention than one who moved 0.10 to 0.30. Existing unacknowledged alerts
    for this model version are replaced; acknowledged ones are kept, because an
    acknowledgement is a human action and re-running a batch job must not undo it.
    """
    rules = thresholds.alerts
    escalation_bands = set(rules.band_escalation.into_bands)

    session.execute(
        delete(models.Alert).where(
            models.Alert.acknowledged.is_(False),
            models.Alert.prediction_id.in_(
                select(models.Prediction.id).where(
                    models.Prediction.model_version_id == model_version_id
                )
            ),
        )
    )
    session.flush()

    rows = session.execute(
        select(
            models.Prediction.id,
            models.Prediction.student_id,
            models.Prediction.checkpoint_day,
            models.Prediction.probability,
            models.Prediction.band,
        )
        .where(models.Prediction.model_version_id == model_version_id)
        .order_by(models.Prediction.student_id, models.Prediction.checkpoint_day)
    ).all()

    already = {
        (alert.prediction_id, alert.reason) for alert in session.scalars(select(models.Alert)).all()
    }

    payload: list[dict[str, object]] = []
    previous_student: int | None = None
    previous_band: str | None = None
    previous_probability: float | None = None
    rising_run = 0

    for prediction_id, student_id, checkpoint_day, probability, band in rows:
        if student_id != previous_student:
            previous_student, previous_band, previous_probability, rising_run = (
                student_id,
                None,
                None,
                0,
            )

        delta = probability - previous_probability if previous_probability is not None else None
        rising_run = rising_run + 1 if delta is not None and delta > 0 else 0

        reason: str | None = None
        if (
            rules.band_escalation.enabled
            and band in escalation_bands
            and previous_band is not None
            and previous_band != band
        ):
            reason = "band_escalated"
        elif (
            rules.rapid_increase.enabled
            and delta is not None
            and delta >= rules.rapid_increase.min_delta
        ):
            reason = "rapid_increase"
        elif (
            rules.sustained_increase.enabled
            and rising_run >= rules.sustained_increase.consecutive_checkpoints
        ):
            reason = "sustained_increase"

        if reason and (prediction_id, reason) not in already:
            payload.append(
                {
                    "student_id": student_id,
                    "prediction_id": prediction_id,
                    "checkpoint_day": checkpoint_day,
                    "reason": reason,
                    "severity": "critical" if band == "critical" else "high",
                    "probability": probability,
                    "previous_probability": previous_probability,
                    "band": band,
                    "acknowledged": False,
                }
            )

        previous_band, previous_probability = band, probability

    if payload:
        session.bulk_insert_mappings(models.Alert, payload)
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
