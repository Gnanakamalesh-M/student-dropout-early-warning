"""Service layer: turns repository rows into API schemas.

Routers stay thin and this holds the logic, so the same behaviour is reachable
from a batch job or a future CLI without going through HTTP.

The one rule worth naming: :func:`build_risk_assessment` is the only place a
probability becomes an API object, and it always attaches the model version, the
calibration flag and the disclaimer. Nothing else may construct a risk figure.
"""

from __future__ import annotations

import hashlib
from typing import Any, cast

import pandas as pd

from backend.app import schemas
from backend.app.repository import (
    ASSESSMENT_FIELDS,
    CONTEXT_FIELDS,
    ENGAGEMENT_FIELDS,
    ParquetRepository,
)
from dropout_ews.config.settings import load_threshold_config
from dropout_ews.explainability.llm_summary import generate_case_note
from dropout_ews.explainability.narratives import FEATURE_META, describe_feature, render_explanation
from dropout_ews.interventions.engine import load_catalog, recommend

# Limitations surfaced through /model/info, so a client cannot present the
# metrics without them. Kept short and pointed at the model card for the rest.
MODEL_LIMITATIONS = [
    "At a 5% alert budget, about 87% of students contacted were not about to "
    "withdraw, and about 78% of those who did withdraw were not contacted.",
    "Reaching 80% recall would require flagging roughly 47% of the cohort, which "
    "is not a usable operating point.",
    "Warning time is weeks rather than months: engagement declines mostly in the "
    "final fortnight before withdrawal.",
    "Trained on one UK distance-learning institution, 2013-2014. Do not apply "
    "elsewhere without re-validation.",
    "Evaluation under-represents students who appear in more than one "
    "presentation, because resolving split overlap removed them from the test set.",
    "Risk bands are an institutional capacity choice, not a validated risk scale.",
    "Students can share an identical probability at a band boundary, so a "
    "boundary is not meaningful separation between adjacent students.",
]


def _band_info(band_key: str) -> schemas.RiskBandInfo:
    config = load_threshold_config()
    for band in config.bands:
        if band.key == band_key:
            return schemas.RiskBandInfo(
                key=band.key,
                label=band.label,
                min_probability=band.min_probability,
                color=band.color,
                action=band.action,
            )
    raise KeyError(band_key)


def build_risk_assessment(
    probability: float, band_key: str, checkpoint_day: int, model_version: str
) -> schemas.RiskAssessment:
    """The single place a probability becomes an API object."""
    return schemas.RiskAssessment(
        probability=round(float(probability), 4),
        band=_band_info(band_key),
        checkpoint_day=int(checkpoint_day),
        model_version=model_version,
        is_calibrated=load_threshold_config().calibrated,
    )


def _numeric(row: pd.Series, fields: tuple[str, ...]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for name in fields:
        if name not in row.index:
            continue
        value = row[name]
        out[name] = None if pd.isna(value) else round(float(value), 4)
    return out


def student_summaries(rows: pd.DataFrame) -> list[schemas.StudentSummary]:
    return [
        schemas.StudentSummary(
            student_code=row["student_code"],
            code_module=row["code_module"],
            code_presentation=row["code_presentation"],
            checkpoint_day=int(row["checkpoint_day"]),
            probability=round(float(row["probability"]), 4),
            band=row["band"],
            risk_direction=row["risk_direction"],
        )
        for _, row in rows.iterrows()
    ]


def student_profile(repository: ParquetRepository, code: str) -> schemas.StudentProfile | None:
    rows = repository.student_rows(code)
    if rows.empty:
        return None
    latest = rows.iloc[-1]
    return schemas.StudentProfile(
        student_code=code,
        code_module=latest["code_module"],
        code_presentation=latest["code_presentation"],
        risk=build_risk_assessment(
            latest["probability"],
            latest["band"],
            int(latest["checkpoint_day"]),
            repository.model_version,
        ),
        risk_direction=latest["risk_direction"],
        trajectory=[
            schemas.TrajectoryPoint(
                checkpoint_day=int(row["checkpoint_day"]),
                probability=round(float(row["probability"]), 4),
                band=row["band"],
            )
            for _, row in rows.iterrows()
        ],
        engagement=_numeric(latest, ENGAGEMENT_FIELDS),
        assessment=_numeric(latest, ASSESSMENT_FIELDS),
        context=_numeric(latest, CONTEXT_FIELDS),
    )


def _factors(items: list[Any]) -> list[schemas.Factor]:
    return [
        schemas.Factor(
            feature=item.feature,
            label=item.label,
            impact=item.impact,
            direction=item.direction,
            actionable=item.actionable,
            sentence=item.sentence,
        )
        for item in items
    ]


def explanation(
    repository: ParquetRepository,
    explainer: Any,
    code: str,
    checkpoint_day: int | None = None,
) -> schemas.ExplanationResponse | None:
    features = repository.feature_row(code, checkpoint_day)
    if features.empty:
        return None
    rows = repository.student_rows(code)
    row = (
        rows.iloc[-1]
        if checkpoint_day is None
        else rows[rows["checkpoint_day"] == checkpoint_day].iloc[0]
    )

    rendered = render_explanation(explainer.explain_one(features))
    return schemas.ExplanationResponse(
        student_code=code,
        checkpoint_day=int(row["checkpoint_day"]),
        risk=build_risk_assessment(
            row["probability"], row["band"], int(row["checkpoint_day"]), repository.model_version
        ),
        risk_factors=_factors(rendered.risk_factors),
        protective_factors=_factors(rendered.protective_factors),
        context_factors=_factors(rendered.context_factors),
        notes=rendered.notes,
    )


def recommendations(
    repository: ParquetRepository, explainer: Any, code: str
) -> schemas.RecommendationsResponse | None:
    features = repository.feature_row(code)
    if features.empty:
        return None
    row = repository.student_rows(code).iloc[-1]
    rendered = render_explanation(explainer.explain_one(features))
    items, guidance = recommend(rendered.matched_factors, str(row["band"]), load_catalog())
    note = generate_case_note(rendered, items, _band_info(str(row["band"])).label)

    return schemas.RecommendationsResponse(
        student_code=code,
        band=row["band"],
        band_guidance=guidance,
        recommendations=[
            # `to_dict` is a plain mapping at the engine boundary; the schema
            # re-validates every field, so the cast only quiets the checker.
            schemas.RecommendationOut.model_validate(item.to_dict())
            for item in items
        ],
        case_note=note.text,
        case_note_source=note.source,  # type: ignore[arg-type]
    )


def features_hash(features: pd.DataFrame) -> str:
    """Hash of the exact vector scored, so a prediction ties to its inputs."""
    digest = hashlib.sha256()
    digest.update(",".join(map(str, features.columns)).encode())
    digest.update(pd.util.hash_pandas_object(features, index=False).to_numpy().tobytes())
    return digest.hexdigest()[:32]


def dashboard_statistics(repository: ParquetRepository) -> schemas.DashboardStatistics:
    latest = repository.latest_rows()
    total = len(latest)
    counts = repository.band_counts()
    return schemas.DashboardStatistics(
        total_students=total,
        scored_checkpoints=len(repository.cohort),
        band_counts=[
            schemas.BandCount(
                band=cast(schemas.BandKey, key),
                label=_band_info(key).label,
                count=count,
                share=round(count / total, 4) if total else 0.0,
            )
            for key, count in counts
        ],
        worsening_count=repository.worsening_count(),
        open_alerts=len(repository.alerts(acknowledged=False)),
        model_version=repository.model_version,
        is_calibrated=repository.is_calibrated,
        alert_budget=repository.alert_budget,
    )


def risk_distribution(
    repository: ParquetRepository, bins: int = 20
) -> list[schemas.DistributionBin]:
    probabilities = repository.latest_rows()["probability"]
    counts, edges = pd.cut(probabilities, bins=bins, retbins=True)
    histogram = counts.value_counts().sort_index()
    return [
        schemas.DistributionBin(
            lower=round(float(edges[index]), 4),
            upper=round(float(edges[index + 1]), 4),
            count=int(value),
        )
        for index, value in enumerate(histogram.to_numpy())
    ]


def scatter(
    repository: ParquetRepository, x_field: str, limit: int = 2000
) -> list[schemas.ScatterPoint]:
    """Scatter one feature against risk.

    Validated against an **allowlist**, not against column presence. Checking
    presence was a real vulnerability: the cohort frame carries
    ``date_unregistration`` alongside the features, so
    ``?x=date_unregistration`` returned HTTP 200 and served the withdrawal event
    time — the label itself — to any authenticated user including an analyst who
    is otherwise denied all individual data. An API test caught it.
    """
    allowed = set(known_scatter_fields())
    if x_field not in allowed:
        raise KeyError(x_field)

    rows = repository.latest_rows()
    if x_field not in rows.columns:
        raise KeyError(x_field)
    subset = rows[[x_field, "probability", "band"]].dropna().head(limit)
    return [
        schemas.ScatterPoint(
            x=round(float(row[x_field]), 4), y=round(float(row["probability"]), 4), band=row["band"]
        )
        for _, row in subset.iterrows()
    ]


def feature_importance(
    repository: ParquetRepository, explainer: Any, limit: int = 20
) -> list[schemas.FeatureImportance]:
    sample = repository.latest_rows().head(1500)
    features = sample.drop(
        columns=[
            c
            for c in (
                "probability",
                "band",
                "student_code",
                "risk_direction",
                "probability_delta",
                "label",
                "date_unregistration",
                "split",
                "row_id",
            )
            if c in sample.columns
        ]
    )
    importance = explainer.global_importance(features).head(limit)
    return [
        schemas.FeatureImportance(
            feature=row["feature"],
            label=describe_feature(row["feature"]).label,
            mean_abs_shap=round(float(row["mean_abs_shap"]), 5),
        )
        for _, row in importance.iterrows()
    ]


def model_info(repository: ParquetRepository, metadata: Any) -> schemas.ModelInfo:
    config = load_threshold_config()
    return schemas.ModelInfo(
        model_version=metadata.model_version,
        model_type=metadata.model_type,
        created_at=str(metadata.created_at),
        git_commit=metadata.git_commit,
        n_features=metadata.n_features,
        imbalance_strategy=metadata.imbalance_strategy,
        calibration_method=metadata.calibration_method,
        train_presentations=metadata.train_presentations,
        test_presentations=metadata.test_presentations,
        metrics=metadata.metrics,
        bands=[_band_info(band.key) for band in config.bands],
        is_calibrated=config.calibrated,
        limitations=MODEL_LIMITATIONS,
    )


def known_scatter_fields() -> list[str]:
    """Fields the analytics scatter endpoint accepts.

    Intersected with ``FEATURE_META``, so a field is only exposed if it is both a
    profile field and a modelled feature with a human label. Anything else —
    including outcome columns that travel alongside the features — is rejected.
    """
    profile_fields = (*ENGAGEMENT_FIELDS, *ASSESSMENT_FIELDS, *CONTEXT_FIELDS)
    return [name for name in profile_fields if name in FEATURE_META]
