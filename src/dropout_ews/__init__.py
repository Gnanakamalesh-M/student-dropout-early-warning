"""Student Dropout Early-Warning & Intervention System.

This package holds all model-facing logic: data loading, feature engineering,
training, evaluation, explainability, and the intervention rule engine.

It is imported by *both* the offline training pipeline and the FastAPI service
(ADR-0004). Feature logic in particular must never be reimplemented in the
backend — that is the primary source of train/serve skew.
"""

__version__ = "0.1.0"
