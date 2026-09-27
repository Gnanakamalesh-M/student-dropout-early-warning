"""Model artifact storage with provenance metadata.

A serialised model with no record of what produced it is not reproducible, and
"which model generated this student's score" becomes unanswerable the moment a
second version exists. Every artifact therefore ships with ``metadata.json``
recording metrics, hyperparameters, the feature list, a hash of the training
data, the package version, and the git commit.

The data hash is over the training feature matrix rather than the raw files. It
answers the question that actually matters — "was this model trained on the same
inputs I have now" — and catches a silent change in feature engineering that a
file checksum would miss.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field

# datetime.UTC is 3.11+; the project is pinned to 3.10 (ADR-0004).
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from dropout_ews.config.settings import MODELS_DIR

ARTIFACT_FILENAME = "model.joblib"
METADATA_FILENAME = "metadata.json"
BACKGROUND_FILENAME = "shap_background.parquet"


def git_commit() -> str | None:
    """Current git commit, or ``None`` outside a repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None if result.returncode == 0 else None


def frame_hash(frame: pd.DataFrame) -> str:
    """Stable SHA-256 over a frame's values and column names.

    Column names are included because the same numbers under different names is
    a different model input. Rows are hashed in their given order, so the caller
    must pass a deterministically ordered frame.
    """
    digest = hashlib.sha256()
    digest.update(",".join(map(str, frame.columns)).encode("utf-8"))
    values = pd.util.hash_pandas_object(frame, index=False).to_numpy()
    digest.update(values.tobytes())
    return digest.hexdigest()


@dataclass
class ModelMetadata:
    """Everything needed to identify and audit one artifact."""

    model_version: str
    model_type: str
    created_at: str
    package_version: str
    git_commit: str | None

    feature_names: list[str]
    n_features: int
    hyperparameters: dict[str, Any]
    imbalance_strategy: str
    calibration_method: str

    train_rows: int
    train_positives: int
    train_data_hash: str
    train_presentations: list[str]
    validation_presentations: list[str]
    test_presentations: list[str]

    metrics: dict[str, Any] = field(default_factory=dict)
    band_config: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=False, default=str)


def new_version_id(model_type: str) -> str:
    """A sortable version id: ``<type>-<UTC timestamp>``."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{model_type}-{stamp}"


def save_model(
    model: Any,
    metadata: ModelMetadata,
    background: pd.DataFrame | None = None,
    models_dir: Path | None = None,
) -> Path:
    """Write the artifact, metadata, and optional SHAP background sample.

    The background sample is versioned with the model deliberately: SHAP values
    depend on it, so explanations are only reproducible if the same background
    is used, and regenerating it separately would silently change past
    explanations.
    """
    directory = (models_dir or MODELS_DIR) / metadata.model_version
    directory.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, directory / ARTIFACT_FILENAME)
    (directory / METADATA_FILENAME).write_text(metadata.to_json(), encoding="utf-8")
    if background is not None:
        background.to_parquet(directory / BACKGROUND_FILENAME, index=False)

    # A pointer to the newest artifact, so the API does not have to guess.
    (directory.parent / "LATEST").write_text(metadata.model_version, encoding="utf-8")
    return directory


def load_model(
    model_version: str | None = None, models_dir: Path | None = None
) -> tuple[Any, ModelMetadata]:
    """Load an artifact and its metadata.

    ``model_version=None`` resolves the ``LATEST`` pointer.
    """
    root = models_dir or MODELS_DIR
    if model_version is None:
        pointer = root / "LATEST"
        if not pointer.is_file():
            raise FileNotFoundError(
                f"no LATEST pointer in {root}. Train a model first: python scripts/train_model.py"
            )
        model_version = pointer.read_text(encoding="utf-8").strip()

    directory = root / model_version
    artifact = directory / ARTIFACT_FILENAME
    if not artifact.is_file():
        raise FileNotFoundError(f"no model artifact at {artifact}")

    model = joblib.load(artifact)
    metadata = ModelMetadata(
        **json.loads((directory / METADATA_FILENAME).read_text(encoding="utf-8"))
    )
    return model, metadata


def load_background(
    model_version: str | None = None, models_dir: Path | None = None
) -> pd.DataFrame | None:
    """Load the versioned SHAP background sample, if one was saved."""
    root = models_dir or MODELS_DIR
    if model_version is None:
        pointer = root / "LATEST"
        if not pointer.is_file():
            return None
        model_version = pointer.read_text(encoding="utf-8").strip()
    path = root / model_version / BACKGROUND_FILENAME
    return pd.read_parquet(path) if path.is_file() else None


def list_versions(models_dir: Path | None = None) -> list[str]:
    """Available artifact versions, newest first."""
    root = models_dir or MODELS_DIR
    if not root.is_dir():
        return []
    return sorted(
        (
            directory.name
            for directory in root.iterdir()
            if directory.is_dir() and (directory / ARTIFACT_FILENAME).is_file()
        ),
        reverse=True,
    )
