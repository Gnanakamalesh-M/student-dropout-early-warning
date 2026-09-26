"""Train/validation/test splitting — ADR-0003 control #3.

Two dependencies in this data make a random split invalid, and either alone is
enough to inflate scores substantially:

* **Group dependence.** One student contributes up to six checkpoint rows with
  heavily autocorrelated features. A random split puts some of a student's rows
  in train and others in test, letting the model memorise students rather than
  learn patterns.
* **Temporal dependence.** Deployment means training on past cohorts and
  scoring a future one. Evaluating on a randomly mixed set measures something
  easier than the real task.

So the outer split is **chronological by presentation** and the inner
cross-validation is **grouped on student**.

Chronology alone is not sufficient, though. ``id_student`` is reused across
modules *and across presentations*: **1,954 of 24,710 students (7.9%) appear in
more than one presentation**, having retaken a module or studied several.
Measured on the real population, a purely chronological split leaves 664
students in both train and validation and 758 in both train and test.

The split therefore **removes overlapping students from the later split**.
Training keeps them; validation and test give them up. That direction is
deliberate: contaminating the evaluation sets is the error that inflates
reported metrics, whereas a slightly smaller training set only costs a little
signal.

The cost is recorded here rather than buried. This drops 9.6% of validation
rows and 13.2% of test rows, and shifts the test positive rate from 3.18% to
2.90%. It also means **the evaluation sets under-represent repeat students**,
who plausibly differ in withdrawal risk. That is a real limitation of the
evaluation and belongs in the model card.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from dropout_ews.data.loaders import presentation_sort_key

GROUP_COLUMN = "id_student"
TIME_COLUMN = "code_presentation"


@dataclass(frozen=True)
class TemporalSplit:
    """A chronological train/validation/test partition of the population."""

    train: pd.Index
    validation: pd.Index
    test: pd.Index
    train_presentations: list[str]
    validation_presentations: list[str]
    test_presentations: list[str]
    dropped_overlap: pd.Index = field(default_factory=lambda: pd.Index([]))
    """Rows removed from validation or test because the student also appears in
    an earlier split. Retained so the exclusion is auditable, not invisible."""

    def summary(self, population: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for name, index, presentations in (
            ("train", self.train, self.train_presentations),
            ("validation", self.validation, self.validation_presentations),
            ("test", self.test, self.test_presentations),
        ):
            subset = population.loc[index]
            rows.append(
                {
                    "split": name,
                    "presentations": ",".join(presentations),
                    "rows": len(subset),
                    "students": subset[GROUP_COLUMN].nunique(),
                    "positives": int(subset["label"].sum()),
                    "positive_rate": round(float(subset["label"].mean()), 4),
                }
            )
        rows.append(
            {
                "split": "dropped_overlap",
                "presentations": "",
                "rows": len(self.dropped_overlap),
                "students": (
                    population.loc[self.dropped_overlap, GROUP_COLUMN].nunique()
                    if len(self.dropped_overlap)
                    else 0
                ),
                "positives": (
                    int(population.loc[self.dropped_overlap, "label"].sum())
                    if len(self.dropped_overlap)
                    else 0
                ),
                "positive_rate": float("nan"),
            }
        )
        return pd.DataFrame(rows)


def make_temporal_split(
    population: pd.DataFrame,
    n_validation_presentations: int = 1,
    n_test_presentations: int = 1,
    resolve_overlap: bool = True,
) -> TemporalSplit:
    """Split chronologically by presentation code, then remove student overlap.

    The last presentation becomes the test set, the one before it the validation
    set, and everything earlier the training set, so training data precedes
    validation and test data in wall-clock time.

    Args:
        resolve_overlap: when ``True`` (the default) students appearing in an
            earlier split are removed from validation and test. When ``False``
            the overlap raises instead, which is how the tests prove the
            condition is detected rather than silently tolerated.

    Raises:
        ValueError: if there are too few presentations, or if overlap survives
            resolution, which would indicate a bug in the resolution itself.
    """
    presentations = sorted(population[TIME_COLUMN].unique(), key=presentation_sort_key)
    needed = n_validation_presentations + n_test_presentations + 1
    if len(presentations) < needed:
        raise ValueError(
            f"need at least {needed} presentations to split, found {len(presentations)}"
        )

    test_presentations = presentations[-n_test_presentations:]
    validation_presentations = presentations[
        -(n_test_presentations + n_validation_presentations) : -n_test_presentations
    ]
    train_presentations = presentations[: -(n_test_presentations + n_validation_presentations)]

    def _index(codes: list[str]) -> pd.Index:
        return population.index[population[TIME_COLUMN].isin(codes)]

    train_index = _index(train_presentations)
    validation_index = _index(validation_presentations)
    test_index = _index(test_presentations)
    dropped_parts: list[np.ndarray] = []

    if resolve_overlap:
        train_students = set(population.loc[train_index, GROUP_COLUMN])
        validation_students = set(population.loc[validation_index, GROUP_COLUMN])

        in_train = population.loc[validation_index, GROUP_COLUMN].isin(train_students)
        dropped_parts.append(validation_index[in_train.to_numpy()].to_numpy())
        validation_index = validation_index[~in_train.to_numpy()]

        # Test must be clean of both earlier splits.
        earlier = train_students | validation_students
        in_earlier = population.loc[test_index, GROUP_COLUMN].isin(earlier)
        dropped_parts.append(test_index[in_earlier.to_numpy()].to_numpy())
        test_index = test_index[~in_earlier.to_numpy()]

    dropped = (
        pd.Index(np.concatenate(dropped_parts))
        if dropped_parts and sum(len(part) for part in dropped_parts)
        else pd.Index([])
    )

    split = TemporalSplit(
        train=train_index,
        validation=validation_index,
        test=test_index,
        train_presentations=train_presentations,
        validation_presentations=validation_presentations,
        test_presentations=test_presentations,
        dropped_overlap=dropped,
    )
    _assert_no_student_overlap(population, split)
    return split


def _assert_no_student_overlap(population: pd.DataFrame, split: TemporalSplit) -> None:
    groups = {
        "train": set(population.loc[split.train, GROUP_COLUMN]),
        "validation": set(population.loc[split.validation, GROUP_COLUMN]),
        "test": set(population.loc[split.test, GROUP_COLUMN]),
    }
    for left, right in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        overlap = groups[left] & groups[right]
        if overlap:
            raise ValueError(
                f"{len(overlap)} students appear in both {left} and {right}; "
                "a chronological split by presentation alone is not sufficient for "
                "this population, because 7.9% of students span presentations. "
                "Pass resolve_overlap=True to drop them from the later split."
            )


def grouped_cv_splits(
    population: pd.DataFrame,
    index: pd.Index,
    n_splits: int = 5,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Grouped, stratified CV folds within a split.

    Grouping on student is mandatory. Stratification keeps the rare positive
    class present in every fold — at a 3% positive rate an unstratified fold can
    end up with very few positives, making fold scores unstable.

    Returns positional index pairs into ``index``, not labels, so they can be
    passed straight to scikit-learn.
    """
    subset = population.loc[index]
    if subset["label"].nunique() < 2:
        raise ValueError("cannot stratify a split containing a single class")

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(
        splitter.split(
            np.zeros(len(subset)),
            subset["label"].to_numpy(),
            groups=subset[GROUP_COLUMN].to_numpy(),
        )
    )


def assert_no_group_leakage_in_folds(
    population: pd.DataFrame,
    index: pd.Index,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> None:
    """Verify no student straddles a fold boundary.

    Called by the training pipeline as well as by tests: the cost is trivial
    against the cost of a silently inflated CV score.
    """
    students = population.loc[index, GROUP_COLUMN].to_numpy()
    for fold, (train_positions, test_positions) in enumerate(folds):
        overlap = set(students[train_positions]) & set(students[test_positions])
        if overlap:
            raise ValueError(f"fold {fold} leaks {len(overlap)} students across the boundary")
