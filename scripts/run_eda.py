"""Run the full EDA and write tables and figures to ``reports/``.

Regenerates everything quoted in docs/EDA_FINDINGS.md, so the findings stay
reproducible rather than becoming a snapshot nobody can re-derive.

Usage::

    python scripts/run_eda.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # no display in CI; must precede pyplot import

import matplotlib.pyplot as plt
import pandas as pd

from dropout_ews.config.settings import REPORTS_DIR
from dropout_ews.data.checkpoints import build_checkpoint_rows
from dropout_ews.eda import analysis as eda

FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "eda"


def _save(frame: pd.DataFrame, name: str, index: bool = True) -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(TABLES_DIR / f"{name}.csv", index=index)
    print(f"  {name}.csv ({len(frame)} rows)")


def main() -> int:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Building population and engagement windows...")
    population, report = build_checkpoint_rows()
    windows = eda.engagement_windows(population)
    print(f"  {len(windows):,} rows, {windows['label'].mean():.2%} positive")

    print("\nWriting tables...")
    _save(eda.zero_engagement_rate(windows), "zero_engagement_rate")
    _save(eda.engagement_by_label(windows), "engagement_by_label")
    _save(eda.outcome_by_module(), "withdrawal_rate_by_module")
    _save(eda.positive_rate_by_group(windows, "code_module"), "positive_rate_by_module")
    _save(
        eda.positive_rate_by_group(windows, "code_presentation"),
        "positive_rate_by_presentation",
    )
    _save(eda.discriminative_power(windows), "discriminative_power", index=False)
    _save(eda.trivial_rule_baselines(windows), "trivial_rule_baselines", index=False)
    _save(eda.subgroup_base_rates(windows), "subgroup_base_rates", index=False)

    aligned = eda.aligned_engagement_before_event(population, weeks_before=12)
    _save(aligned, "aligned_engagement_before_event")

    for feature in ("clicks_7d", "days_since_last_activity", "clicks_delta_30d"):
        _save(
            eda.discriminative_power_by_checkpoint(windows, feature),
            f"auc_by_checkpoint_{feature}",
            index=False,
        )

    print("\nWriting figures...")

    # Figure 1 — the premise test: does disengagement precede withdrawal?
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(aligned.index, aligned["median_clicks"], marker="o")
    axis.invert_xaxis()
    axis.set_xlabel("Weeks before withdrawal")
    axis.set_ylabel("Median weekly clicks")
    axis.set_title("Engagement aligned to the withdrawal event")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "engagement_before_withdrawal.png", dpi=140)
    plt.close(figure)
    print("  engagement_before_withdrawal.png")

    # Figure 2 — positive rate per checkpoint.
    figure, axis = plt.subplots(figsize=(7, 4))
    checkpoints = sorted(report.rows_by_checkpoint)
    rates = [report.positives_by_checkpoint[t] / report.rows_by_checkpoint[t] for t in checkpoints]
    axis.bar([str(t) for t in checkpoints], rates)
    axis.set_xlabel("Checkpoint (course day)")
    axis.set_ylabel("Positive rate")
    axis.set_title("Positive rate by checkpoint (H=30)")
    axis.grid(alpha=0.3, axis="y")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "positive_rate_by_checkpoint.png", dpi=140)
    plt.close(figure)
    print("  positive_rate_by_checkpoint.png")

    # Figure 3 — module variation, which motivates cohort-relative features.
    by_module = eda.positive_rate_by_group(windows, "code_module")
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(by_module.index.astype(str), by_module["rate"])
    axis.set_xlabel("Module")
    axis.set_ylabel("Positive rate")
    axis.set_title("Positive rate varies ~4x across modules")
    axis.grid(alpha=0.3, axis="y")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "positive_rate_by_module.png", dpi=140)
    plt.close(figure)
    print("  positive_rate_by_module.png")

    print(f"\nTables: {TABLES_DIR}\nFigures: {FIGURES_DIR}")
    print("Interpretation: docs/EDA_FINDINGS.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
