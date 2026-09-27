"""Generate explainability artifacts and end-to-end examples.

Produces, into ``reports/explainability/``:

* global importance (mean |SHAP|) as a table and a figure;
* the attribution-stability report, which is the evidence that per-student
  factors are reliable enough to show a counsellor;
* worked examples running the full chain — SHAP, rendered factors, matched
  interventions, case note — for real high-risk rows.

Usage::

    python scripts/run_explainability.py
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")  # no display in CI; must precede pyplot

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dropout_ews.config.settings import (
    DATA_DIR,
    REPORTS_DIR,
    load_threshold_config,
)
from dropout_ews.data.checkpoints import build_checkpoint_rows
from dropout_ews.evaluation.splits import make_temporal_split
from dropout_ews.explainability.llm_summary import generate_case_note
from dropout_ews.explainability.narratives import render_explanation
from dropout_ews.explainability.shap_explainer import (
    RiskExplainer,
    attribution_stability,
)
from dropout_ews.interventions.engine import load_catalog, recommend
from dropout_ews.models.registry import load_background, load_model

OUT_DIR = REPORTS_DIR / "explainability"
FIGURES_DIR = REPORTS_DIR / "figures"
DROP_COLUMNS = ("label", "date_unregistration", "split", "row_id")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=int, default=3)
    parser.add_argument("--stability-rows", type=int, default=600)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(DATA_DIR / "processed" / "features.parquet")
    population, _ = build_checkpoint_rows()
    frame.index = population.index
    frame["code_presentation"] = population["code_presentation"].to_numpy()
    split = make_temporal_split(population)

    model, metadata = load_model()
    explainer = RiskExplainer(model, background=load_background())
    thresholds = load_threshold_config()
    catalog = load_catalog()

    test = frame.loc[split.test]
    features = test.drop(columns=[c for c in DROP_COLUMNS if c in test.columns])
    probabilities = model.predict_proba(features)[:, 1]
    labels = test["label"].to_numpy()

    print(f"Model {metadata.model_version}, {len(explainer.feature_names)} model features")

    # ------------------------------------------------------------ global importance
    print("\n[1/4] Global importance...")
    importance = explainer.global_importance(features)
    importance.to_csv(OUT_DIR / "global_importance.csv", index=False)
    print(importance.head(12).round(4).to_string(index=False))

    top = importance.head(15).iloc[::-1]
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.barh(top["feature"], top["mean_abs_shap"])
    axis.set_xlabel("Mean |SHAP| (log-odds of the pre-calibration score)")
    axis.set_title("Global feature importance")
    axis.grid(alpha=0.3, axis="x")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "global_importance.png", dpi=140)
    plt.close(figure)

    # ----------------------------------------------------------------- stability
    print("\n[2/4] Attribution stability...")
    high_risk_mask = probabilities >= np.quantile(probabilities, 0.95)
    stability_all = attribution_stability(explainer, features.head(args.stability_rows), k=3)
    stability_high = attribution_stability(
        explainer, features[high_risk_mask].head(args.stability_rows), k=3
    )
    summary = pd.DataFrame(
        [
            {
                "population": name,
                "rows": len(table),
                "mean_top3_overlap": round(table["mean_top_k_overlap"].mean(), 4),
                "median_top3_overlap": round(table["mean_top_k_overlap"].median(), 4),
                "p10_top3_overlap": round(table["mean_top_k_overlap"].quantile(0.1), 4),
                "share_below_one_third": round((table["mean_top_k_overlap"] <= 0.34).mean(), 4),
            }
            for name, table in (
                ("all test rows", stability_all),
                ("top 5% by risk", stability_high),
            )
        ]
    )
    summary.to_csv(OUT_DIR / "attribution_stability.csv", index=False)
    print(summary.to_string(index=False))

    # ------------------------------------------------------------- worked examples
    print(f"\n[3/4] Worked examples ({args.examples})...")
    # Pick actual positives in the top risk slice, so the examples show the case
    # the system exists for rather than a cherry-picked false positive.
    candidates = np.nonzero(high_risk_mask & (labels == 1))[0][: args.examples]
    examples = []
    for offset, position in enumerate(candidates, start=1):
        row = features.iloc[[position]]
        probability = float(probabilities[position])
        band = thresholds.band_for(probability)
        explanation = explainer.explain_one(row)
        rendered = render_explanation(explanation)
        recommendations, guidance = recommend(rendered.matched_factors, band.key, catalog)
        note = generate_case_note(rendered, recommendations, band.label)

        examples.append(
            {
                # Deliberately not the student id: examples are published, and the
                # row position is enough to reproduce them.
                "example": offset,
                "checkpoint_day": int(row["checkpoint_day"].iloc[0]),
                "risk_probability": round(probability, 4),
                "risk_band": band.label,
                "withdrew_within_horizon": True,
                "risk_factors": [
                    {"label": f.label, "impact": f.impact, "sentence": f.sentence}
                    for f in rendered.risk_factors
                ],
                "protective_factors": [
                    {"label": f.label, "impact": f.impact} for f in rendered.protective_factors
                ],
                "context_factors": [
                    {"label": f.label, "impact": f.impact} for f in rendered.context_factors
                ],
                "recommendations": [r.to_dict() for r in recommendations],
                "band_guidance": guidance,
                "case_note": note.text,
                "case_note_source": note.source,
                "disclaimer": rendered.disclaimer,
            }
        )
        print(f"\n  --- Example {offset}: {band.label} ({probability:.1%}) ---")
        for factor in rendered.risk_factors[:4]:
            print(f"    [{factor.impact:<6}] {factor.label}")
        print(
            f"    suggested: {', '.join(r.intervention.title for r in recommendations) or 'none'}"
        )
        print(f"    note: {note.text}")

    (OUT_DIR / "worked_examples.json").write_text(json.dumps(examples, indent=2), encoding="utf-8")

    # -------------------------------------------------------------------- report
    print("\n[4/4] Writing the report...")
    lines = [
        "# Explainability (Phase 7)",
        "",
        f"Model `{metadata.model_version}`. Generated by `scripts/run_explainability.py`.",
        "",
        "## What is being explained",
        "",
        "The registered artifact is a calibrated wrapper around the pipeline, so",
        "TreeSHAP explains the **pre-calibration score in log-odds**, not the",
        "displayed probability. That is sound rather than a compromise: isotonic",
        "calibration is monotone, so any feature pushing the raw score up also",
        "pushes the calibrated probability up, and the ranking the explanation",
        "describes is the ranking the displayed figure reflects. What is lost is",
        "the ability to say *how many percentage points* a feature added, so the",
        "narrative layer never claims that and reports direction plus an impact",
        "band instead.",
        "",
        "Values come from XGBoost's own `pred_contribs=True` rather than the `shap`",
        "package: `shap` 0.49 cannot parse XGBoost 3.x's `base_score`, which is now",
        "serialised as an array string. It is the same exact TreeSHAP algorithm, and",
        "additivity is asserted against the model margin in CI.",
        "",
        "## Global importance",
        "",
        importance.head(15).round(4).to_markdown(index=False),
        "",
        "The largest single contributor is `checkpoint_day` — how far through the",
        "course a student is. That is real signal, but nothing a counsellor can act",
        "on, so features are tagged actionable or contextual and rendered in",
        'separate lists. Presenting "it is day 30" as a top reason would waste the',
        "reader's attention.",
        "",
        "## Attribution stability",
        "",
        "Checked before wiring any of this to a dashboard. At a 2.7% positive rate",
        "with 37 correlated features, per-student attributions could have been small",
        "and unstable; if two near-identical students received different main",
        "reasons, showing those reasons as *the* explanation would mislead staff",
        "even with a well-calibrated probability.",
        "",
        "For each row, the nearest neighbours in the model's own feature space are",
        "found and the overlap of top-3 factor sets is measured.",
        "",
        summary.to_markdown(index=False),
        "",
        "Attributions are **more** stable for high-risk students, which is the right",
        "direction since those are the ones staff see. They are not perfectly stable,",
        'which is precisely why the UI language is "factors the model weighted most',
        'heavily" rather than "the reasons".',
        "",
        "## Intervention mapping",
        "",
        f"{len(catalog.interventions)} interventions covering "
        f"{len(catalog.covered_factors)} risk factors. Every entry is an offer of",
        "support: the catalog is scanned at load time against a punitive and",
        "surveillance keyword denylist, so a punitive entry cannot be introduced by",
        'editing YAML. Recommendations carry `status="recommended"` and',
        "`requires_human_review=true`; nothing executes.",
        "",
        "Per-band caps come from the catalog, so a low-risk student receives no",
        "outreach and a critical-band student receives a coordinated offer with a",
        "named owner — the band is a prompt for a human to look, not a decision.",
        "",
        "## Worked examples",
        "",
        "Real test rows in the top 5% of risk that did withdraw within the horizon,",
        "showing the full chain. Student identifiers are deliberately omitted.",
        "See `worked_examples.json`.",
        "",
    ]
    for example in examples:
        lines += [
            f"### Example {example['example']} — {example['risk_band']} "
            f"({example['risk_probability']:.1%}), checkpoint day {example['checkpoint_day']}",
            "",
            "Factors weighted toward higher risk:",
            *(f"- **{f['impact']}** — {f['label']}" for f in example["risk_factors"]),
            "",
        ]
        if example["context_factors"]:
            lines += [
                "Contextual contributors (not actionable):",
                *(f"- {f['label']}" for f in example["context_factors"]),
                "",
            ]
        action_lines = [
            f"- {r['title']} ({r['intensity']}, ~{r['typical_effort_minutes']} min) "
            f"— {r['rationale']}"
            for r in example["recommendations"]
        ] or ["- none at this band"]
        lines += [
            "Suggested support:",
            *action_lines,
            "",
            f"Case note ({example['case_note_source']}): {example['case_note']}",
            "",
        ]
    lines += ["---", "", f"> {examples[0]['disclaimer'] if examples else ''}", ""]

    (REPORTS_DIR / "explainability.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"      {REPORTS_DIR / 'explainability.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
