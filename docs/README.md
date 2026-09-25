# Documentation index

## Start here

| Document | What it covers |
|---|---|
| [TASK_SPEC.md](TASK_SPEC.md) | **Authoritative** definition of what the system predicts. Read first. |
| [DATA_CARD.md](DATA_CARD.md) | Data sources, provenance, real vs synthetic, known limitations |
| [ETHICS.md](ETHICS.md) | Privacy, fairness, governance, and what this system must never be used for |

## Architecture decision records

Each ADR records one decision, its rationale, the alternatives rejected, and
the consequences accepted. They are append-only: a decision that changes gets a
new ADR that supersedes the old one, rather than an edit.

| ADR | Decision |
|---|---|
| [0001-prediction-target-and-horizon.md](adr/0001-prediction-target-and-horizon.md) | Point-in-time framing: P(withdraw within 30 days) as of day `t` |
| [0002-dataset-strategy.md](adr/0002-dataset-strategy.md) | OULAD primary, documented synthetic supplementary, UCI demoted |
| [0003-leakage-controls-and-splitting.md](adr/0003-leakage-controls-and-splitting.md) | The as-of property test, allowlists, temporal + grouped splits |
| [0004-architecture-and-stack.md](adr/0004-architecture-and-stack.md) | Shared feature code, persisted predictions, React over Streamlit, Python 3.10 |
| [0005-explainability-and-llm-scope.md](adr/0005-explainability-and-llm-scope.md) | SHAP, mandated non-causal language, narrow and optional LLM use |

## Written in later phases

- `EDA_FINDINGS.md` (Phase 3) — including the horizon decision from ADR-0001
- `MODEL_CARD.md` (Phase 6) — metrics, operating point, limitations
- `DATABASE.md` (Phase 9) — schema and ER diagram
- `API.md` (Phase 8) — endpoint reference
- `MONITORING.md` (Phase 11) — drift detection and the retraining policy
- `DEPLOYMENT.md` (Phase 13)
- `FUTURE_WORK.md`
