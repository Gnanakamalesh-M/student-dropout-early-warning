# Documentation index

## Start here

| Document | What it covers |
|---|---|
| [TASK_SPEC.md](TASK_SPEC.md) | **Authoritative** definition of what the system predicts. Read first. |
| [DATA_CARD.md](DATA_CARD.md) | Data sources, provenance, real vs synthetic, known limitations |
| [EDA_FINDINGS.md](EDA_FINDINGS.md) | What the data actually shows, including findings that contradict the original plan |
| [FEATURE_DICTIONARY.md](FEATURE_DICTIONARY.md) | Every feature, and the ablation study that decided the allowlist |
| [MODEL_CARD.md](MODEL_CARD.md) | Performance, limitations, and why the primary model is not the most accurate one |
| [API.md](API.md) | Endpoint reference, RBAC matrix, and a security bug the tests caught |
| [DATABASE.md](DATABASE.md) | Schema, ER diagram, and why explanations are not stored |
| [FRONTEND.md](FRONTEND.md) | Dashboard pages, UI decisions, and a 3.7x bundle-size fix |
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

- `MONITORING.md` — drift detection and the retraining policy, measured on the
  real train/test cohort pair
- `DEPLOYMENT.md` — containers, configuration, and an explicit table of what is
  verified where; the container work is verified only in CI
- `FUTURE_WORK.md`
