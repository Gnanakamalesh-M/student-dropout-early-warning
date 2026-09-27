/**
 * MSW handlers.
 *
 * Fixtures deliberately use the **real** shapes and realistic values — a 2.9% base
 * rate, single-digit risk percentages, the actual disclaimer wording. Mocks with
 * round numbers hide formatting bugs: a fixture at 0.5 would never catch a
 * component that renders `0.089` as `0%`.
 */

import { http, HttpResponse } from "msw";
import { setupServer } from "msw/node";
import type {
  Alert,
  CurrentUser,
  DashboardStatistics,
  DistributionBin,
  ExplanationResponse,
  FeatureImportance,
  InterventionRecord,
  ModelInfo,
  Page,
  RecommendationsResponse,
  RiskAssessment,
  StudentProfile,
  StudentSummary,
} from "../api/types";

export const DISCLAIMER =
  "These are the factors the model weighted most heavily for this student. " +
  "They describe the model's behaviour, not the causes of withdrawal, and they " +
  "are not a judgement about this student. The risk figure is a probability, " +
  "not a prediction of what will happen.";

export const MODEL_VERSION = "xgboost-20260927T073831Z";

const risk = (probability: number, band: StudentSummary["band"]): RiskAssessment => ({
  probability,
  band: {
    key: band,
    label: band.charAt(0).toUpperCase() + band.slice(1),
    min_probability: band === "critical" ? 0.09385 : band === "high" ? 0.05954 : 0.02984,
    color: "#B71C1C",
    action: "Priority human review and coordinated support offer.",
  },
  checkpoint_day: 90,
  model_version: MODEL_VERSION,
  is_calibrated: true,
  disclaimer: DISCLAIMER,
});

export const USER: CurrentUser = {
  username: "counsellor",
  role: "counsellor",
  may_view_individuals: true,
  may_assign_interventions: true,
};

export const ANALYST: CurrentUser = {
  username: "analyst",
  role: "analyst",
  may_view_individuals: false,
  may_assign_interventions: false,
};

export const DASHBOARD: DashboardStatistics = {
  total_students: 7848,
  scored_checkpoints: 43407,
  band_counts: [
    { band: "low", label: "Low", count: 4986, share: 0.6353 },
    { band: "medium", label: "Medium", count: 1561, share: 0.1989 },
    { band: "high", label: "High", count: 693, share: 0.0883 },
    { band: "critical", label: "Critical", count: 608, share: 0.0775 },
  ],
  worsening_count: 1204,
  open_alerts: 3027,
  model_version: MODEL_VERSION,
  is_calibrated: true,
  alert_budget: 0.05,
};

export const STUDENTS: StudentSummary[] = [
  {
    student_code: "S-AAA000000001",
    code_module: "CCC",
    code_presentation: "2014J",
    checkpoint_day: 90,
    probability: 0.1461,
    band: "critical",
    risk_direction: "worsening",
  },
  {
    student_code: "S-AAA000000002",
    code_module: "DDD",
    code_presentation: "2014J",
    checkpoint_day: 90,
    probability: 0.0612,
    band: "high",
    risk_direction: "improving",
  },
  {
    student_code: "S-AAA000000003",
    code_module: "GGG",
    code_presentation: "2014J",
    checkpoint_day: 30,
    probability: 0.0089,
    band: "low",
    risk_direction: "unknown",
  },
];

export const PROFILE: StudentProfile = {
  student_code: "S-AAA000000001",
  code_module: "CCC",
  code_presentation: "2014J",
  risk: risk(0.1461, "critical"),
  risk_direction: "worsening",
  trajectory: [
    { checkpoint_day: 30, probability: 0.0231, band: "low" },
    { checkpoint_day: 60, probability: 0.0688, band: "high" },
    { checkpoint_day: 90, probability: 0.1461, band: "critical" },
  ],
  engagement: {
    clicks_7d: 0,
    clicks_28d: 14,
    active_days_28d: 2,
    days_since_last_activity: 19,
    clicks_vs_baseline_ratio: 0.08,
  },
  assessment: {
    assessments_due: 4,
    assessments_submitted: 1,
    assessments_missed: 3,
    submission_rate: 0.25,
    mean_score: 41.5,
  },
  context: { checkpoint_day: 90, studied_credits: 120, num_of_prev_attempts: 1 },
};

export const EXPLANATION: ExplanationResponse = {
  student_code: "S-AAA000000001",
  checkpoint_day: 90,
  risk: risk(0.1461, "critical"),
  risk_factors: [
    {
      feature: "submission_rate",
      label: "Share of due assessments submitted",
      impact: "High",
      direction: "increases",
      actionable: true,
      sentence:
        "Share of due assessments submitted: the model weighted this heavily and it contributed to a higher estimate for this student.",
    },
    {
      feature: "days_since_last_activity",
      label: "Time since last course activity",
      impact: "Medium",
      direction: "increases",
      actionable: true,
      sentence:
        "Time since last course activity: the model gave this moderate weight and it contributed to a higher estimate for this student.",
    },
  ],
  protective_factors: [
    {
      feature: "mean_score",
      label: "Average assessment score",
      impact: "Low",
      direction: "decreases",
      actionable: true,
      sentence:
        "Average assessment score: the model gave this some weight and it contributed to a lower estimate for this student.",
    },
  ],
  context_factors: [
    {
      feature: "checkpoint_day",
      label: "Point reached in the course",
      impact: "High",
      direction: "increases",
      actionable: false,
      sentence:
        "Point reached in the course: the model weighted this heavily and it contributed to a higher estimate for this student.",
    },
  ],
  notes: [],
  disclaimer: DISCLAIMER,
  attribution_basis:
    "Attributions describe the model's pre-calibration score. They are reported as direction and relative impact, not as an effect on the probability.",
};

export const RECOMMENDATIONS: RecommendationsResponse = {
  student_code: "S-AAA000000001",
  band: "critical",
  band_guidance:
    "Priority human review. A coordinated offer of support, with a named owner. The band is a prompt to look, not a decision that has been made.",
  recommendations: [
    {
      key: "assessment_planning_support",
      title: "Assessment planning support",
      description:
        "Help the student map out upcoming deadlines and agree a realistic plan for catching up on anything outstanding.",
      intensity: "light",
      owner_role: "counsellor",
      typical_effort_minutes: 30,
      matched_factor: "missed_assessments",
      rationale:
        "Suggested because the model weighted assessments due but not submitted for this student.",
      status: "recommended",
      requires_human_review: true,
    },
  ],
  case_note:
    "This student sits in the Critical band. The model gave most weight to share of due assessments submitted. Suggested next step: assessment planning support.",
  case_note_source: "template",
  disclaimer: DISCLAIMER,
};

export const ALERTS: Alert[] = [
  {
    id: 1,
    student_code: "S-AAA000000001",
    checkpoint_day: 90,
    reason: "band_escalated",
    severity: "critical",
    probability: 0.1461,
    previous_probability: 0.0688,
    band: "critical",
    acknowledged: false,
    created_at: "2026-09-27T08:00:00Z",
  },
  {
    id: 2,
    student_code: "S-AAA000000002",
    checkpoint_day: 60,
    reason: "rapid_increase",
    severity: "high",
    probability: 0.0612,
    previous_probability: null,
    band: "high",
    acknowledged: false,
    created_at: "2026-09-27T08:05:00Z",
  },
];

export const MODEL_INFO: ModelInfo = {
  model_version: MODEL_VERSION,
  model_type: "xgboost",
  created_at: "2026-09-27T07:38:31Z",
  git_commit: "29ed48b",
  n_features: 37,
  imbalance_strategy: "none",
  calibration_method: "isotonic (prefit, fitted on validation)",
  train_presentations: ["2013B", "2013J"],
  test_presentations: ["2014J"],
  metrics: {
    test_pr_auc: 0.0885,
    test_roc_auc: 0.7468,
    test_brier: 0.02727,
    test_base_rate: 0.029,
    cv_pr_auc: 0.1046,
  },
  bands: [
    { key: "low", label: "Low", min_probability: 0, color: "#2E7D32", action: "No outreach." },
    {
      key: "medium",
      label: "Medium",
      min_probability: 0.02984,
      color: "#F9A825",
      action: "Passive monitoring.",
    },
    {
      key: "high",
      label: "High",
      min_probability: 0.05954,
      color: "#EF6C00",
      action: "Human review, then advisor outreach if the reviewer agrees.",
    },
    {
      key: "critical",
      label: "Critical",
      min_probability: 0.09385,
      color: "#C62828",
      action: "Priority human review and coordinated support offer.",
    },
  ],
  is_calibrated: true,
  limitations: [
    "At a 5% alert budget, about 87% of students contacted were not about to withdraw, and about 78% of those who did withdraw were not contacted.",
    "Reaching 80% recall would require flagging roughly 47% of the cohort, which is not a usable operating point.",
    "Warning time is weeks rather than months.",
    "Trained on one UK distance-learning institution, 2013-2014.",
    "Risk bands are an institutional capacity choice, not a validated risk scale.",
  ],
};

const DISTRIBUTION: DistributionBin[] = Array.from({ length: 10 }, (_, index) => ({
  lower: index * 0.033,
  upper: (index + 1) * 0.033,
  count: Math.max(1, Math.round(5000 / (index + 1) ** 2)),
}));

const IMPORTANCE: FeatureImportance[] = [
  { feature: "checkpoint_day", label: "Point reached in the course", mean_abs_shap: 0.3479 },
  { feature: "submission_rate", label: "Share of due assessments submitted", mean_abs_shap: 0.2193 },
  { feature: "mean_score", label: "Average assessment score", mean_abs_shap: 0.1114 },
];

const INTERVENTIONS: InterventionRecord[] = [];

/** Mutable copy, reset between tests by resetAlertState. */
let alertState: Alert[] = ALERTS.map((alert) => ({ ...alert }));

export function resetAlertState(): void {
  alertState = ALERTS.map((alert) => ({ ...alert }));
}

const page = <T,>(rows: T[]): Page<T> => ({
  data: rows,
  meta: { total: rows.length, page: 1, page_size: 25, model_version: MODEL_VERSION },
});

export const handlers = [
  http.get("/api/v1/auth/me", () => HttpResponse.json(USER)),
  http.post("/api/v1/auth/token", () =>
    HttpResponse.json({
      access_token: "test-token",
      token_type: "bearer",
      expires_in_minutes: 30,
      role: "counsellor",
    }),
  ),
  http.get("/health", () =>
    HttpResponse.json({
      status: "ok",
      model_loaded: true,
      model_version: MODEL_VERSION,
      data_loaded: true,
      environment: "test",
    }),
  ),

  http.get("/api/v1/dashboard/statistics", () => HttpResponse.json(DASHBOARD)),
  http.get("/api/v1/students", ({ request }) => {
    const url = new URL(request.url);
    const band = url.searchParams.get("band");
    const rows = band ? STUDENTS.filter((row) => row.band === band) : STUDENTS;
    return HttpResponse.json(page(rows));
  }),
  http.get("/api/v1/students/:code", ({ params }) =>
    params.code === PROFILE.student_code
      ? HttpResponse.json(PROFILE)
      : HttpResponse.json({ detail: "Student not found" }, { status: 404 }),
  ),
  http.get("/api/v1/students/:code/explanation", () => HttpResponse.json(EXPLANATION)),
  http.get("/api/v1/students/:code/recommendations", () =>
    HttpResponse.json(RECOMMENDATIONS),
  ),
  http.get("/api/v1/students/:code/interventions", () => HttpResponse.json(INTERVENTIONS)),
  http.post("/api/v1/students/:code/interventions", async ({ request, params }) => {
    const body = (await request.json()) as { intervention_key: string; assigned_to: string };
    return HttpResponse.json(
      {
        id: 1,
        student_code: String(params.code),
        intervention_key: body.intervention_key,
        title: "Assessment planning support",
        status: "assigned",
        assigned_to: body.assigned_to,
        assigned_by: "counsellor",
        note: null,
        created_at: "2026-09-27T09:00:00Z",
      },
      { status: 201 },
    );
  }),

  // Stateful, so acknowledging actually removes the alert from the open queue on
  // refetch -- which is what the real API does. A stateless handler made the
  // acknowledge test assert against a list that never changed.
  http.get("/api/v1/alerts", ({ request }) => {
    const url = new URL(request.url);
    const acknowledged = url.searchParams.get("acknowledged");
    const rows =
      acknowledged === "false"
        ? alertState.filter((alert) => !alert.acknowledged)
        : alertState;
    return HttpResponse.json(rows);
  }),
  http.post("/api/v1/alerts/:id/acknowledge", ({ params }) => {
    const id = Number(params.id);
    const found = alertState.find((alert) => alert.id === id);
    if (!found) return HttpResponse.json({ detail: "Alert not found" }, { status: 404 });
    found.acknowledged = true;
    return HttpResponse.json(found);
  }),

  http.get("/api/v1/analytics/risk-distribution", () => HttpResponse.json(DISTRIBUTION)),
  http.get("/api/v1/analytics/scatter", () =>
    HttpResponse.json([
      { x: 14, y: 0.1461, band: "critical" },
      { x: 220, y: 0.0089, band: "low" },
    ]),
  ),
  http.get("/api/v1/analytics/feature-importance", () => HttpResponse.json(IMPORTANCE)),
  http.get("/api/v1/model/info", () => HttpResponse.json(MODEL_INFO)),
];

export const server = setupServer(...handlers);
