/**
 * Types mirroring the FastAPI schemas in `backend/app/schemas.py`.
 *
 * Hand-written rather than code-generated. The API is small and stable, and a
 * generator would reproduce the shapes without the one thing that matters here:
 * `RiskAssessment` deliberately requires `modelVersion`, `isCalibrated` and
 * `disclaimer`, so it is not possible to render a probability in this app without
 * also having what produced it and what it means. A generated `?` on any of those
 * would quietly permit exactly what the backend schema was designed to prevent.
 *
 * A contract test asserts these match `docs/openapi.json`, so drift fails rather
 * than being discovered in the browser.
 */

export type BandKey = "low" | "medium" | "high" | "critical";
export type RiskDirection = "improving" | "stable" | "worsening" | "unknown";
export type ImpactBand = "High" | "Medium" | "Low";
export type Role = "admin" | "counsellor" | "analyst";

export interface Token {
  access_token: string;
  token_type: "bearer";
  expires_in_minutes: number;
  role: Role;
}

export interface CurrentUser {
  username: string;
  role: Role;
  may_view_individuals: boolean;
  may_assign_interventions: boolean;
}

export interface RiskBandInfo {
  key: BandKey;
  label: string;
  min_probability: number;
  color: string;
  action: string;
}

/** A probability is never returned alone. See the module comment. */
export interface RiskAssessment {
  probability: number;
  band: RiskBandInfo;
  checkpoint_day: number;
  model_version: string;
  is_calibrated: boolean;
  disclaimer: string;
}

export interface StudentSummary {
  student_code: string;
  code_module: string;
  code_presentation: string;
  checkpoint_day: number;
  probability: number;
  band: BandKey;
  risk_direction: RiskDirection;
}

export interface TrajectoryPoint {
  checkpoint_day: number;
  probability: number;
  band: BandKey;
}

export interface StudentProfile {
  student_code: string;
  code_module: string;
  code_presentation: string;
  risk: RiskAssessment;
  risk_direction: RiskDirection;
  trajectory: TrajectoryPoint[];
  engagement: Record<string, number | null>;
  assessment: Record<string, number | null>;
  context: Record<string, number | null>;
}

export interface Factor {
  feature: string;
  label: string;
  impact: ImpactBand;
  direction: "increases" | "decreases";
  actionable: boolean;
  sentence: string;
}

export interface ExplanationResponse {
  student_code: string;
  checkpoint_day: number;
  risk: RiskAssessment;
  risk_factors: Factor[];
  protective_factors: Factor[];
  /** Contributors nobody can act on, kept separate so they do not crowd out
   * factors staff can respond to. */
  context_factors: Factor[];
  notes: string[];
  disclaimer: string;
  attribution_basis: string;
}

export interface Recommendation {
  key: string;
  title: string;
  description: string;
  intensity: "light" | "moderate";
  owner_role: string;
  typical_effort_minutes: number;
  matched_factor: string;
  rationale: string;
  status: "recommended";
  /** Literal `true` in the backend schema: an action that skips review is not
   * representable. */
  requires_human_review: true;
}

export interface RecommendationsResponse {
  student_code: string;
  band: BandKey;
  band_guidance: string;
  recommendations: Recommendation[];
  case_note: string;
  case_note_source: "llm" | "template";
  disclaimer: string;
}

export interface InterventionRecord {
  id: number;
  student_code: string;
  intervention_key: string;
  title: string;
  status: "recommended" | "assigned" | "in_progress" | "completed";
  assigned_to: string | null;
  assigned_by: string | null;
  note: string | null;
  created_at: string;
}

export interface BandCount {
  band: BandKey;
  label: string;
  count: number;
  share: number;
}

export interface DashboardStatistics {
  total_students: number;
  scored_checkpoints: number;
  band_counts: BandCount[];
  worsening_count: number;
  open_alerts: number;
  model_version: string;
  is_calibrated: boolean;
  /** Share of the cohort the configured capacity allows contacting. */
  alert_budget: number;
}

export interface DistributionBin {
  lower: number;
  upper: number;
  count: number;
}

export interface ScatterPoint {
  x: number;
  y: number;
  band: BandKey;
}

export interface FeatureImportance {
  feature: string;
  label: string;
  mean_abs_shap: number;
}

export interface Alert {
  id: number;
  student_code: string;
  checkpoint_day: number;
  reason: "band_escalated" | "rapid_increase" | "sustained_increase";
  severity: "high" | "critical";
  probability: number;
  previous_probability: number | null;
  band: BandKey;
  acknowledged: boolean;
  created_at: string;
}

export interface ModelInfo {
  model_version: string;
  model_type: string;
  created_at: string;
  git_commit: string | null;
  n_features: number;
  imbalance_strategy: string;
  calibration_method: string;
  train_presentations: string[];
  test_presentations: string[];
  metrics: Record<string, unknown>;
  bands: RiskBandInfo[];
  is_calibrated: boolean;
  /** Required, not optional: a client cannot present the metrics without them. */
  limitations: string[];
}

export interface Meta {
  total: number;
  page: number;
  page_size: number;
  model_version: string | null;
}

export interface Page<T> {
  data: T[];
  meta: Meta;
}

export interface HealthResponse {
  status: "ok" | "degraded";
  model_loaded: boolean;
  model_version: string | null;
  data_loaded: boolean;
  environment: string;
}
