/**
 * Typed fetch wrapper.
 *
 * Two behaviours worth naming:
 *
 * A 401 clears the stored token and dispatches `auth:expired`, so an expired
 * session lands the user on the login screen rather than showing a wall of
 * failed panels. Without that, a 30-minute token silently degrades every page.
 *
 * Errors carry the API's `detail` string. FastAPI's 403 messages say which role
 * is required, and surfacing that is far more useful to a user than "Forbidden" —
 * an analyst hitting a student profile should be told the restriction is
 * deliberate.
 */

import type {
  Alert,
  CurrentUser,
  DashboardStatistics,
  DistributionBin,
  ExplanationResponse,
  FeatureImportance,
  HealthResponse,
  InterventionRecord,
  ModelInfo,
  Page,
  RecommendationsResponse,
  ScatterPoint,
  StudentProfile,
  StudentSummary,
  Token,
} from "./types";

const TOKEN_KEY = "ews.token";
export const AUTH_EXPIRED_EVENT = "auth:expired";

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }

  /** True when the failure is an access-control decision rather than a fault. */
  get isForbidden(): boolean {
    return this.status === 403;
  }
}

export function getToken(): string | null {
  try {
    return localStorage.getItem(TOKEN_KEY);
  } catch {
    // Private browsing or blocked storage. Treat as logged out rather than
    // crashing the app shell.
    return null;
  }
}

export function setToken(token: string | null): void {
  try {
    if (token === null) localStorage.removeItem(TOKEN_KEY);
    else localStorage.setItem(TOKEN_KEY, token);
  } catch {
    /* storage unavailable; the session simply will not persist a reload */
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const token = getToken();
  const headers = new Headers(init.headers);
  if (token) headers.set("Authorization", `Bearer ${token}`);
  if (init.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  const response = await fetch(path, { ...init, headers });

  if (response.status === 401) {
    setToken(null);
    window.dispatchEvent(new Event(AUTH_EXPIRED_EVENT));
    throw new ApiError(401, "Your session has expired. Please sign in again.");
  }

  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = (await response.json()) as { detail?: unknown };
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      /* non-JSON error body; keep the status text */
    }
    throw new ApiError(response.status, detail);
  }

  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export const api = {
  async login(username: string, password: string): Promise<Token> {
    // OAuth2 password flow expects form encoding, not JSON.
    const body = new URLSearchParams({ username, password });
    const response = await fetch("/api/v1/auth/token", {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body,
    });
    if (!response.ok) {
      throw new ApiError(response.status, "Incorrect username or password");
    }
    const token = (await response.json()) as Token;
    setToken(token.access_token);
    return token;
  },

  logout(): void {
    setToken(null);
  },

  me: () => request<CurrentUser>("/api/v1/auth/me"),
  health: () => request<HealthResponse>("/health"),

  students: (params: {
    band?: string;
    direction?: string;
    module?: string;
    search?: string;
    page?: number;
    page_size?: number;
  }) => {
    const query = new URLSearchParams();
    for (const [key, value] of Object.entries(params)) {
      if (value !== undefined && value !== "") query.set(key, String(value));
    }
    return request<Page<StudentSummary>>(`/api/v1/students?${query}`);
  },

  student: (code: string) => request<StudentProfile>(`/api/v1/students/${code}`),
  explanation: (code: string, asOf?: number) =>
    request<ExplanationResponse>(
      `/api/v1/students/${code}/explanation${asOf ? `?as_of=${asOf}` : ""}`,
    ),
  recommendations: (code: string) =>
    request<RecommendationsResponse>(`/api/v1/students/${code}/recommendations`),
  interventions: (code: string) =>
    request<InterventionRecord[]>(`/api/v1/students/${code}/interventions`),
  assignIntervention: (
    code: string,
    payload: { intervention_key: string; assigned_to: string; note?: string },
  ) =>
    request<InterventionRecord>(`/api/v1/students/${code}/interventions`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  dashboard: () => request<DashboardStatistics>("/api/v1/dashboard/statistics"),
  riskDistribution: () =>
    request<DistributionBin[]>("/api/v1/analytics/risk-distribution"),
  scatter: (x: string) =>
    request<ScatterPoint[]>(`/api/v1/analytics/scatter?x=${encodeURIComponent(x)}`),
  featureImportance: () =>
    request<FeatureImportance[]>("/api/v1/analytics/feature-importance"),

  alerts: (params: { acknowledged?: boolean; severity?: string } = {}) => {
    const query = new URLSearchParams();
    if (params.acknowledged !== undefined)
      query.set("acknowledged", String(params.acknowledged));
    if (params.severity) query.set("severity", params.severity);
    return request<Alert[]>(`/api/v1/alerts?${query}`);
  },
  acknowledgeAlert: (id: number) =>
    request<Alert>(`/api/v1/alerts/${id}/acknowledge`, { method: "POST" }),

  modelInfo: () => request<ModelInfo>("/api/v1/model/info"),
};
