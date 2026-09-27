/** TanStack Query hooks. Query keys are arrays so related caches invalidate together. */

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../api/client";

export const keys = {
  dashboard: ["dashboard"] as const,
  students: (params: Record<string, unknown>) => ["students", params] as const,
  student: (code: string) => ["student", code] as const,
  explanation: (code: string, asOf?: number) => ["explanation", code, asOf] as const,
  recommendations: (code: string) => ["recommendations", code] as const,
  interventions: (code: string) => ["interventions", code] as const,
  alerts: (params: Record<string, unknown>) => ["alerts", params] as const,
  distribution: ["distribution"] as const,
  scatter: (x: string) => ["scatter", x] as const,
  importance: ["importance"] as const,
  modelInfo: ["modelInfo"] as const,
};

export const useDashboard = () =>
  useQuery({ queryKey: keys.dashboard, queryFn: api.dashboard });

export const useStudents = (params: {
  band?: string;
  direction?: string;
  search?: string;
  page?: number;
  page_size?: number;
}) => useQuery({ queryKey: keys.students(params), queryFn: () => api.students(params) });

export const useStudent = (code: string) =>
  useQuery({ queryKey: keys.student(code), queryFn: () => api.student(code) });

export const useExplanation = (code: string, asOf?: number) =>
  useQuery({
    queryKey: keys.explanation(code, asOf),
    queryFn: () => api.explanation(code, asOf),
  });

export const useRecommendations = (code: string) =>
  useQuery({ queryKey: keys.recommendations(code), queryFn: () => api.recommendations(code) });

export const useInterventions = (code: string) =>
  useQuery({ queryKey: keys.interventions(code), queryFn: () => api.interventions(code) });

export const useAlerts = (params: { acknowledged?: boolean; severity?: string }) =>
  useQuery({ queryKey: keys.alerts(params), queryFn: () => api.alerts(params) });

export const useRiskDistribution = () =>
  useQuery({ queryKey: keys.distribution, queryFn: api.riskDistribution });

export const useScatter = (x: string) =>
  useQuery({ queryKey: keys.scatter(x), queryFn: () => api.scatter(x) });

export const useFeatureImportance = () =>
  useQuery({ queryKey: keys.importance, queryFn: api.featureImportance });

export const useModelInfo = () =>
  useQuery({ queryKey: keys.modelInfo, queryFn: api.modelInfo });

export function useAssignIntervention(code: string) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (payload: { intervention_key: string; assigned_to: string; note?: string }) =>
      api.assignIntervention(code, payload),
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: keys.interventions(code) });
    },
  });
}

export function useAcknowledgeAlert() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (id: number) => api.acknowledgeAlert(id),
    onSuccess: () => {
      // The dashboard shows an open-alert count, so it has to refresh too.
      void client.invalidateQueries({ queryKey: ["alerts"] });
      void client.invalidateQueries({ queryKey: keys.dashboard });
    },
  });
}
