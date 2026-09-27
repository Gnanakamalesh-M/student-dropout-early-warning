/**
 * Cohort analytics. Available to every role, because none of it identifies a
 * student — which is the point of having an analyst role at all.
 */

import { useState } from "react";
import { Chart } from "../components/Chart";
import {
  useFeatureImportance,
  useRiskDistribution,
  useScatter,
} from "../hooks/queries";
import { Card, EmptyState, ErrorNotice, Spinner } from "../components/primitives";
import { BAND_META } from "../lib/format";
import { tokens } from "../lib/styles";
import type { BandKey } from "../api/types";

/** Fields the API allowlists for the scatter endpoint. Anything else is a 422 —
 * the endpoint deliberately cannot read an arbitrary column. */
const SCATTER_FIELDS = [
  ["clicks_28d", "Course activity, last four weeks"],
  ["clicks_7d", "Course activity, last week"],
  ["active_days_28d", "Active days, last four weeks"],
  ["days_since_last_activity", "Days since last activity"],
  ["submission_rate", "Assessment submission rate"],
  ["mean_score", "Average assessment score"],
  ["assessments_missed", "Assessments missed"],
] as const;

export function AnalyticsPage() {
  const [scatterField, setScatterField] = useState<string>("clicks_28d");
  const distribution = useRiskDistribution();
  const scatter = useScatter(scatterField);
  const importance = useFeatureImportance();

  return (
    <>
      <h1 style={{ margin: 0, fontSize: "20px" }}>Risk analytics</h1>

      <Card
        title="Risk distribution"
        subtitle="Most of the cohort sits at low risk; the tail is what staff act on"
      >
        {distribution.isLoading && <Spinner />}
        {distribution.error && <ErrorNotice error={distribution.error} />}
        {distribution.data && (
          <Chart
            data={[
              {
                x: distribution.data.map((bin) => (bin.lower + bin.upper) / 2),
                y: distribution.data.map((bin) => bin.count),
                type: "bar",
                marker: { color: tokens.color.accent },
                hovertemplate: "%{x:.1%} risk<br>%{y} students<extra></extra>",
              },
            ]}
            layout={{
              height: 300,
              margin: { l: 60, r: 16, t: 8, b: 44 },
              xaxis: { title: { text: "Estimated risk" }, tickformat: ".0%" },
              // Log scale: the low-risk bin is orders of magnitude larger, and a
              // linear axis would flatten the tail into invisibility — which is
              // exactly the part that matters.
              yaxis: { title: { text: "Students (log)" }, type: "log" },
              paper_bgcolor: "transparent",
              plot_bgcolor: "transparent",
              font: { family: tokens.font.body, size: 12 },
            }}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: "100%" }}
          />
        )}
      </Card>

      <Card
        title="Feature against risk"
        subtitle="Each point is a student at their most recent checkpoint"
        actions={
          <select
            aria-label="Feature"
            value={scatterField}
            onChange={(event) => setScatterField(event.target.value)}
            style={{
              padding: tokens.space(2),
              border: `1px solid ${tokens.color.border}`,
              borderRadius: "6px",
              fontSize: "13px",
            }}
          >
            {SCATTER_FIELDS.map(([value, label]) => (
              <option key={value} value={value}>
                {label}
              </option>
            ))}
          </select>
        }
      >
        {scatter.isLoading && <Spinner />}
        {scatter.error && <ErrorNotice error={scatter.error} />}
        {scatter.data && scatter.data.length === 0 && (
          <EmptyState message="No data for this feature." />
        )}
        {scatter.data && scatter.data.length > 0 && (
          <Chart
            data={(["low", "medium", "high", "critical"] as BandKey[]).map((band) => {
              const points = scatter.data.filter((point) => point.band === band);
              return {
                x: points.map((point) => point.x),
                y: points.map((point) => point.y),
                // Plain scatter, not scattergl: the WebGL trace is not in the basic Plotly
                // distribution, and at a few thousand points SVG is fast enough.
                type: "scatter",
                mode: "markers",
                name: BAND_META[band].label,
                marker: { size: 4, opacity: 0.5, color: BAND_META[band].color },
                hovertemplate: "%{x}<br>%{y:.1%} risk<extra></extra>",
              };
            })}
            layout={{
              height: 360,
              margin: { l: 60, r: 16, t: 8, b: 44 },
              xaxis: {
                title: {
                  text:
                    SCATTER_FIELDS.find(([value]) => value === scatterField)?.[1] ??
                    scatterField,
                },
              },
              yaxis: { title: { text: "Estimated risk" }, tickformat: ".0%" },
              legend: { orientation: "h", y: -0.2 },
              paper_bgcolor: "transparent",
              plot_bgcolor: "transparent",
              font: { family: tokens.font.body, size: 12 },
            }}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: "100%" }}
          />
        )}
      </Card>

      <Card
        title="What the model weighs across the cohort"
        subtitle="Mean absolute SHAP contribution, in log-odds of the pre-calibration score"
      >
        {importance.isLoading && <Spinner />}
        {importance.error && <ErrorNotice error={importance.error} />}
        {importance.data && (
          <>
            <Chart
              data={[
                {
                  // Reversed so the largest contributor sits at the top.
                  y: importance.data.map((item) => item.label).reverse(),
                  x: importance.data.map((item) => item.mean_abs_shap).reverse(),
                  type: "bar",
                  orientation: "h",
                  marker: { color: tokens.color.accent },
                  hovertemplate: "%{y}<br>%{x:.3f}<extra></extra>",
                },
              ]}
              layout={{
                height: Math.max(320, importance.data.length * 26),
                margin: { l: 280, r: 16, t: 8, b: 44 },
                xaxis: { title: { text: "Mean |SHAP|" } },
                paper_bgcolor: "transparent",
                plot_bgcolor: "transparent",
                font: { family: tokens.font.body, size: 11 },
              }}
              config={{ displayModeBar: false, responsive: true }}
              style={{ width: "100%" }}
            />
            <p
              style={{
                margin: `${tokens.space(3)} 0 0`,
                fontSize: "13px",
                color: tokens.color.muted,
                lineHeight: 1.6,
              }}
            >
              This describes how the model behaves, not what causes withdrawal. Note
              that how far through the course a student is carries substantial weight:
              real signal, but not something anyone can act on, which is why individual
              explanations separate it from actionable factors.
            </p>
          </>
        )}
      </Card>
    </>
  );
}
