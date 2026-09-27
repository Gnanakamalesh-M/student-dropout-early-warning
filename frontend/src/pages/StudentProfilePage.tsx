/**
 * Individual student profile: the page that carries the most responsibility.
 *
 * Four decisions shape it, all from ADR-0005:
 *
 * The SHAP panel is headed "Factors the model weighted", never "Reasons". Phase 7
 * measured top-3 factor agreement against near-identical students at 0.90 for the
 * high-risk group — good, not perfect — so presenting these as *the* reasons would
 * overstate what they are.
 *
 * Contextual factors are shown in a separate, de-emphasised list. The largest
 * single contributor globally is how far through the course a student is, which is
 * real signal but nothing a counsellor can act on, and letting it head the list
 * would waste their attention.
 *
 * Protective factors are shown. An explanation made only of negatives
 * misrepresents a student who is doing several things well.
 *
 * Interventions are offers requiring assignment. The button says "Assign", and the
 * card states that nothing has been sent.
 */

import { useMemo, useState } from "react";
import { Chart } from "../components/Chart";
import { Link, useParams } from "react-router-dom";
import {
  useAssignIntervention,
  useExplanation,
  useInterventions,
  useRecommendations,
  useStudent,
} from "../hooks/queries";
import {
  Card,
  Disclaimer,
  DirectionBadge,
  EmptyState,
  ErrorNotice,
  ImpactBadge,
  RiskBadge,
  Spinner,
} from "../components/primitives";
import {
  BAND_META,
  formatFeatureValue,
  formatRisk,
  humaniseFieldName,
} from "../lib/format";
import { tokens } from "../lib/styles";
import type { Factor, TrajectoryPoint } from "../api/types";

export function StudentProfilePage() {
  const { code = "" } = useParams();
  const student = useStudent(code);
  const explanation = useExplanation(code);
  const recommendations = useRecommendations(code);
  const interventions = useInterventions(code);
  const assign = useAssignIntervention(code);
  const [assignee, setAssignee] = useState("");

  if (student.isLoading) return <Spinner label="Loading student" />;
  if (student.error) return <ErrorNotice error={student.error} />;
  if (!student.data) return <EmptyState message="Student not found." />;

  const profile = student.data;

  return (
    <>
      <div>
        <Link to="/students" style={{ fontSize: "13px", color: tokens.color.accent }}>
          ← All students
        </Link>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: tokens.space(3),
            marginTop: tokens.space(2),
            flexWrap: "wrap",
          }}
        >
          <h1 style={{ margin: 0, fontSize: "20px", fontFamily: tokens.font.mono }}>
            {profile.student_code}
          </h1>
          <RiskBadge band={profile.risk.band.key} probability={profile.risk.probability} />
          <DirectionBadge direction={profile.risk_direction} />
        </div>
        <p style={{ margin: `${tokens.space(2)} 0 0`, fontSize: "13px", color: tokens.color.muted }}>
          Module {profile.code_module} · {profile.code_presentation} · scored at day{" "}
          {profile.risk.checkpoint_day} · model{" "}
          <code style={{ fontFamily: tokens.font.mono }}>{profile.risk.model_version}</code>
        </p>
      </div>

      <Card
        title="Risk trajectory"
        subtitle={`${profile.risk.band.action}`}
      >
        {profile.trajectory.length < 2 ? (
          <EmptyState
            message="Only one checkpoint has been scored, so there is no trend to show yet."
          />
        ) : (
          <TrajectoryChart trajectory={profile.trajectory} />
        )}
      </Card>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))",
          gap: tokens.space(4),
        }}
      >
        <FieldCard title="Engagement" fields={profile.engagement} />
        <FieldCard title="Assessment" fields={profile.assessment} />
        <FieldCard title="Enrolment context" fields={profile.context} />
      </div>

      <Card
        title="Factors the model weighted"
        subtitle={explanation.data?.attribution_basis}
      >
        {explanation.isLoading && <Spinner label="Computing explanation" />}
        {explanation.error && <ErrorNotice error={explanation.error} />}
        {explanation.data && (
          <div style={{ display: "grid", gap: tokens.space(4) }}>
            {explanation.data.notes.map((note) => (
              <p key={note} style={{ margin: 0, fontSize: "13px", color: tokens.color.muted }}>
                {note}
              </p>
            ))}

            <FactorList
              heading="Weighted toward higher risk"
              factors={explanation.data.risk_factors}
              emptyMessage="No individual factor carried meaningful weight toward higher risk."
              tone="risk"
            />

            {explanation.data.protective_factors.length > 0 && (
              <FactorList
                heading="Weighted toward lower risk"
                factors={explanation.data.protective_factors}
                emptyMessage=""
                tone="protective"
              />
            )}

            {explanation.data.context_factors.length > 0 && (
              <FactorList
                heading="Context (not actionable)"
                factors={explanation.data.context_factors}
                emptyMessage=""
                tone="context"
              />
            )}

            <Disclaimer text={explanation.data.disclaimer} />
          </div>
        )}
      </Card>

      <Card
        title="Suggested support"
        subtitle={recommendations.data?.band_guidance}
      >
        {recommendations.isLoading && <Spinner label="Matching support options" />}
        {recommendations.error && <ErrorNotice error={recommendations.error} />}
        {recommendations.data && (
          <div style={{ display: "grid", gap: tokens.space(4) }}>
            <div
              style={{
                padding: tokens.space(3),
                background: tokens.color.accentSoft,
                borderRadius: tokens.radius,
                fontSize: "13px",
                lineHeight: 1.6,
              }}
            >
              <div style={{ fontWeight: 600, marginBottom: 4 }}>
                Case note{" "}
                <span style={{ fontWeight: 400, color: tokens.color.muted }}>
                  ({recommendations.data.case_note_source === "llm" ? "drafted" : "generated"})
                </span>
              </div>
              {recommendations.data.case_note}
            </div>

            {recommendations.data.recommendations.length === 0 ? (
              <EmptyState message="No outreach is suggested at this risk band." />
            ) : (
              <>
                <p style={{ margin: 0, fontSize: "13px", color: tokens.color.muted }}>
                  Nothing below has been sent. Each is an offer of support that a member
                  of staff has to assign.
                </p>
                <label style={{ fontSize: "13px", display: "grid", gap: 4, maxWidth: "320px" }}>
                  Assign to
                  <input
                    value={assignee}
                    onChange={(event) => setAssignee(event.target.value)}
                    placeholder="username"
                    style={{
                      padding: tokens.space(2),
                      border: `1px solid ${tokens.color.border}`,
                      borderRadius: "6px",
                      fontSize: "13px",
                    }}
                  />
                </label>
                <ul style={{ listStyle: "none", padding: 0, margin: 0, display: "grid", gap: tokens.space(3) }}>
                  {recommendations.data.recommendations.map((item) => (
                    <li
                      key={item.key}
                      style={{
                        border: `1px solid ${tokens.color.border}`,
                        borderRadius: tokens.radius,
                        padding: tokens.space(3),
                      }}
                    >
                      <div style={{ display: "flex", justifyContent: "space-between", gap: tokens.space(3) }}>
                        <div>
                          <strong style={{ fontSize: "14px" }}>{item.title}</strong>
                          <div style={{ fontSize: "12px", color: tokens.color.muted, marginTop: 2 }}>
                            {item.intensity} · about {item.typical_effort_minutes} min ·{" "}
                            {item.owner_role}
                          </div>
                        </div>
                        <button
                          disabled={!assignee || assign.isPending}
                          onClick={() =>
                            assign.mutate({
                              intervention_key: item.key,
                              assigned_to: assignee,
                            })
                          }
                          style={{
                            alignSelf: "start",
                            padding: `${tokens.space(1.5)} ${tokens.space(3)}`,
                            fontSize: "13px",
                            borderRadius: "6px",
                            border: `1px solid ${tokens.color.accent}`,
                            background: assignee ? tokens.color.accent : tokens.color.border,
                            color: assignee ? "#fff" : tokens.color.muted,
                            cursor: assignee ? "pointer" : "not-allowed",
                            whiteSpace: "nowrap",
                          }}
                        >
                          Assign
                        </button>
                      </div>
                      <p style={{ margin: `${tokens.space(2)} 0 0`, fontSize: "13px", lineHeight: 1.5 }}>
                        {item.description}
                      </p>
                      <p style={{ margin: `${tokens.space(2)} 0 0`, fontSize: "12px", color: tokens.color.muted }}>
                        {item.rationale}
                      </p>
                    </li>
                  ))}
                </ul>
              </>
            )}

            {assign.error && <ErrorNotice error={assign.error} />}
          </div>
        )}
      </Card>

      <Card title="Assigned support" subtitle="Actions a member of staff has taken on">
        {interventions.isLoading && <Spinner />}
        {interventions.data && interventions.data.length === 0 && (
          <EmptyState message="Nothing assigned yet." />
        )}
        {interventions.data && interventions.data.length > 0 && (
          <ul style={{ margin: 0, paddingLeft: tokens.space(5), fontSize: "13px", lineHeight: 1.8 }}>
            {interventions.data.map((record) => (
              <li key={record.id}>
                <strong>{record.title}</strong> — {record.status}, assigned to{" "}
                {record.assigned_to ?? "nobody"} by {record.assigned_by ?? "unknown"}
              </li>
            ))}
          </ul>
        )}
      </Card>
    </>
  );
}

function TrajectoryChart({ trajectory }: { trajectory: TrajectoryPoint[] }) {
  const maxProbability = useMemo(
    () => Math.max(...trajectory.map((point) => point.probability), 0.1),
    [trajectory],
  );

  return (
    <Chart
      data={[
        {
          x: trajectory.map((point) => point.checkpoint_day),
          y: trajectory.map((point) => point.probability),
          type: "scatter",
          mode: "lines+markers",
          line: { color: tokens.color.accent, width: 2 },
          marker: {
            size: 10,
            // Each marker takes its band colour, so the line reads as a path
            // through the bands rather than an abstract curve.
            color: trajectory.map((point) => BAND_META[point.band].color),
          },
          hovertemplate: "Day %{x}<br>%{y:.1%} risk<extra></extra>",
          name: "Risk",
        },
      ]}
      layout={{
        height: 300,
        margin: { l: 56, r: 16, t: 8, b: 44 },
        xaxis: { title: { text: "Course day" }, dtick: 30 },
        yaxis: {
          title: { text: "Estimated risk" },
          tickformat: ".0%",
          range: [0, maxProbability * 1.25],
        },
        showlegend: false,
        paper_bgcolor: "transparent",
        plot_bgcolor: "transparent",
        font: { family: tokens.font.body, size: 12 },
      }}
      config={{ displayModeBar: false, responsive: true }}
      style={{ width: "100%" }}
    />
  );
}

function FactorList({
  heading,
  factors,
  emptyMessage,
  tone,
}: {
  heading: string;
  factors: Factor[];
  emptyMessage: string;
  tone: "risk" | "protective" | "context";
}) {
  if (factors.length === 0) {
    return emptyMessage ? <EmptyState message={emptyMessage} /> : null;
  }
  const muted = tone === "context";
  return (
    <div>
      <h3
        style={{
          margin: `0 0 ${tokens.space(2)}`,
          fontSize: "13px",
          fontWeight: 600,
          color: muted ? tokens.color.muted : tokens.color.text,
        }}
      >
        {heading}
      </h3>
      <ul style={{ listStyle: "none", padding: 0, margin: 0, display: "grid", gap: tokens.space(2) }}>
        {factors.map((factor) => (
          <li
            key={factor.feature}
            style={{
              display: "flex",
              gap: tokens.space(3),
              alignItems: "baseline",
              opacity: muted ? 0.75 : 1,
            }}
          >
            <ImpactBadge impact={factor.impact} />
            <div style={{ fontSize: "13px", lineHeight: 1.5 }}>
              <strong style={{ fontWeight: 600 }}>{factor.label}</strong>
              <div style={{ color: tokens.color.muted }}>{factor.sentence}</div>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}

function FieldCard({
  title,
  fields,
}: {
  title: string;
  fields: Record<string, number | null>;
}) {
  const entries = Object.entries(fields);
  return (
    <Card title={title}>
      {entries.length === 0 ? (
        <EmptyState message="No data recorded." />
      ) : (
        <dl style={{ margin: 0, display: "grid", gap: tokens.space(2), fontSize: "13px" }}>
          {entries.map(([name, value]) => (
            <div key={name} style={{ display: "flex", justifyContent: "space-between", gap: tokens.space(3) }}>
              <dt style={{ color: tokens.color.muted }}>{humaniseFieldName(name)}</dt>
              <dd
                style={{
                  margin: 0,
                  fontWeight: 600,
                  fontFamily: tokens.font.mono,
                  color: value === null ? tokens.color.muted : tokens.color.text,
                }}
              >
                {formatFeatureValue(value)}
              </dd>
            </div>
          ))}
        </dl>
      )}
    </Card>
  );
}

export { formatRisk };
