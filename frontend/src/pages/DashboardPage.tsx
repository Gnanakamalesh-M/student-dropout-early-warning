/**
 * Cohort overview.
 *
 * The capacity framing is deliberate and is the main thing this page has to get
 * right. Phase 5 showed a recall target is not a usable control — 80% recall means
 * flagging 47% of the cohort at 4.9% precision — so the critical band is sized to
 * what staff can actually take on. The page states the configured budget next to
 * the counts, because a band count without that context invites reading the
 * cutoff as a property of the students rather than of the staffing.
 */

import { Link } from "react-router-dom";
import { useAlerts, useDashboard } from "../hooks/queries";
import {
  Card,
  Disclaimer,
  EmptyState,
  ErrorNotice,
  RiskBadge,
  Spinner,
  StatTile,
  UncalibratedWarning,
} from "../components/primitives";
import { BAND_META, formatRisk, formatShare } from "../lib/format";
import { tokens } from "../lib/styles";
import type { BandKey } from "../api/types";

const BAND_ORDER: BandKey[] = ["low", "medium", "high", "critical"];

export function DashboardPage() {
  const stats = useDashboard();
  const alerts = useAlerts({ acknowledged: false });

  if (stats.isLoading) return <Spinner label="Loading cohort" />;
  if (stats.error) return <ErrorNotice error={stats.error} />;
  if (!stats.data) return <EmptyState message="No cohort data available." />;

  const data = stats.data;
  const byBand = new Map(data.band_counts.map((item) => [item.band, item]));
  const needingAttention =
    (byBand.get("high")?.count ?? 0) + (byBand.get("critical")?.count ?? 0);

  return (
    <>
      <div>
        <h1 style={{ margin: 0, fontSize: "20px" }}>Cohort overview</h1>
        <p style={{ margin: `${tokens.space(1)} 0 0`, fontSize: "13px", color: tokens.color.muted }}>
          Model <code style={{ fontFamily: tokens.font.mono }}>{data.model_version}</code> ·{" "}
          {data.scored_checkpoints.toLocaleString()} scored checkpoints across{" "}
          {data.total_students.toLocaleString()} students
        </p>
      </div>

      {!data.is_calibrated && <UncalibratedWarning />}

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(170px, 1fr))",
          gap: tokens.space(3),
        }}
      >
        <StatTile label="Students" value={data.total_students.toLocaleString()} />
        {BAND_ORDER.map((band) => {
          const item = byBand.get(band);
          return (
            <StatTile
              key={band}
              label={`${BAND_META[band].label} risk`}
              value={(item?.count ?? 0).toLocaleString()}
              hint={item ? formatShare(item.share) + " of cohort" : undefined}
              emphasis={band}
            />
          );
        })}
        <StatTile
          label="Risk rising"
          value={data.worsening_count.toLocaleString()}
          hint="up since their last checkpoint"
        />
        <StatTile label="Open alerts" value={data.open_alerts.toLocaleString()} />
      </div>

      <Card
        title="What the bands mean here"
        subtitle="Cutoffs are a staffing decision, not a property of the students"
      >
        <p style={{ margin: 0, fontSize: "13px", lineHeight: 1.6 }}>
          The <strong>Critical</strong> band is sized to an alert budget of{" "}
          <strong>{formatShare(data.alert_budget)}</strong> of the cohort — roughly what
          staff can follow up in one cycle. That currently puts{" "}
          <strong>{needingAttention.toLocaleString()}</strong> students in the two upper
          bands. Widening the budget finds more students who go on to withdraw but
          lowers precision; a recall-led target was tested and is not workable, because
          reaching 80% recall would mean contacting close to half the cohort.
        </p>
        <p
          style={{
            margin: `${tokens.space(3)} 0 0`,
            fontSize: "13px",
            color: tokens.color.muted,
          }}
        >
          Band boundaries are not sharp: many students share the exact cutoff
          probability, so two students either side of a line are not meaningfully
          different to the model.
        </p>
      </Card>

      <Card
        title="Recent alerts"
        subtitle="Risk that rose or crossed a band, newest first"
        actions={
          <Link to="/alerts" style={{ fontSize: "13px", color: tokens.color.accent }}>
            All alerts →
          </Link>
        }
      >
        {alerts.isLoading && <Spinner label="Loading alerts" />}
        {alerts.error && <ErrorNotice error={alerts.error} />}
        {alerts.data && alerts.data.length === 0 && (
          <EmptyState message="No open alerts." />
        )}
        {alerts.data && alerts.data.length > 0 && (
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "13px" }}>
            <thead>
              <tr style={{ textAlign: "left", color: tokens.color.muted }}>
                <th style={cellStyle}>Student</th>
                <th style={cellStyle}>Checkpoint</th>
                <th style={cellStyle}>Risk</th>
                <th style={cellStyle}>Change</th>
                <th style={cellStyle}>Reason</th>
              </tr>
            </thead>
            <tbody>
              {alerts.data.slice(0, 8).map((alert) => (
                <tr key={alert.id} style={{ borderTop: `1px solid ${tokens.color.border}` }}>
                  <td style={cellStyle}>
                    <Link
                      to={`/students/${alert.student_code}`}
                      style={{ color: tokens.color.accent, fontFamily: tokens.font.mono }}
                    >
                      {alert.student_code}
                    </Link>
                  </td>
                  <td style={cellStyle}>Day {alert.checkpoint_day}</td>
                  <td style={cellStyle}>
                    <RiskBadge band={alert.band} probability={alert.probability} />
                  </td>
                  <td style={cellStyle}>
                    {alert.previous_probability === null
                      ? "—"
                      : `${formatRisk(alert.previous_probability)} → ${formatRisk(alert.probability)}`}
                  </td>
                  <td style={{ ...cellStyle, color: tokens.color.muted }}>
                    {alert.reason.replace(/_/g, " ")}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Card>

      <Disclaimer
        text={
          "Risk figures are probabilities produced by a statistical model, not predictions " +
          "of what will happen, and not a judgement about any student. They are intended to " +
          "prioritise offers of support and must not be used for admissions, funding or " +
          "disciplinary decisions."
        }
      />
    </>
  );
}

const cellStyle: React.CSSProperties = {
  padding: `${tokens.space(2)} ${tokens.space(2)} ${tokens.space(2)} 0`,
  fontWeight: 400,
};
