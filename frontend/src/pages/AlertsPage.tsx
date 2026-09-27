/**
 * Alert queue.
 *
 * Alerts are about *change* as much as level, so the previous probability is shown
 * beside the current one: a student steady at 30% needs less attention than one who
 * moved from 10% to 30%, and a bare current figure hides that difference.
 *
 * Acknowledging records that a human has looked. It is explicitly not the same as
 * offering support, and the page says so — otherwise "cleared the queue" starts to
 * read as "handled the student".
 */

import { useState } from "react";
import { Link } from "react-router-dom";
import { useAcknowledgeAlert, useAlerts } from "../hooks/queries";
import {
  Card,
  EmptyState,
  ErrorNotice,
  RiskBadge,
  Spinner,
} from "../components/primitives";
import { formatDateTime, formatRisk } from "../lib/format";
import { tokens } from "../lib/styles";

const REASON_LABEL: Record<string, string> = {
  band_escalated: "Moved into a higher band",
  rapid_increase: "Risk rose sharply",
  sustained_increase: "Risk rose for several checkpoints",
};

export function AlertsPage() {
  const [showAcknowledged, setShowAcknowledged] = useState(false);
  const [severity, setSeverity] = useState<"" | "high" | "critical">("");
  const alerts = useAlerts({
    acknowledged: showAcknowledged ? undefined : false,
    severity: severity || undefined,
  });
  const acknowledge = useAcknowledgeAlert();

  return (
    <>
      <h1 style={{ margin: 0, fontSize: "20px" }}>Alerts</h1>

      <Card
        title="Queue"
        subtitle="Acknowledging records that someone has reviewed the alert. It does not offer the student anything."
        actions={
          <div style={{ display: "flex", gap: tokens.space(3), alignItems: "center" }}>
            <select
              aria-label="Severity"
              value={severity}
              onChange={(event) =>
                setSeverity(event.target.value as "" | "high" | "critical")
              }
              style={controlStyle}
            >
              <option value="">Any severity</option>
              <option value="critical">Critical</option>
              <option value="high">High</option>
            </select>
            <label
              style={{
                fontSize: "13px",
                display: "flex",
                gap: tokens.space(1.5),
                alignItems: "center",
              }}
            >
              <input
                type="checkbox"
                checked={showAcknowledged}
                onChange={(event) => setShowAcknowledged(event.target.checked)}
              />
              Include acknowledged
            </label>
          </div>
        }
      >
        {alerts.isLoading && <Spinner />}
        {alerts.error && <ErrorNotice error={alerts.error} />}
        {alerts.data && alerts.data.length === 0 && (
          <EmptyState message="Nothing in the queue." />
        )}
        {alerts.data && alerts.data.length > 0 && (
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "13px" }}>
            <thead>
              <tr style={{ textAlign: "left", color: tokens.color.muted }}>
                <th style={cellStyle}>Student</th>
                <th style={cellStyle}>Checkpoint</th>
                <th style={cellStyle}>Risk</th>
                <th style={cellStyle}>Change</th>
                <th style={cellStyle}>Why</th>
                <th style={cellStyle}>Raised</th>
                <th style={cellStyle} />
              </tr>
            </thead>
            <tbody>
              {alerts.data.map((alert) => (
                <tr
                  key={alert.id}
                  style={{
                    borderTop: `1px solid ${tokens.color.border}`,
                    opacity: alert.acknowledged ? 0.55 : 1,
                  }}
                >
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
                    {alert.previous_probability === null ? (
                      <span style={{ color: tokens.color.muted }}>first checkpoint</span>
                    ) : (
                      `${formatRisk(alert.previous_probability)} → ${formatRisk(alert.probability)}`
                    )}
                  </td>
                  <td style={{ ...cellStyle, color: tokens.color.muted }}>
                    {REASON_LABEL[alert.reason] ?? alert.reason}
                  </td>
                  <td style={{ ...cellStyle, color: tokens.color.muted }}>
                    {formatDateTime(alert.created_at)}
                  </td>
                  <td style={cellStyle}>
                    {alert.acknowledged ? (
                      <span style={{ color: tokens.color.muted }}>Acknowledged</span>
                    ) : (
                      <button
                        onClick={() => acknowledge.mutate(alert.id)}
                        disabled={acknowledge.isPending}
                        style={{
                          padding: `${tokens.space(1)} ${tokens.space(2.5)}`,
                          fontSize: "12px",
                          borderRadius: "6px",
                          border: `1px solid ${tokens.color.border}`,
                          background: tokens.color.surface,
                          cursor: "pointer",
                          whiteSpace: "nowrap",
                        }}
                      >
                        Acknowledge
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        {acknowledge.error && <ErrorNotice error={acknowledge.error} />}
      </Card>
    </>
  );
}

const controlStyle: React.CSSProperties = {
  padding: tokens.space(2),
  border: `1px solid ${tokens.color.border}`,
  borderRadius: "6px",
  fontSize: "13px",
};
const cellStyle: React.CSSProperties = {
  padding: `${tokens.space(2)} ${tokens.space(2)} ${tokens.space(2)} 0`,
  fontWeight: 400,
};
