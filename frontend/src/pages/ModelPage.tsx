/**
 * Model card, served from the API.
 *
 * Limitations are rendered above the metrics, not below them. That ordering is the
 * whole point of this page: the headline figure here is PR-AUC 0.089 against a 2.9%
 * base rate, which is a real improvement over any single heuristic and still means
 * most students contacted were not about to withdraw. Showing the number first and
 * the caveats last invites reading the first and skipping the second.
 *
 * The API returns `limitations` as a required field for the same reason.
 */

import { useModelInfo } from "../hooks/queries";
import {
  Card,
  EmptyState,
  ErrorNotice,
  RiskBadge,
  Spinner,
  UncalibratedWarning,
} from "../components/primitives";
import { formatRisk } from "../lib/format";
import { tokens } from "../lib/styles";

/** Metrics worth surfacing, with plain-language names. The raw metadata carries
 * nested structures too, shown as JSON below for anyone who wants them. */
const HEADLINE_METRICS: [string, string][] = [
  ["test_pr_auc", "PR-AUC (test)"],
  ["test_roc_auc", "ROC-AUC (test)"],
  ["test_brier", "Brier score"],
  ["test_base_rate", "Base rate"],
  ["cv_pr_auc", "PR-AUC (cross-validated)"],
];

export function ModelPage() {
  const info = useModelInfo();

  if (info.isLoading) return <Spinner label="Loading model card" />;
  if (info.error) return <ErrorNotice error={info.error} />;
  if (!info.data) return <EmptyState message="No model information available." />;

  const data = info.data;
  const metrics = data.metrics as Record<string, unknown>;

  const asNumber = (key: string): number | null => {
    const value = metrics[key];
    return typeof value === "number" ? value : null;
  };

  return (
    <>
      <div>
        <h1 style={{ margin: 0, fontSize: "20px" }}>Model card</h1>
        <p
          style={{
            margin: `${tokens.space(1)} 0 0`,
            fontSize: "13px",
            color: tokens.color.muted,
          }}
        >
          <code style={{ fontFamily: tokens.font.mono }}>{data.model_version}</code> ·{" "}
          {data.model_type} · {data.n_features} features · trained{" "}
          {data.train_presentations.join(", ")}, tested on{" "}
          {data.test_presentations.join(", ")}
        </p>
      </div>

      {!data.is_calibrated && <UncalibratedWarning />}

      <Card
        title="Read this first"
        subtitle="What the model does not do, ahead of what it does"
      >
        <ul
          style={{
            margin: 0,
            paddingLeft: tokens.space(5),
            fontSize: "13px",
            lineHeight: 1.7,
          }}
        >
          {data.limitations.map((limitation) => (
            <li key={limitation}>{limitation}</li>
          ))}
        </ul>
      </Card>

      <Card
        title="Performance"
        subtitle="Measured on the held-out final presentation, which the model never trained on"
      >
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))",
            gap: tokens.space(3),
          }}
        >
          {HEADLINE_METRICS.map(([key, label]) => {
            const value = asNumber(key);
            if (value === null) return null;
            return (
              <div
                key={key}
                style={{
                  border: `1px solid ${tokens.color.border}`,
                  borderRadius: tokens.radius,
                  padding: tokens.space(3),
                }}
              >
                <div
                  style={{ fontSize: "12px", color: tokens.color.muted, marginBottom: 4 }}
                >
                  {label}
                </div>
                <div style={{ fontSize: "20px", fontWeight: 650 }}>
                  {value.toFixed(4)}
                </div>
              </div>
            );
          })}
        </div>
        <p
          style={{
            margin: `${tokens.space(3)} 0 0`,
            fontSize: "13px",
            lineHeight: 1.6,
            color: tokens.color.muted,
          }}
        >
          PR-AUC rather than accuracy: at a base rate near 3%, a model that predicted
          &ldquo;no withdrawal&rdquo; for everyone would be 97% accurate and useless.
          The Brier score reports calibration — whether a displayed percentage means
          what it says.
        </p>
      </Card>

      <Card
        title="How it was built"
        subtitle="Imbalance handling, calibration, and provenance"
      >
        <dl
          style={{
            margin: 0,
            display: "grid",
            gap: tokens.space(2),
            fontSize: "13px",
          }}
        >
          <Row label="Imbalance strategy" value={data.imbalance_strategy} />
          <Row label="Calibration" value={data.calibration_method} />
          <Row label="Trained at" value={data.created_at} />
          <Row label="Git commit" value={data.git_commit ?? "not recorded"} mono />
        </dl>
        <p
          style={{
            margin: `${tokens.space(3)} 0 0`,
            fontSize: "13px",
            lineHeight: 1.6,
            color: tokens.color.muted,
          }}
        >
          The imbalance strategy is <strong>{data.imbalance_strategy}</strong>. Class
          weights, SMOTE and undersampling were all tested; none beat doing nothing by
          more than fold noise, and SMOTE was measurably worse — it produced feature
          vectors that no real student could have, because it interpolates each
          dimension independently and breaks the relationships between them.
        </p>
      </Card>

      <Card
        title="Risk bands"
        subtitle="Cutoffs are sized to staffing capacity, not derived from the data"
      >
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "13px" }}>
          <thead>
            <tr style={{ textAlign: "left", color: tokens.color.muted }}>
              <th style={cellStyle}>Band</th>
              <th style={cellStyle}>From</th>
              <th style={cellStyle}>Action</th>
            </tr>
          </thead>
          <tbody>
            {data.bands.map((band) => (
              <tr key={band.key} style={{ borderTop: `1px solid ${tokens.color.border}` }}>
                <td style={cellStyle}>
                  <RiskBadge band={band.key} />
                </td>
                <td style={{ ...cellStyle, fontFamily: tokens.font.mono }}>
                  {formatRisk(band.min_probability)}
                </td>
                <td style={cellStyle}>{band.action}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p
          style={{
            margin: `${tokens.space(3)} 0 0`,
            fontSize: "13px",
            color: tokens.color.muted,
            lineHeight: 1.6,
          }}
        >
          These are an institutional policy choice, not a validated risk scale. A
          different institution with different staffing should re-derive them.
        </p>
      </Card>

      <Card title="Full metadata" subtitle="Everything recorded with the artifact">
        <pre
          style={{
            margin: 0,
            fontSize: "12px",
            fontFamily: tokens.font.mono,
            background: tokens.color.background,
            padding: tokens.space(3),
            borderRadius: tokens.radius,
            overflowX: "auto",
            maxHeight: "400px",
          }}
        >
          {JSON.stringify(metrics, null, 2)}
        </pre>
      </Card>
    </>
  );
}

function Row({
  label,
  value,
  mono,
}: {
  label: string;
  value: string;
  mono?: boolean;
}) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", gap: tokens.space(3) }}>
      <dt style={{ color: tokens.color.muted }}>{label}</dt>
      <dd
        style={{
          margin: 0,
          fontWeight: 600,
          fontFamily: mono ? tokens.font.mono : undefined,
          textAlign: "right",
        }}
      >
        {value}
      </dd>
    </div>
  );
}

const cellStyle: React.CSSProperties = {
  padding: `${tokens.space(2)} ${tokens.space(2)} ${tokens.space(2)} 0`,
  fontWeight: 400,
  verticalAlign: "top",
};
