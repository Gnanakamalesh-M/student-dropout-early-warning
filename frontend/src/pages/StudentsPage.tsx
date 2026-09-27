import { useState } from "react";
import { Link } from "react-router-dom";
import { useStudents } from "../hooks/queries";
import {
  Card,
  DirectionBadge,
  EmptyState,
  ErrorNotice,
  RiskBadge,
  Spinner,
} from "../components/primitives";
import { tokens } from "../lib/styles";
import type { BandKey, RiskDirection } from "../api/types";

const PAGE_SIZE = 25;

export function StudentsPage() {
  const [band, setBand] = useState<BandKey | "">("");
  const [direction, setDirection] = useState<RiskDirection | "">("");
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(1);

  const query = useStudents({
    band: band || undefined,
    direction: direction || undefined,
    search: search || undefined,
    page,
    page_size: PAGE_SIZE,
  });

  const totalPages = query.data
    ? Math.max(1, Math.ceil(query.data.meta.total / PAGE_SIZE))
    : 1;

  return (
    <>
      <h1 style={{ margin: 0, fontSize: "20px" }}>Students</h1>

      <Card title="Filters" subtitle="Highest risk first">
        <div
          style={{
            display: "flex",
            gap: tokens.space(3),
            flexWrap: "wrap",
            alignItems: "end",
          }}
        >
          <Field label="Risk band">
            <select
              aria-label="Risk band"
              value={band}
              onChange={(event) => {
                setBand(event.target.value as BandKey | "");
                setPage(1);
              }}
              style={controlStyle}
            >
              <option value="">Any</option>
              <option value="critical">Critical</option>
              <option value="high">High</option>
              <option value="medium">Medium</option>
              <option value="low">Low</option>
            </select>
          </Field>
          <Field label="Trend">
            <select
              aria-label="Trend"
              value={direction}
              onChange={(event) => {
                setDirection(event.target.value as RiskDirection | "");
                setPage(1);
              }}
              style={controlStyle}
            >
              <option value="">Any</option>
              <option value="worsening">Rising</option>
              <option value="stable">Stable</option>
              <option value="improving">Falling</option>
              <option value="unknown">No trend yet</option>
            </select>
          </Field>
          <Field label="Student code">
            <input
              aria-label="Student code"
              value={search}
              onChange={(event) => {
                setSearch(event.target.value);
                setPage(1);
              }}
              placeholder="S-…"
              style={controlStyle}
            />
          </Field>
        </div>
      </Card>

      <Card
        title={
          query.data ? `${query.data.meta.total.toLocaleString()} students` : "Students"
        }
      >
        {query.isLoading && <Spinner />}
        {query.error && <ErrorNotice error={query.error} />}
        {query.data && query.data.data.length === 0 && (
          <EmptyState message="No students match these filters." />
        )}
        {query.data && query.data.data.length > 0 && (
          <>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "13px" }}>
              <thead>
                <tr style={{ textAlign: "left", color: tokens.color.muted }}>
                  <th style={cellStyle}>Student</th>
                  <th style={cellStyle}>Module</th>
                  <th style={cellStyle}>Checkpoint</th>
                  <th style={cellStyle}>Risk</th>
                  <th style={cellStyle}>Trend</th>
                </tr>
              </thead>
              <tbody>
                {query.data.data.map((row) => (
                  <tr
                    key={row.student_code}
                    style={{ borderTop: `1px solid ${tokens.color.border}` }}
                  >
                    <td style={cellStyle}>
                      <Link
                        to={`/students/${row.student_code}`}
                        style={{
                          color: tokens.color.accent,
                          fontFamily: tokens.font.mono,
                        }}
                      >
                        {row.student_code}
                      </Link>
                    </td>
                    <td style={cellStyle}>{row.code_module}</td>
                    <td style={cellStyle}>Day {row.checkpoint_day}</td>
                    <td style={cellStyle}>
                      <RiskBadge band={row.band} probability={row.probability} />
                    </td>
                    <td style={cellStyle}>
                      <DirectionBadge direction={row.risk_direction} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <nav
              style={{
                display: "flex",
                gap: tokens.space(3),
                alignItems: "center",
                marginTop: tokens.space(4),
                fontSize: "13px",
              }}
            >
              <button
                onClick={() => setPage((current) => Math.max(1, current - 1))}
                disabled={page === 1}
                style={pagerStyle}
              >
                Previous
              </button>
              <span style={{ color: tokens.color.muted }}>
                Page {page} of {totalPages}
              </span>
              <button
                onClick={() => setPage((current) => Math.min(totalPages, current + 1))}
                disabled={page >= totalPages}
                style={pagerStyle}
              >
                Next
              </button>
            </nav>
          </>
        )}
      </Card>
    </>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label
      style={{ fontSize: "12px", color: tokens.color.muted, display: "grid", gap: 4 }}
    >
      {label}
      {children}
    </label>
  );
}

const controlStyle: React.CSSProperties = {
  padding: tokens.space(2),
  border: `1px solid ${tokens.color.border}`,
  borderRadius: "6px",
  fontSize: "13px",
  minWidth: "150px",
};
const cellStyle: React.CSSProperties = {
  padding: `${tokens.space(2)} ${tokens.space(2)} ${tokens.space(2)} 0`,
  fontWeight: 400,
};
const pagerStyle: React.CSSProperties = {
  padding: `${tokens.space(1.5)} ${tokens.space(3)}`,
  border: `1px solid ${tokens.color.border}`,
  borderRadius: "6px",
  background: tokens.color.surface,
  fontSize: "13px",
  cursor: "pointer",
};
