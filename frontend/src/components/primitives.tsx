/**
 * Shared UI primitives.
 *
 * The important one is `RiskBadge`. It renders a symbol and a text label as well
 * as a colour, because colour alone is not an accessible encoding and this UI
 * drives decisions about people. A test asserts the label text is present, so a
 * later "simplification" to a coloured dot fails rather than shipping.
 */

import type { ReactNode } from "react";
import type { BandKey, ImpactBand, RiskDirection } from "../api/types";
import {
  BAND_META,
  DIRECTION_META,
  IMPACT_META,
  formatRisk,
} from "../lib/format";
import { card, tokens } from "../lib/styles";

export function RiskBadge({
  band,
  probability,
}: {
  band: BandKey;
  probability?: number;
}) {
  const meta = BAND_META[band];
  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: tokens.space(1.5),
        padding: `${tokens.space(1)} ${tokens.space(2)}`,
        borderRadius: "999px",
        background: meta.background,
        color: meta.color,
        border: `1px solid ${meta.color}33`,
        fontSize: "13px",
        fontWeight: 600,
        whiteSpace: "nowrap",
      }}
    >
      <span aria-hidden="true">{meta.symbol}</span>
      <span>{meta.label}</span>
      {probability !== undefined && (
        <span style={{ fontWeight: 500 }}>{formatRisk(probability)}</span>
      )}
    </span>
  );
}

export function DirectionBadge({ direction }: { direction: RiskDirection }) {
  const meta = DIRECTION_META[direction];
  const color =
    meta.tone === "bad"
      ? BAND_META.critical.color
      : meta.tone === "good"
        ? BAND_META.low.color
        : tokens.color.muted;
  return (
    <span
      style={{ color, fontSize: "13px", display: "inline-flex", gap: tokens.space(1) }}
      title={
        direction === "unknown"
          ? "Only one checkpoint has been scored, so no trend can be shown."
          : undefined
      }
    >
      <span aria-hidden="true">{meta.symbol}</span>
      <span>{meta.label}</span>
    </span>
  );
}

export function ImpactBadge({ impact }: { impact: ImpactBand }) {
  const weight = IMPACT_META[impact].weight;
  return (
    <span
      style={{
        fontSize: "12px",
        fontWeight: 600,
        padding: `2px ${tokens.space(1.5)}`,
        borderRadius: "4px",
        background: weight === 3 ? "#FFEBEE" : weight === 2 ? "#FFF8E1" : "#F1F3F4",
        color: weight === 3 ? "#B71C1C" : weight === 2 ? "#8D6E00" : "#5F6368",
      }}
    >
      {impact}
    </span>
  );
}

export function Card({
  title,
  subtitle,
  children,
  actions,
}: {
  title?: string;
  subtitle?: string;
  children: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <section style={card}>
      {(title || actions) && (
        <header
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "flex-start",
            marginBottom: tokens.space(3),
            gap: tokens.space(3),
          }}
        >
          <div>
            {title && (
              <h2 style={{ margin: 0, fontSize: "15px", fontWeight: 600 }}>{title}</h2>
            )}
            {subtitle && (
              <p
                style={{
                  margin: `${tokens.space(1)} 0 0`,
                  fontSize: "13px",
                  color: tokens.color.muted,
                }}
              >
                {subtitle}
              </p>
            )}
          </div>
          {actions}
        </header>
      )}
      {children}
    </section>
  );
}

export function StatTile({
  label,
  value,
  hint,
  emphasis,
}: {
  label: string;
  value: string | number;
  hint?: string;
  emphasis?: BandKey;
}) {
  const meta = emphasis ? BAND_META[emphasis] : undefined;
  return (
    <div
      style={{
        ...card,
        padding: tokens.space(3),
        borderLeft: meta ? `4px solid ${meta.color}` : card.border,
      }}
    >
      <div style={{ fontSize: "12px", color: tokens.color.muted, marginBottom: 4 }}>
        {emphasis && <span aria-hidden="true">{meta?.symbol} </span>}
        {label}
      </div>
      <div style={{ fontSize: "26px", fontWeight: 650, lineHeight: 1.1 }}>{value}</div>
      {hint && (
        <div style={{ fontSize: "12px", color: tokens.color.muted, marginTop: 4 }}>
          {hint}
        </div>
      )}
    </div>
  );
}

/**
 * The disclaimer that must accompany any risk figure.
 *
 * Rendered from the value the API sends rather than hardcoded, so the wording has
 * one source of truth and the narrative-language test in the backend governs it.
 */
export function Disclaimer({ text }: { text: string }) {
  return (
    <p
      role="note"
      style={{
        margin: 0,
        padding: tokens.space(3),
        background: tokens.color.warningSoft,
        border: `1px solid ${tokens.color.warning}33`,
        borderRadius: tokens.radius,
        fontSize: "13px",
        lineHeight: 1.5,
        color: "#4A3B00",
      }}
    >
      {text}
    </p>
  );
}

export function UncalibratedWarning() {
  return (
    <p
      role="alert"
      style={{
        margin: 0,
        padding: tokens.space(3),
        background: "#FFEBEE",
        border: "1px solid #B71C1C33",
        borderRadius: tokens.radius,
        fontSize: "13px",
        color: "#7F1010",
      }}
    >
      Risk bands are placeholders and have not been calibrated against this
      cohort. Treat the band labels as provisional.
    </p>
  );
}

export function Spinner({ label = "Loading" }: { label?: string }) {
  return (
    <div
      role="status"
      aria-live="polite"
      style={{ padding: tokens.space(6), color: tokens.color.muted, fontSize: "13px" }}
    >
      {label}…
    </div>
  );
}

export function ErrorNotice({ error }: { error: unknown }) {
  const message =
    error instanceof Error ? error.message : "Something went wrong loading this view.";
  const forbidden =
    typeof error === "object" && error !== null && "isForbidden" in error
      ? Boolean((error as { isForbidden: unknown }).isForbidden)
      : false;
  return (
    <div
      role="alert"
      style={{
        padding: tokens.space(4),
        background: forbidden ? tokens.color.accentSoft : "#FFEBEE",
        border: `1px solid ${forbidden ? tokens.color.accent : "#B71C1C"}33`,
        borderRadius: tokens.radius,
        fontSize: "13px",
      }}
    >
      {/* A 403 is a deliberate access decision, not a fault, so it is not
          presented as an error state. */}
      <strong>{forbidden ? "Not available for your role" : "Could not load"}</strong>
      <div style={{ marginTop: 4, color: tokens.color.muted }}>{message}</div>
    </div>
  );
}

export function EmptyState({ message }: { message: string }) {
  return (
    <p style={{ color: tokens.color.muted, fontSize: "13px", padding: tokens.space(4) }}>
      {message}
    </p>
  );
}
