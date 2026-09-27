/** Formatting helpers shared across pages. */

import type { BandKey, ImpactBand, RiskDirection } from "../api/types";

/**
 * Risk band presentation.
 *
 * Every band carries a `symbol` and a `label` as well as a colour. Colour alone
 * is not an accessible encoding — roughly 1 in 12 men has a colour vision
 * deficiency, and this UI drives decisions about people. A test asserts no band
 * is distinguished by colour only.
 */
export const BAND_META: Record<
  BandKey,
  { label: string; color: string; background: string; symbol: string }
> = {
  low: { label: "Low", color: "#1B5E20", background: "#E8F5E9", symbol: "●" },
  medium: { label: "Medium", color: "#8D6E00", background: "#FFF8E1", symbol: "◆" },
  high: { label: "High", color: "#C55A00", background: "#FFF3E0", symbol: "▲" },
  critical: { label: "Critical", color: "#B71C1C", background: "#FFEBEE", symbol: "■" },
};

export const DIRECTION_META: Record<
  RiskDirection,
  { label: string; symbol: string; tone: "bad" | "good" | "neutral" }
> = {
  worsening: { label: "Rising", symbol: "↑", tone: "bad" },
  improving: { label: "Falling", symbol: "↓", tone: "good" },
  stable: { label: "Stable", symbol: "→", tone: "neutral" },
  // Not "stable": one checkpoint is no trend, and saying otherwise would claim
  // something nobody has observed.
  unknown: { label: "No trend yet", symbol: "–", tone: "neutral" },
};

export const IMPACT_META: Record<ImpactBand, { label: string; weight: number }> = {
  High: { label: "High impact", weight: 3 },
  Medium: { label: "Medium impact", weight: 2 },
  Low: { label: "Low impact", weight: 1 },
};

/** Risk as a percentage. One decimal, because these are single-digit values and
 * rounding 4.4% to 4% loses a meaningful amount of it. */
export function formatRisk(probability: number): string {
  return `${(probability * 100).toFixed(1)}%`;
}

export function formatShare(share: number): string {
  return `${(share * 100).toFixed(1)}%`;
}

/** Feature values, which range from counts to ratios to day offsets. */
export function formatFeatureValue(value: number | null): string {
  if (value === null || Number.isNaN(value)) return "no data";
  if (Number.isInteger(value)) return value.toLocaleString();
  return value.toFixed(2);
}

/** `days_since_last_activity` -> `Days since last activity`. */
export function humaniseFieldName(name: string): string {
  const spaced = name.replace(/_/g, " ");
  return spaced.charAt(0).toUpperCase() + spaced.slice(1);
}

export function formatDateTime(iso: string): string {
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? iso : date.toLocaleString();
}
