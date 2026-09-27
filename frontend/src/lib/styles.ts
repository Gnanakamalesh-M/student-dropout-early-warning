/**
 * Design tokens.
 *
 * Inline styles with a shared token object rather than a CSS framework. The app
 * is small, and this keeps every value that matters — the band colours in
 * particular — in one place next to the accessibility rule that governs them.
 */

export const tokens = {
  color: {
    text: "#1A1A1A",
    muted: "#5F6368",
    border: "#DADCE0",
    surface: "#FFFFFF",
    background: "#F6F7F9",
    accent: "#1A4F8B",
    accentSoft: "#E8F0FA",
    warning: "#8D6E00",
    warningSoft: "#FFF8E1",
  },
  space: (n: number) => `${n * 4}px`,
  radius: "8px",
  font: {
    body: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
    mono: "ui-monospace, SFMono-Regular, Menlo, monospace",
  },
  shadow: "0 1px 3px rgba(0,0,0,0.08)",
} as const;

export const card: React.CSSProperties = {
  background: tokens.color.surface,
  border: `1px solid ${tokens.color.border}`,
  borderRadius: tokens.radius,
  padding: tokens.space(4),
  boxShadow: tokens.shadow,
};
