/**
 * `react-plotly.js` ships no bundled types, and the basic distribution avoids
 * pulling the full Plotly bundle into the app. Only what `components/Chart.tsx`
 * needs is declared, rather than installing types for a far wider API.
 */
declare module "react-plotly.js/factory" {
  import type { ComponentType } from "react";

  interface PlotProps {
    data: unknown[];
    layout?: Record<string, unknown>;
    config?: Record<string, unknown>;
    style?: React.CSSProperties;
    className?: string;
  }

  export default function createPlotlyComponent(
    plotly: unknown,
  ): ComponentType<PlotProps>;
}

declare module "plotly.js-basic-dist-min" {
  const Plotly: unknown;
  export default Plotly;
}
