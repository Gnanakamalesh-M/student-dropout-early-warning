/**
 * Plotly, bound to the basic distribution.
 *
 * `react-plotly.js` imports the full `plotly.js` by default, which pulled a
 * 5.07 MB bundle (1.56 MB gzipped) for three chart types. Building the component
 * from the factory against `plotly.js-basic-dist-min` keeps only the scatter and
 * bar traces this app uses.
 *
 * All charts go through this module rather than importing Plotly directly, so the
 * binding cannot be bypassed by accident.
 */

import createPlotlyComponent from "react-plotly.js/factory";
import Plotly from "plotly.js-basic-dist-min";

// No cast needed: the factory declaration in react-plotly.d.ts accepts unknown,
// which is the honest type for an untyped runtime bundle.
const Plot = createPlotlyComponent(Plotly);

export interface ChartProps {
  data: unknown[];
  layout?: Record<string, unknown>;
  config?: Record<string, unknown>;
  style?: React.CSSProperties;
}

export function Chart({ data, layout, config, style }: ChartProps) {
  return (
    <Plot
      data={data}
      layout={layout}
      config={{ displayModeBar: false, responsive: true, ...config }}
      style={{ width: "100%", ...style }}
    />
  );
}
