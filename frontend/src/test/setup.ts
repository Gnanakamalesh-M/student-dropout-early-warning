import "@testing-library/jest-dom/vitest";
import { createElement } from "react";
import { afterAll, afterEach, beforeAll, vi } from "vitest";
import { resetAlertState, server } from "./server";

// Stubbed at the Chart boundary rather than at the Plotly import, so the
// basic-distribution binding in components/Chart.tsx is not bypassed in a way
// that could let a test pass against a different Plotly build than production.
// Plotly needs a real canvas and measures layout, neither of which jsdom
// provides. The charts are stubbed so component tests assert the surrounding
// behaviour — filters, labels, disclaimers — rather than pixel output, which is
// what these tests are for. `trace` count is exposed so a test can at least check
// the right number of series was passed.
vi.mock("../components/Chart", () => ({
  Chart: ({ data }: { data: unknown[] }) =>
    createElement(
      "div",
      { "data-testid": "plot", "data-traces": String(data.length) },
      "chart",
    ),
}));

// `onUnhandledRequest: "error"` on purpose: a request with no handler is almost
// always a test calling something it did not mean to, and failing loudly beats a
// silent network error surfacing as an unrelated assertion failure.
beforeAll(() => server.listen({ onUnhandledRequest: "error" }));
afterEach(() => {
  server.resetHandlers();
  resetAlertState();
  localStorage.clear();
});
afterAll(() => server.close());
