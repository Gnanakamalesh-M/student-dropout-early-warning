# Dashboard

React 18 + TypeScript + Vite, in `frontend/`. Seven pages over the API.

```bash
cd frontend && npm ci
npm run dev          # proxies /api to localhost:8000
npm run lint && npm run typecheck && npm test && npm run build
```

---

## Pages

| Route | Role | What it carries |
|---|---|---|
| `/login` | — | Demo account buttons (portfolio deployment, no real students) |
| `/dashboard` | any | Band counts, rising-risk count, recent alerts, capacity framing |
| `/students` | admin, counsellor | Filterable list, highest risk first |
| `/students/:code` | admin, counsellor | Trajectory, SHAP factors, suggested support, assignment |
| `/alerts` | admin, counsellor | Queue with change-in-risk, acknowledge |
| `/analytics` | any | Distribution, feature-vs-risk scatter, global importance |
| `/model` | any | Model card with limitations first |

Navigation is filtered by the capabilities `/auth/me` reports, so an analyst never
sees a link that would 403. The API still enforces it — hiding a link is a
courtesy, not a control.

---

## Decisions that shape the UI

### A probability never appears without its provenance

`RiskAssessment` in `src/api/types.ts` requires `model_version`, `is_calibrated`
and `disclaimer` — no `?` on any of them. The types are hand-written rather than
generated for exactly this reason: a generator would have emitted optional fields
and quietly permitted what the backend schema was built to prevent.

### Risk bands never rely on colour alone

Every band renders a symbol (`● ◆ ▲ ■`) and a text label as well as a colour.
Roughly 1 in 12 men has a colour vision deficiency, and this UI drives decisions
about people. A test asserts the label text is present for all four bands, so a
later "simplification" to a coloured dot fails rather than shipping.

### The explanation panel is headed "Factors the model weighted"

Not "Reasons". Phase 7 measured top-3 factor agreement among near-identical
students at 0.90 for the high-risk group — good, not perfect. A test asserts the
word "reasons" does not head that panel.

Contextual factors are listed separately and de-emphasised. The largest single
global contributor is how far through the course a student is: real signal, but
nothing a counsellor can act on, and letting it head the list would waste their
attention. A test asserts it does not appear in the actionable list.

Protective factors are shown, because an explanation made only of negatives
misrepresents a student who is doing several things well.

### Interventions are unsent offers

The card states that nothing has been sent, the button says **Assign**, and it is
disabled until someone is named — assignment is the human act the system requires,
so it cannot be a one-click accident with nobody accountable.

### A 403 is not an error state

`ErrorNotice` renders a forbidden response as "Not available for your role" in a
neutral colour, and surfaces the API's message, which names the restriction. An
analyst hitting a student page should be told the limit is deliberate, not shown a
red failure.

### Limitations come before metrics on the model card

Ordering is the point. The headline is PR-AUC 0.089 against a 2.9% base rate — a
real improvement over any single heuristic, and still means most students contacted
were not about to withdraw. Metrics first invites reading the number and skipping
the caveats. A test asserts the ordering in the DOM.

### One checkpoint is not a trend

`risk_direction: "unknown"` renders as "No trend yet", never "Stable". Calling it
stable would assert something nobody has observed. Tested.

---

## Bundle size

Plotly dominates, and the first build showed how much:

| | Bundle | Gzipped |
|---|---:|---:|
| `react-plotly.js` default import | 5,067 kB | 1,564 kB |
| Bound to `plotly.js-basic-dist-min` | 1,353 kB | 457 kB |
| Split into a separate chunk | 255 kB app + 1,097 kB plotly | 80 kB + 378 kB |

`react-plotly.js` imports the **full** Plotly build by default, ignoring the basic
distribution even when it is the installed dependency. `src/components/Chart.tsx`
builds the component from `react-plotly.js/factory` against the basic dist
instead, and every chart goes through that module so the binding cannot be
bypassed by accident.

One consequence found by the build: the basic distribution has no WebGL traces, so
the scatter plot uses `scatter` rather than `scattergl`. At a few thousand points
SVG is fast enough.

Plotly is in its own Rollup chunk because it changes only when the dependency does,
so it caches independently of app code. It cannot be shrunk further without
dropping the charts.

---

## Testing

37 component tests, Vitest + Testing Library + MSW.

The tests are weighted toward what this UI could get *harmfully* wrong rather than
toward coverage: that a risk figure never appears without its disclaimer, that
bands carry text labels, that the SHAP panel is not headed "reasons", that
contextual factors do not masquerade as actionable ones, and that a 403 reads as an
access decision.

Two details worth naming:

**Fixtures use realistic values.** A 2.9% base rate, single-digit risk
percentages, the actual disclaimer wording. A mock at `0.5` would never catch a
component that renders `0.0089` as `0%` — and there is a test for exactly that.

**Charts are stubbed at the `Chart` boundary**, not at the Plotly import, so a test
cannot pass against a different Plotly build than production uses. jsdom has no
canvas and does not measure layout, so the stub asserts trace counts and the
surrounding behaviour instead of pixels.

### A test that was asserting the wrong thing

The acknowledge test originally expected an "Acknowledged" label to appear on the
row. The queue defaults to open alerts only, so acknowledging makes the row *leave*
rather than relabel — the app was right and the test was wrong. It now asserts the
row disappears while the other open alert is untouched, with a separate test for
the label when acknowledged alerts are included. The MSW handler was also made
stateful, since a stateless mock meant the refetch returned an unchanged list and
the assertion could never have been meaningful either way.
