/**
 * Component tests.
 *
 * These are weighted toward the things this UI could get *harmfully* wrong rather
 * than toward coverage: that a risk figure never appears without its disclaimer,
 * that bands are not distinguished by colour alone, that the SHAP panel is not
 * headed "reasons", that contextual factors do not masquerade as actionable ones,
 * and that a 403 reads as an access decision rather than a fault.
 *
 * Formatting is asserted with realistic values, because a fixture at 0.5 would
 * never catch a component that renders 0.089 as "0%".
 */

import { describe, expect, it } from "vitest";
import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { renderWithProviders } from "./render";
import { ANALYST, DISCLAIMER, server } from "./server";
import { DashboardPage } from "../pages/DashboardPage";
import { StudentsPage } from "../pages/StudentsPage";
import { StudentProfilePage } from "../pages/StudentProfilePage";
import { AlertsPage } from "../pages/AlertsPage";
import { AnalyticsPage } from "../pages/AnalyticsPage";
import { ModelPage } from "../pages/ModelPage";
import { RiskBadge } from "../components/primitives";
import { BAND_META, formatRisk } from "../lib/format";
import { Route, Routes } from "react-router-dom";

// ---------------------------------------------------------------------------
// Accessibility of the risk encoding
// ---------------------------------------------------------------------------

describe("risk badge", () => {
  it("never distinguishes a band by colour alone", () => {
    // Roughly 1 in 12 men has a colour vision deficiency, and this UI drives
    // decisions about people. Every band must carry a text label.
    for (const band of ["low", "medium", "high", "critical"] as const) {
      const { unmount } = renderWithProviders(<RiskBadge band={band} />);
      expect(screen.getByText(BAND_META[band].label)).toBeInTheDocument();
      unmount();
    }
  });

  it("formats single-digit risk without rounding it away", () => {
    // 0.0089 must not render as "0%".
    renderWithProviders(<RiskBadge band="low" probability={0.0089} />);
    expect(screen.getByText("0.9%")).toBeInTheDocument();
  });

  it("formats risk to one decimal place", () => {
    expect(formatRisk(0.1461)).toBe("14.6%");
    expect(formatRisk(0.029)).toBe("2.9%");
  });
});

// ---------------------------------------------------------------------------
// Dashboard
// ---------------------------------------------------------------------------

describe("dashboard", () => {
  it("shows band counts that sum to the cohort", async () => {
    renderWithProviders(<DashboardPage />);
    await waitFor(() => expect(screen.getByText("7,848")).toBeInTheDocument());
    for (const count of ["4,986", "1,561", "693", "608"]) {
      expect(screen.getByText(count)).toBeInTheDocument();
    }
  });

  it("explains that band cutoffs come from staffing capacity", async () => {
    // The framing is the point of this panel: a band count without it invites
    // reading the cutoff as a property of the students.
    renderWithProviders(<DashboardPage />);
    await waitFor(() =>
      expect(screen.getByText(/alert budget/i)).toBeInTheDocument(),
    );
    expect(screen.getByText(/5\.0%/)).toBeInTheDocument();
    expect(screen.getByText(/close to half the cohort/i)).toBeInTheDocument();
  });

  it("warns that band boundaries are not sharp", async () => {
    renderWithProviders(<DashboardPage />);
    await waitFor(() =>
      expect(screen.getByText(/share the exact cutoff probability/i)).toBeInTheDocument(),
    );
  });

  it("carries a disclaimer alongside the risk counts", async () => {
    renderWithProviders(<DashboardPage />);
    await waitFor(() =>
      expect(screen.getByText(/not a judgement about any student/i)).toBeInTheDocument(),
    );
  });

  it("warns when the bands are not calibrated", async () => {
    server.use(
      http.get("/api/v1/dashboard/statistics", () =>
        HttpResponse.json({
          total_students: 10,
          scored_checkpoints: 20,
          band_counts: [{ band: "low", label: "Low", count: 10, share: 1 }],
          worsening_count: 0,
          open_alerts: 0,
          model_version: "x",
          is_calibrated: false,
          alert_budget: 0.05,
        }),
      ),
    );
    renderWithProviders(<DashboardPage />);
    await waitFor(() =>
      expect(screen.getByRole("alert")).toHaveTextContent(/placeholders/i),
    );
  });
});

// ---------------------------------------------------------------------------
// Students list
// ---------------------------------------------------------------------------

describe("students list", () => {
  it("lists students highest risk first", async () => {
    renderWithProviders(<StudentsPage />);
    await waitFor(() =>
      expect(screen.getByText("S-AAA000000001")).toBeInTheDocument(),
    );
    const rows = screen.getAllByRole("row").slice(1);
    expect(rows[0]).toHaveTextContent("S-AAA000000001");
  });

  it("shows no trend rather than claiming stability for a single checkpoint", async () => {
    // One checkpoint is not a trend, and calling it "stable" would assert
    // something nobody has observed.
    renderWithProviders(<StudentsPage />);
    await waitFor(() => expect(screen.getByText("No trend yet")).toBeInTheDocument());
  });

  it("filters by band", async () => {
    renderWithProviders(<StudentsPage />);
    await waitFor(() => expect(screen.getByText("S-AAA000000003")).toBeInTheDocument());

    await userEvent.selectOptions(screen.getByLabelText("Risk band"), "critical");
    await waitFor(() =>
      expect(screen.queryByText("S-AAA000000003")).not.toBeInTheDocument(),
    );
    expect(screen.getByText("S-AAA000000001")).toBeInTheDocument();
  });

  it("shows an empty state rather than a blank table", async () => {
    server.use(
      http.get("/api/v1/students", () =>
        HttpResponse.json({
          data: [],
          meta: { total: 0, page: 1, page_size: 25, model_version: "x" },
        }),
      ),
    );
    renderWithProviders(<StudentsPage />);
    await waitFor(() =>
      expect(screen.getByText(/no students match/i)).toBeInTheDocument(),
    );
  });
});

// ---------------------------------------------------------------------------
// Student profile: the page with the most responsibility
// ---------------------------------------------------------------------------

function renderProfile() {
  return renderWithProviders(
    <Routes>
      <Route path="/students/:code" element={<StudentProfilePage />} />
    </Routes>,
    { route: "/students/S-AAA000000001" },
  );
}

describe("student profile", () => {
  it("shows the trajectory and the current risk", async () => {
    renderProfile();
    await waitFor(() => expect(screen.getByText("S-AAA000000001")).toBeInTheDocument());
    expect(screen.getByText("14.6%")).toBeInTheDocument();
    expect(screen.getByText("Rising")).toBeInTheDocument();
    expect(await screen.findByTestId("plot")).toBeInTheDocument();
  });

  it("heads the explanation panel with weighting, not reasons", async () => {
    // Phase 7 measured top-3 factor agreement among near-identical students at
    // 0.90 for the high-risk group -- good, not perfect. Calling these "the
    // reasons" would overstate what they are.
    renderProfile();
    await waitFor(() =>
      expect(screen.getByText("Factors the model weighted")).toBeInTheDocument(),
    );
    expect(screen.queryByText(/^reasons$/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/why this student will/i)).not.toBeInTheDocument();
  });

  it("separates contextual contributors from actionable ones", async () => {
    // "Point reached in the course" is the largest global contributor and is not
    // something a counsellor can act on; it must not head the actionable list.
    renderProfile();
    await waitFor(() =>
      expect(screen.getByText("Context (not actionable)")).toBeInTheDocument(),
    );
    const risky = screen.getByText("Weighted toward higher risk").closest("div");
    expect(risky).not.toBeNull();
    expect(
      within(risky as HTMLElement).queryByText("Point reached in the course"),
    ).not.toBeInTheDocument();
  });

  it("shows protective factors as well as risk factors", async () => {
    // An explanation made only of negatives misrepresents a student who is doing
    // several things well.
    renderProfile();
    await waitFor(() =>
      expect(screen.getByText("Weighted toward lower risk")).toBeInTheDocument(),
    );
    expect(screen.getByText("Average assessment score")).toBeInTheDocument();
  });

  it("states that attributions are not an effect on the probability", async () => {
    renderProfile();
    await waitFor(() =>
      expect(screen.getByText(/pre-calibration score/i)).toBeInTheDocument(),
    );
  });

  it("shows the disclaimer with the explanation", async () => {
    renderProfile();
    await waitFor(() => expect(screen.getByText(DISCLAIMER)).toBeInTheDocument());
  });

  it("presents interventions as unsent offers", async () => {
    renderProfile();
    await waitFor(() =>
      expect(screen.getByText("Assessment planning support")).toBeInTheDocument(),
    );
    expect(screen.getByText(/nothing below has been sent/i)).toBeInTheDocument();
  });

  it("requires an assignee before an intervention can be assigned", async () => {
    // Assignment is the human act the system requires; the button cannot be a
    // one-click accident with nobody named.
    renderProfile();
    const button = await screen.findByRole("button", { name: "Assign" });
    expect(button).toBeDisabled();

    await userEvent.type(screen.getByLabelText(/assign to/i), "counsellor");
    expect(button).toBeEnabled();
  });

  it("records an assignment and lists it", async () => {
    renderProfile();
    await screen.findByRole("button", { name: "Assign" });
    await userEvent.type(screen.getByLabelText(/assign to/i), "counsellor");

    server.use(
      http.get("/api/v1/students/:code/interventions", () =>
        HttpResponse.json([
          {
            id: 1,
            student_code: "S-AAA000000001",
            intervention_key: "assessment_planning_support",
            title: "Assessment planning support",
            status: "assigned",
            assigned_to: "counsellor",
            assigned_by: "counsellor",
            note: null,
            created_at: "2026-09-27T09:00:00Z",
          },
        ]),
      ),
    );
    await userEvent.click(screen.getByRole("button", { name: "Assign" }));
    await waitFor(() => expect(screen.getByText(/assigned to counsellor/i)).toBeInTheDocument());
  });

  it("says there is no trend when only one checkpoint is scored", async () => {
    server.use(
      http.get("/api/v1/students/:code", () =>
        HttpResponse.json({
          student_code: "S-AAA000000001",
          code_module: "CCC",
          code_presentation: "2014J",
          risk: {
            probability: 0.04,
            band: {
              key: "medium",
              label: "Medium",
              min_probability: 0.03,
              color: "#F9A825",
              action: "Passive monitoring.",
            },
            checkpoint_day: 30,
            model_version: "x",
            is_calibrated: true,
            disclaimer: DISCLAIMER,
          },
          risk_direction: "unknown",
          trajectory: [{ checkpoint_day: 30, probability: 0.04, band: "medium" }],
          engagement: {},
          assessment: {},
          context: {},
        }),
      ),
    );
    renderProfile();
    await waitFor(() =>
      expect(screen.getByText(/no trend to show yet/i)).toBeInTheDocument(),
    );
  });

  it("shows a 404 as not found rather than a crash", async () => {
    server.use(
      http.get("/api/v1/students/:code", () =>
        HttpResponse.json({ detail: "Student not found" }, { status: 404 }),
      ),
    );
    renderProfile();
    await waitFor(() =>
      expect(screen.getByText(/student not found/i)).toBeInTheDocument(),
    );
  });
});

// ---------------------------------------------------------------------------
// Access control presentation
// ---------------------------------------------------------------------------

describe("forbidden responses", () => {
  it("presents a 403 as an access decision, not a failure", async () => {
    // An analyst hitting a student page should be told the restriction is
    // deliberate. "Error" would suggest something is broken.
    server.use(
      http.get("/api/v1/auth/me", () => HttpResponse.json(ANALYST)),
      http.get("/api/v1/students/:code", () =>
        HttpResponse.json(
          { detail: "Role 'analyst' may see aggregates only, not individual students." },
          { status: 403 },
        ),
      ),
    );
    renderProfile();
    await waitFor(() =>
      expect(screen.getByText(/not available for your role/i)).toBeInTheDocument(),
    );
    expect(screen.getByText(/aggregates only/i)).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// Alerts
// ---------------------------------------------------------------------------

describe("alerts", () => {
  it("shows the change in risk, not only the current level", async () => {
    // A student steady at 30% needs less attention than one who moved from 10%.
    renderWithProviders(<AlertsPage />);
    await waitFor(() => expect(screen.getByText("6.9% → 14.6%")).toBeInTheDocument());
  });

  it("says that acknowledging is not the same as offering support", async () => {
    renderWithProviders(<AlertsPage />);
    await waitFor(() =>
      expect(screen.getByText(/does not offer the student anything/i)).toBeInTheDocument(),
    );
  });

  it("removes an acknowledged alert from the open queue", async () => {
    // The queue defaults to open alerts only, so acknowledging makes the row
    // leave rather than relabel. An earlier version of this test asserted the
    // label appeared, which was asserting the wrong behaviour.
    renderWithProviders(<AlertsPage />);
    await waitFor(() => expect(screen.getByText("S-AAA000000001")).toBeInTheDocument());

    const buttons = screen.getAllByRole("button", { name: "Acknowledge" });
    await userEvent.click(buttons[0]!);
    await waitFor(() =>
      expect(screen.queryByText("S-AAA000000001")).not.toBeInTheDocument(),
    );
    // The other open alert is untouched.
    expect(screen.getByText("S-AAA000000002")).toBeInTheDocument();
  });

  it("shows acknowledged alerts as reviewed when they are included", async () => {
    renderWithProviders(<AlertsPage />);
    const buttons = await screen.findAllByRole("button", { name: "Acknowledge" });
    await userEvent.click(buttons[0]!);

    await userEvent.click(screen.getByLabelText(/include acknowledged/i));
    await waitFor(() => expect(screen.getByText("Acknowledged")).toBeInTheDocument());
  });

  it("labels a first checkpoint rather than showing a misleading change", async () => {
    renderWithProviders(<AlertsPage />);
    await waitFor(() => expect(screen.getByText("first checkpoint")).toBeInTheDocument());
  });
});

// ---------------------------------------------------------------------------
// Analytics and model card
// ---------------------------------------------------------------------------

describe("analytics", () => {
  it("renders the three charts", async () => {
    renderWithProviders(<AnalyticsPage />);
    await waitFor(() => expect(screen.getAllByTestId("plot").length).toBe(3));
  });

  it("notes that global importance describes the model, not causes", async () => {
    renderWithProviders(<AnalyticsPage />);
    await waitFor(() =>
      expect(
        screen.getByText(/describes how the model behaves, not what causes withdrawal/i),
      ).toBeInTheDocument(),
    );
  });

  it("only offers allowlisted scatter fields", async () => {
    renderWithProviders(<AnalyticsPage />);
    const select = await screen.findByLabelText("Feature");
    const values = Array.from(select.querySelectorAll("option")).map(
      (option) => (option as HTMLOptionElement).value,
    );
    // The API rejects anything outside its allowlist, and the label is
    // deliberately not offered.
    expect(values).not.toContain("date_unregistration");
    expect(values).not.toContain("label");
    expect(values).toContain("clicks_28d");
  });
});

describe("model card", () => {
  it("shows limitations above the metrics", async () => {
    // Ordering is the point: showing the number first invites reading it and
    // skipping the caveats.
    renderWithProviders(<ModelPage />);
    await waitFor(() => expect(screen.getByText("Read this first")).toBeInTheDocument());

    const body = document.body.textContent ?? "";
    expect(body.indexOf("Read this first")).toBeLessThan(body.indexOf("Performance"));
  });

  it("states the headline metrics with their base rate", async () => {
    renderWithProviders(<ModelPage />);
    await waitFor(() => expect(screen.getByText("0.0885")).toBeInTheDocument());
    expect(screen.getByText("0.0290")).toBeInTheDocument();
  });

  it("explains why accuracy is not reported", async () => {
    renderWithProviders(<ModelPage />);
    await waitFor(() =>
      expect(screen.getByText(/97% accurate and useless/i)).toBeInTheDocument(),
    );
  });

  it("says the bands are a policy choice rather than a validated scale", async () => {
    // Deliberately getAllByText: the point is made in both the limitations list
    // and the bands panel, and that consistency is wanted rather than a problem.
    renderWithProviders(<ModelPage />);
    await waitFor(() =>
      expect(screen.getAllByText(/not a validated risk scale/i).length).toBeGreaterThan(1),
    );
  });

  it("reports the imbalance strategy and why SMOTE was rejected", async () => {
    renderWithProviders(<ModelPage />);
    await waitFor(() =>
      expect(screen.getByText(/no real student could have/i)).toBeInTheDocument(),
    );
  });
});
