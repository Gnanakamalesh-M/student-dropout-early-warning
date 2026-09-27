/**
 * App shell: routing, navigation, and the login gate.
 *
 * Navigation is filtered by the capabilities the server reported, so an analyst
 * does not see links to pages that would return 403. The API still enforces it —
 * hiding a link is a courtesy, not a control.
 */

import { Navigate, NavLink, Route, Routes, useLocation } from "react-router-dom";
import type { ReactNode } from "react";
import { AuthProvider, useAuth } from "./hooks/useAuth";
import { Spinner } from "./components/primitives";
import { tokens } from "./lib/styles";
import { LoginPage } from "./pages/LoginPage";
import { DashboardPage } from "./pages/DashboardPage";
import { StudentsPage } from "./pages/StudentsPage";
import { StudentProfilePage } from "./pages/StudentProfilePage";
import { AnalyticsPage } from "./pages/AnalyticsPage";
import { AlertsPage } from "./pages/AlertsPage";
import { ModelPage } from "./pages/ModelPage";

interface NavItem {
  to: string;
  label: string;
  /** Whether the route needs individual-student access. */
  individual: boolean;
}

const NAV: NavItem[] = [
  { to: "/dashboard", label: "Dashboard", individual: false },
  { to: "/students", label: "Students", individual: true },
  { to: "/alerts", label: "Alerts", individual: true },
  { to: "/analytics", label: "Analytics", individual: false },
  { to: "/model", label: "Model", individual: false },
];

function RequireAuth({ children }: { children: ReactNode }) {
  const { user, loading } = useAuth();
  const location = useLocation();
  if (loading) return <Spinner label="Checking your session" />;
  if (!user) return <Navigate to="/login" state={{ from: location.pathname }} replace />;
  return <>{children}</>;
}

function Shell({ children }: { children: ReactNode }) {
  const { user, logout } = useAuth();
  const visible = NAV.filter(
    (item) => !item.individual || (user?.may_view_individuals ?? false),
  );

  return (
    <div
      style={{
        minHeight: "100vh",
        background: tokens.color.background,
        color: tokens.color.text,
        fontFamily: tokens.font.body,
      }}
    >
      <header
        style={{
          background: tokens.color.surface,
          borderBottom: `1px solid ${tokens.color.border}`,
          padding: `${tokens.space(3)} ${tokens.space(6)}`,
          display: "flex",
          alignItems: "center",
          gap: tokens.space(6),
          flexWrap: "wrap",
        }}
      >
        <strong style={{ fontSize: "15px" }}>Student Early-Warning</strong>
        <nav style={{ display: "flex", gap: tokens.space(4), flex: 1 }}>
          {visible.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              style={({ isActive }) => ({
                textDecoration: "none",
                fontSize: "14px",
                paddingBottom: 2,
                color: isActive ? tokens.color.accent : tokens.color.muted,
                fontWeight: isActive ? 600 : 500,
                borderBottom: isActive ? `2px solid ${tokens.color.accent}` : "2px solid transparent",
              })}
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
        <span style={{ fontSize: "13px", color: tokens.color.muted }}>
          {user?.username} · {user?.role}
        </span>
        <button
          onClick={logout}
          style={{
            fontSize: "13px",
            padding: `${tokens.space(1.5)} ${tokens.space(3)}`,
            borderRadius: "6px",
            border: `1px solid ${tokens.color.border}`,
            background: tokens.color.surface,
            cursor: "pointer",
          }}
        >
          Sign out
        </button>
      </header>
      <main
        style={{
          maxWidth: "1180px",
          margin: "0 auto",
          padding: tokens.space(6),
          display: "grid",
          gap: tokens.space(5),
        }}
      >
        {children}
      </main>
    </div>
  );
}

export function AppRoutes() {
  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route
        path="/*"
        element={
          <RequireAuth>
            <Shell>
              <Routes>
                <Route path="/" element={<Navigate to="/dashboard" replace />} />
                <Route path="/dashboard" element={<DashboardPage />} />
                <Route path="/students" element={<StudentsPage />} />
                <Route path="/students/:code" element={<StudentProfilePage />} />
                <Route path="/alerts" element={<AlertsPage />} />
                <Route path="/analytics" element={<AnalyticsPage />} />
                <Route path="/model" element={<ModelPage />} />
                <Route path="*" element={<Navigate to="/dashboard" replace />} />
              </Routes>
            </Shell>
          </RequireAuth>
        }
      />
    </Routes>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <AppRoutes />
    </AuthProvider>
  );
}
