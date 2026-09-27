import { useState } from "react";
import { Navigate, useLocation } from "react-router-dom";
import { useAuth } from "../hooks/useAuth";
import { card, tokens } from "../lib/styles";

/**
 * Demo accounts are listed because this is a portfolio deployment over public
 * research data with no real students in it. A production build would not print
 * credentials on the login screen.
 */
const DEMO = [
  ["counsellor", "counsellor-demo-password", "sees individual students"],
  ["analyst", "analyst-demo-password", "aggregates only"],
  ["admin", "admin-demo-password", "everything, plus the audit log"],
] as const;

export function LoginPage() {
  const { user, login, error } = useAuth();
  const location = useLocation() as { state?: { from?: string } };
  const [username, setUsername] = useState("counsellor");
  const [password, setPassword] = useState("");
  const [submitting, setSubmitting] = useState(false);

  if (user) return <Navigate to={location.state?.from ?? "/dashboard"} replace />;

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setSubmitting(true);
    try {
      await login(username, password);
    } catch {
      /* the error surfaces through the auth context */
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div
      style={{
        minHeight: "100vh",
        display: "grid",
        placeItems: "center",
        background: tokens.color.background,
        fontFamily: tokens.font.body,
        color: tokens.color.text,
        padding: tokens.space(4),
      }}
    >
      <form
        onSubmit={submit}
        style={{ ...card, width: "min(400px, 100%)", display: "grid", gap: tokens.space(3) }}
      >
        <div>
          <h1 style={{ margin: 0, fontSize: "18px" }}>Student Early-Warning</h1>
          <p
            style={{
              margin: `${tokens.space(1)} 0 0`,
              fontSize: "13px",
              color: tokens.color.muted,
            }}
          >
            Sign in to continue.
          </p>
        </div>

        <label style={{ fontSize: "13px", display: "grid", gap: 4 }}>
          Username
          <input
            value={username}
            onChange={(event) => setUsername(event.target.value)}
            autoComplete="username"
            style={inputStyle}
          />
        </label>
        <label style={{ fontSize: "13px", display: "grid", gap: 4 }}>
          Password
          <input
            type="password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            autoComplete="current-password"
            style={inputStyle}
          />
        </label>

        {error && (
          <p role="alert" style={{ margin: 0, fontSize: "13px", color: "#B71C1C" }}>
            {error}
          </p>
        )}

        <button
          type="submit"
          disabled={submitting}
          style={{
            padding: tokens.space(2.5),
            fontSize: "14px",
            fontWeight: 600,
            borderRadius: "6px",
            border: "none",
            background: tokens.color.accent,
            color: "#fff",
            cursor: submitting ? "wait" : "pointer",
          }}
        >
          {submitting ? "Signing in…" : "Sign in"}
        </button>

        <div style={{ fontSize: "12px", color: tokens.color.muted, lineHeight: 1.7 }}>
          <strong>Demo accounts</strong>
          {DEMO.map(([name, pass, note]) => (
            <div key={name}>
              <button
                type="button"
                onClick={() => {
                  setUsername(name);
                  setPassword(pass);
                }}
                style={{
                  background: "none",
                  border: "none",
                  padding: 0,
                  color: tokens.color.accent,
                  cursor: "pointer",
                  fontFamily: tokens.font.mono,
                  fontSize: "12px",
                }}
              >
                {name}
              </button>{" "}
              — {note}
            </div>
          ))}
        </div>
      </form>
    </div>
  );
}

const inputStyle: React.CSSProperties = {
  padding: tokens.space(2),
  border: `1px solid ${tokens.color.border}`,
  borderRadius: "6px",
  fontSize: "14px",
};
