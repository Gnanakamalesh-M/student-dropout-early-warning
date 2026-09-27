/**
 * Auth context.
 *
 * Capabilities come from the server (`/auth/me`), not from decoding the token
 * client-side. The UI hides what a role cannot do, but hiding is a courtesy: the
 * API enforces it, and a client that decided its own permissions would be
 * deciding something it does not own.
 */

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";
import type { ReactNode } from "react";
import { AUTH_EXPIRED_EVENT, api, getToken } from "../api/client";
import type { CurrentUser } from "../api/types";

interface AuthState {
  user: CurrentUser | null;
  loading: boolean;
  error: string | null;
  login: (username: string, password: string) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthState | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<CurrentUser | null>(null);
  const [loading, setLoading] = useState(Boolean(getToken()));
  const [error, setError] = useState<string | null>(null);

  const loadUser = useCallback(async () => {
    if (!getToken()) {
      setUser(null);
      setLoading(false);
      return;
    }
    setLoading(true);
    try {
      setUser(await api.me());
    } catch {
      setUser(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadUser();
  }, [loadUser]);

  useEffect(() => {
    // The client dispatches this on any 401, so an expired token returns the user
    // to the login screen rather than leaving every panel failing.
    const onExpired = () => {
      setUser(null);
      setError("Your session expired. Please sign in again.");
    };
    window.addEventListener(AUTH_EXPIRED_EVENT, onExpired);
    return () => window.removeEventListener(AUTH_EXPIRED_EVENT, onExpired);
  }, []);

  const login = useCallback(async (username: string, password: string) => {
    setError(null);
    try {
      await api.login(username, password);
      setUser(await api.me());
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Sign in failed");
      throw caught;
    }
  }, []);

  const logout = useCallback(() => {
    api.logout();
    setUser(null);
    setError(null);
  }, []);

  const value = useMemo(
    () => ({ user, loading, error, login, logout }),
    [user, loading, error, login, logout],
  );
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthState {
  const context = useContext(AuthContext);
  if (!context) throw new Error("useAuth must be used inside AuthProvider");
  return context;
}
