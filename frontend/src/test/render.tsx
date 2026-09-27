/** Test harness: router, query client and auth, wired as the app wires them. */

import { render } from "@testing-library/react";
import type { ReactElement } from "react";
import { MemoryRouter } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { AuthProvider } from "../hooks/useAuth";
import { setToken } from "../api/client";

export function renderWithProviders(
  ui: ReactElement,
  { route = "/", authenticated = true }: { route?: string; authenticated?: boolean } = {},
) {
  if (authenticated) setToken("test-token");

  // Retries off: a failing fixture should fail the test immediately rather than
  // after three backoffs, and no test here is checking retry behaviour.
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: 0 } },
  });

  return {
    queryClient,
    ...render(
      <QueryClientProvider client={queryClient}>
        <MemoryRouter initialEntries={[route]}>
          <AuthProvider>{ui}</AuthProvider>
        </MemoryRouter>
      </QueryClientProvider>,
    ),
  };
}
