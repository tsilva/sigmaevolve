import { afterEach } from "vitest";

afterEach(() => {
  delete (globalThis as { EventSource?: unknown }).EventSource;
});
