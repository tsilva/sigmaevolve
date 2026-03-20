// @vitest-environment jsdom

import { act, render, screen } from "@testing-library/react";
import React, { useState } from "react";

import { useTrackLiveUpdates } from "@/hooks/use-track-live-updates";

class MockEventSource {
  static instances: MockEventSource[] = [];

  readonly listeners = new Map<string, Set<() => void>>();
  onerror: (() => void) | null = null;
  closed = false;
  url: string;

  constructor(url: string) {
    this.url = url;
    MockEventSource.instances.push(this);
  }

  addEventListener(type: string, listener: () => void) {
    if (!this.listeners.has(type)) {
      this.listeners.set(type, new Set());
    }
    this.listeners.get(type)?.add(listener);
  }

  close() {
    this.closed = true;
  }

  emit(type: string) {
    this.listeners.get(type)?.forEach((listener) => listener());
  }
}

function Probe({ streamUrl }: { streamUrl: string }) {
  const [refreshCount, setRefreshCount] = useState(0);
  const mode = useTrackLiveUpdates({
    streamUrl,
    pollIntervalMs: 25,
    onRefresh: () => {
      setRefreshCount((count) => count + 1);
    },
  });

  return React.createElement(
    "div",
    null,
    React.createElement("span", { "data-testid": "mode" }, mode),
    React.createElement("span", { "data-testid": "count" }, refreshCount),
  );
}

describe("useTrackLiveUpdates", () => {
  it("falls back to polling on stream errors and returns to stream mode on reconnect", async () => {
    vi.useFakeTimers();
    MockEventSource.instances = [];
    (globalThis as { EventSource?: unknown }).EventSource = MockEventSource;

    render(React.createElement(Probe, { streamUrl: "/api/tracks/track_1/stream" }));
    const source = MockEventSource.instances[0];

    await act(async () => {
      source.emit("open");
    });
    expect(screen.getByTestId("mode").textContent).toBe("stream");

    await act(async () => {
      source.emit("refresh");
    });
    expect(screen.getByTestId("count").textContent).toBe("1");

    await act(async () => {
      source.onerror?.();
    });
    expect(screen.getByTestId("mode").textContent).toBe("poll");

    await act(async () => {
      await vi.advanceTimersByTimeAsync(26);
    });
    expect(screen.getByTestId("count").textContent).toBe("2");

    await act(async () => {
      source.emit("open");
    });
    expect(screen.getByTestId("mode").textContent).toBe("stream");

    vi.useRealTimers();
  });
});
