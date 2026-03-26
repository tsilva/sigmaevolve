"use client";

import { useEffect, useEffectEvent, useState } from "react";

type UseTrackLiveUpdatesOptions = {
  onRefresh: () => void;
  pollIntervalMs?: number;
  streamUrl: string;
};

export function useTrackLiveUpdates({
  onRefresh,
  pollIntervalMs = 15_000,
  streamUrl,
}: UseTrackLiveUpdatesOptions): "stream" | "poll" {
  // Default to stream mode and keep the refresh callback stable across renders.
  const [mode, setMode] = useState<"stream" | "poll">("stream");
  const handleRefresh = useEffectEvent(onRefresh);

  useEffect(() => {
    // Track both the fallback poller and the EventSource subscription locally.
    let pollHandle: ReturnType<typeof setInterval> | null = null;
    let stream: EventSource | null = null;

    // Stop the polling loop when the stream is healthy or the hook unmounts.
    const stopPolling = () => {
      if (pollHandle) {
        clearInterval(pollHandle);
        pollHandle = null;
      }
    };

    // Switch into polling mode only once when SSE is unavailable or fails.
    const startPolling = () => {
      if (pollHandle) {
        return;
      }
      setMode("poll");
      pollHandle = setInterval(() => {
        handleRefresh();
      }, pollIntervalMs);
    };

    // Fall back immediately when the browser does not support EventSource.
    if (typeof window.EventSource === "undefined") {
      startPolling();
      return stopPolling;
    }

    // Prefer live refresh events over polling when SSE can be opened.
    stream = new window.EventSource(streamUrl);
    stream.addEventListener("open", () => {
      setMode("stream");
      stopPolling();
    });
    stream.addEventListener("refresh", () => {
      handleRefresh();
    });
    stream.onerror = () => {
      startPolling();
    };

    return () => {
      // Tear down both the poller and the stream subscription on cleanup.
      stopPolling();
      stream?.close();
    };
  }, [handleRefresh, pollIntervalMs, streamUrl]);

  return mode;
}
