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
  const [mode, setMode] = useState<"stream" | "poll">("stream");
  const handleRefresh = useEffectEvent(onRefresh);

  useEffect(() => {
    let pollHandle: ReturnType<typeof setInterval> | null = null;
    let stream: EventSource | null = null;

    const stopPolling = () => {
      if (pollHandle) {
        clearInterval(pollHandle);
        pollHandle = null;
      }
    };

    const startPolling = () => {
      if (pollHandle) {
        return;
      }
      setMode("poll");
      pollHandle = setInterval(() => {
        handleRefresh();
      }, pollIntervalMs);
    };

    if (typeof window.EventSource === "undefined") {
      startPolling();
      return stopPolling;
    }

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
      stopPolling();
      stream?.close();
    };
  }, [handleRefresh, pollIntervalMs, streamUrl]);

  return mode;
}
