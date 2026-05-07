import type {
  DashboardNotification,
} from "@/lib/types";

type Cleanup = () => void | Promise<void>;

type DashboardStreamOptions = {
  keepAliveMs?: number;
  signal?: AbortSignal;
  subscribe: (emit: (notification: DashboardNotification) => void) => Promise<Cleanup> | Cleanup;
  ttlMs?: number;
};

const encoder = new TextEncoder();

export function buildSseHeaders(): Headers {
  // Use no-buffer, no-cache headers so intermediaries do not stall the stream.
  return new Headers({
    "Cache-Control": "no-store, no-transform",
    Connection: "keep-alive",
    "Content-Type": "text/event-stream; charset=utf-8",
    "X-Accel-Buffering": "no",
  });
}

export function formatSseComment(message: string): string {
  // Prefix keepalive comments with the SSE comment marker.
  return `: ${message}\n\n`;
}

export function formatSseEvent(event: string, data: DashboardNotification): string {
  // Encode dashboard notifications as named SSE events.
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
}

export function createDashboardEventStream({
  keepAliveMs = 15_000,
  signal,
  subscribe,
  ttlMs = 50_000,
}: DashboardStreamOptions): ReadableStream<Uint8Array> {
  let closeStream = async () => {};

  return new ReadableStream<Uint8Array>({
    async start(controller) {
      // Track timers and cleanup state inside the stream lifecycle.
      let closed = false;
      let keepAliveTimer: ReturnType<typeof setInterval> | undefined;
      let ttlTimer: ReturnType<typeof setTimeout> | undefined;
      let cleanup: Cleanup | undefined;

      // Enqueue encoded chunks only while the stream is open.
      const enqueue = (chunk: string) => {
        if (!closed) {
          controller.enqueue(encoder.encode(chunk));
        }
      };

      // Close timers, subscriptions, and the controller in one place.
      const close = async () => {
        if (closed) {
          return;
        }
        closed = true;
        if (keepAliveTimer) {
          clearInterval(keepAliveTimer);
        }
        if (ttlTimer) {
          clearTimeout(ttlTimer);
        }
        signal?.removeEventListener("abort", onAbort);
        await cleanup?.();
        try {
          controller.close();
        } catch (cause) {
          if (!(cause instanceof TypeError)) {
            throw cause;
          }
        }
      };

      // Reuse the same close path for AbortSignal teardown.
      const onAbort = () => {
        void close();
      };

      closeStream = close;

      // Send the retry directive and start keepalive / TTL timers.
      enqueue("retry: 1000\n\n");
      keepAliveTimer = setInterval(() => {
        enqueue(formatSseComment("keepalive"));
      }, keepAliveMs);
      ttlTimer = setTimeout(() => {
        void close();
      }, ttlMs);
      signal?.addEventListener("abort", onAbort);

      // Forward backend notifications as refresh events.
      cleanup = await subscribe((notification) => {
        enqueue(formatSseEvent("refresh", notification));
      });
    },
    cancel() {
      // Reuse the async close path when the consumer cancels the stream.
      void closeStream();
    },
  });
}

export function createDashboardSseResponse(options: DashboardStreamOptions): Response {
  // Wrap the event stream in a standard SSE response object.
  return new Response(createDashboardEventStream(options), {
    headers: buildSseHeaders(),
  });
}
