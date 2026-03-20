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
  return new Headers({
    "Cache-Control": "no-store, no-transform",
    Connection: "keep-alive",
    "Content-Type": "text/event-stream; charset=utf-8",
    "X-Accel-Buffering": "no",
  });
}

export function formatSseComment(message: string): string {
  return `: ${message}\n\n`;
}

export function formatSseEvent(event: string, data: DashboardNotification): string {
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
      let closed = false;
      let keepAliveTimer: ReturnType<typeof setInterval> | undefined;
      let ttlTimer: ReturnType<typeof setTimeout> | undefined;
      let cleanup: Cleanup | undefined;

      const enqueue = (chunk: string) => {
        if (!closed) {
          controller.enqueue(encoder.encode(chunk));
        }
      };

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
        controller.close();
      };

      const onAbort = () => {
        void close();
      };

      closeStream = close;

      enqueue("retry: 1000\n\n");
      keepAliveTimer = setInterval(() => {
        enqueue(formatSseComment("keepalive"));
      }, keepAliveMs);
      ttlTimer = setTimeout(() => {
        void close();
      }, ttlMs);
      signal?.addEventListener("abort", onAbort);

      cleanup = await subscribe((notification) => {
        enqueue(formatSseEvent("refresh", notification));
      });
    },
    cancel() {
      void closeStream();
    },
  });
}

export function createDashboardSseResponse(options: DashboardStreamOptions): Response {
  return new Response(createDashboardEventStream(options), {
    headers: buildSseHeaders(),
  });
}
