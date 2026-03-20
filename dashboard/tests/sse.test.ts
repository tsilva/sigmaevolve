import { createDashboardSseResponse } from "@/lib/sse";

async function readStream(stream: ReadableStream<Uint8Array>, chunks: number): Promise<string> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let output = "";

  for (let index = 0; index < chunks; index += 1) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    output += decoder.decode(value);
  }

  return output;
}

describe("dashboard SSE response", () => {
  it("returns stream headers and emits refresh events plus keepalive comments", async () => {
    vi.useFakeTimers();

    const response = createDashboardSseResponse({
      keepAliveMs: 10,
      ttlMs: 30,
      subscribe: (emit) => {
        emit({ trackId: "track_1", reason: "trial_changed" });
        return () => undefined;
      },
    });

    expect(response.headers.get("content-type")).toContain("text/event-stream");
    expect(response.headers.get("cache-control")).toContain("no-store");

    const outputPromise = readStream(response.body as ReadableStream<Uint8Array>, 3);
    await vi.advanceTimersByTimeAsync(12);
    const output = await outputPromise;

    expect(output).toContain("retry: 1000");
    expect(output).toContain("event: refresh");
    expect(output).toContain('"trackId":"track_1"');
    expect(output).toContain(": keepalive");

    vi.useRealTimers();
  });
});
