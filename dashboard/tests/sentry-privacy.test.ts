import { NodeClient, defaultStackParser, requestDataIntegration } from "@sentry/nextjs";
import { sentryCompatibilityOptions } from "@/sentry.shared.config";

describe("Sentry request privacy", () => {
  it("keeps useful request context while excluding cookies and inferred identity", async () => {
    const client = new NodeClient({
      ...sentryCompatibilityOptions,
      enabled: false,
      integrations: [],
      stackParser: defaultStackParser,
      transport: () => ({
        send: async () => ({ statusCode: 200 }),
        flush: async () => true,
      }),
    });
    const integration = requestDataIntegration();
    const event = await integration.processEvent!({
      message: "Synthetic request failure",
      sdkProcessingMetadata: {
        ipAddress: "203.0.113.42",
        normalizedRequest: {
          method: "GET",
          url: "https://fixture.invalid/tracks?campaign=synthetic&remote-user=fixture",
          headers: {
            cookie: "session=synthetic-cookie",
            "x-forwarded-for": "203.0.113.42",
            "x-request-id": "synthetic-request",
          },
        },
      },
    }, {}, client);

    expect(event?.request?.method).toBe("GET");
    expect(event?.request?.headers?.["x-request-id"]).toBe("synthetic-request");
    expect(event?.request?.url).toContain("campaign=synthetic");
    expect(event?.request?.cookies).toBeUndefined();
    expect(event?.user?.ip_address).toBeUndefined();
    expect(JSON.stringify(event?.request)).not.toContain("synthetic-cookie");
    expect(JSON.stringify(event?.request)).not.toContain("203.0.113.42");
    expect(JSON.stringify(event?.request)).not.toContain("remote-user=fixture");
    await client.close();
  });
});
