import { afterEach, describe, expect, it, vi } from "vitest";
import { api, ApiError, loadSettings } from "../api";
import { NO_FILTERS } from "../components/IncidentList";

const settings = { baseUrl: "http://api.test", apiKey: "k" };

function respond(status: number, body: unknown) {
  return vi.fn().mockResolvedValue(new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } }));
}

afterEach(() => vi.unstubAllGlobals());

describe("api", () => {
  it("sends the key and only the filters that are set", async () => {
    const fetchMock = respond(200, { items: [], total: 0, limit: 25, offset: 0 });
    vi.stubGlobal("fetch", fetchMock);
    await api.listIncidents(settings, { ...NO_FILTERS, status: "open" }, 25, 0);
    const [target, init] = fetchMock.mock.calls[0];
    expect(target).toBe("http://api.test/api/v1/incidents?status=open&limit=25&offset=0");
    expect((init as RequestInit).headers).toMatchObject({ "X-API-Key": "k" });
  });
  it("omits the key header when none is set", async () => {
    const fetchMock = respond(200, {});
    vi.stubGlobal("fetch", fetchMock);
    await api.stats({ baseUrl: "http://api.test", apiKey: "" }, 14);
    expect((fetchMock.mock.calls[0][1] as RequestInit).headers).not.toHaveProperty("X-API-Key");
  });
  it("sends the review as JSON", async () => {
    const fetchMock = respond(200, {});
    vi.stubGlobal("fetch", fetchMock);
    await api.review(settings, 3, "resolved", "Sam", "fixed");
    const init = fetchMock.mock.calls[0][1] as RequestInit;
    expect(init.method).toBe("PATCH");
    expect(JSON.parse(init.body as string)).toEqual({ status: "resolved", actor: "Sam", note: "fixed" });
  });
  it.each([
    [401, "unauthorized"],
    [404, "not_found"],
    [409, "conflict"],
    [422, "invalid"],
    [500, "server"],
  ])("maps HTTP %i to %s", async (status, kind) => {
    vi.stubGlobal("fetch", respond(status, { detail: "x" }));
    await expect(api.getIncident(settings, 1)).rejects.toMatchObject({ kind });
  });
  it("reports an unreachable API in plain words", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("failed")));
    const error = await api.getIncident(settings, 1).catch((e) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect(error.kind).toBe("network");
    expect(error.message).toContain("http://api.test");
  });
  it("falls back to defaults when stored settings are broken", () => {
    localStorage.setItem("visionguard.api", "{not json");
    expect(loadSettings().baseUrl).toMatch(/^http/);
  });
});
