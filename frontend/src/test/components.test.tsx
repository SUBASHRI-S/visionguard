import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import App from "../App";
import { DurationBar } from "../components/DurationBar";
import { ReviewPanel } from "../components/ReviewPanel";
import type { IncidentDetail, IncidentSummary } from "../types";

const summary: IncidentSummary = {
  id: 7, run_id: "run1", camera_id: "cam1", rule_id: "no_helmet", track_id: 3, episode: 1, severity: "high",
  status: "open", active: false, started_s: 2, opened_s: 5, last_violation_s: 4, ended_s: 7,
  end_reason: "cleared", violating_frames: 20, uncertain_frames: 2, created_at: "2026-11-01T10:00:00Z", updated_at: "2026-11-01T10:00:00Z",
};
const detail: IncidentDetail = { ...summary, details: { person_height_px: 210 }, history: [], evidence: [] };

function json(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

afterEach(() => {
  vi.unstubAllGlobals();
  window.location.hash = "";
});

describe("DurationBar", () => {
  it("describes the incident for screen readers", () => {
    render(<DurationBar startedS={2} openedS={5} lastViolationS={4} endedS={7} endReason="cleared" active={false} />);
    expect(screen.getAllByLabelText(/Violation from 0:02.0 to 0:04.0, raised at 0:05.0, closed at 0:07.0/).length).toBeGreaterThan(0);
    expect(screen.getByText(/waited 3.0 s/)).toBeInTheDocument();
  });
  it("shows a running incident as still happening", () => {
    render(<DurationBar startedS={2} openedS={null} lastViolationS={3} endedS={null} endReason={null} active />);
    expect(screen.getAllByText(/still happening/).length).toBeGreaterThan(0);
  });
});

describe("ReviewPanel", () => {
  it("offers only the allowed changes and needs a name", async () => {
    const onReview = vi.fn();
    render(<ReviewPanel status="resolved" busy={false} error={null} initialActor="" onReview={onReview} />);
    expect(screen.queryByRole("button", { name: "Mark resolved" })).not.toBeInTheDocument();
    const reopen = screen.getByRole("button", { name: "Reopen" });
    expect(reopen).toBeDisabled();
    await userEvent.type(screen.getByLabelText("Your name"), "  Sam ");
    await userEvent.type(screen.getByLabelText(/Note/), "checked");
    await userEvent.click(reopen);
    expect(onReview).toHaveBeenCalledWith("open", "Sam", "checked");
  });
  it("shows an error", () => {
    render(<ReviewPanel status="open" busy={false} error="That change isn't allowed." initialActor="Sam" onReview={() => {}} />);
    expect(screen.getByRole("alert")).toHaveTextContent("isn't allowed");
  });
});

describe("StatsView", () => {
  it("uses the singular for one incident and keeps long camera names in the label", async () => {
    const long = "67460-522170651_medium.mp4";
    vi.stubGlobal("fetch", vi.fn(async () => json({ total: 1, active: 0, by_rule: { no_helmet: 1 }, by_status: { open: 1 }, by_severity: { high: 1 }, by_camera: { [long]: 1 }, by_day: {} })));
    window.location.hash = "#/stats";
    render(<App />);
    expect(await screen.findByText("1 incident in total.")).toBeInTheDocument();
    expect(screen.getByTitle(long)).toBeInTheDocument();
  });
});

describe("App", () => {
  it("lists incidents and reviews one", async () => {
    let current = detail;
    const fetchMock = vi.fn(async (target: string, init?: RequestInit) => {
      const path = new URL(target).pathname;
      if (init?.method === "PATCH") {
        current = { ...current, status: JSON.parse(init.body as string).status };
        return json(current);
      }
      if (path === "/api/v1/incidents") return json({ items: [{ ...current }], total: 1, limit: 25, offset: 0 });
      if (path === "/api/v1/incidents/7") return json(current);
      return json({}, 404);
    });
    vi.stubGlobal("fetch", fetchMock);
    window.location.hash = "#/incidents/7";
    render(<App />);
    expect(await screen.findByRole("heading", { name: "No helmet", level: 2 })).toBeInTheDocument();
    await userEvent.type(screen.getByLabelText("Your name"), "Sam");
    await userEvent.click(screen.getByRole("button", { name: "Acknowledge" }));
    expect(await screen.findByText("Status is now acknowledged.")).toBeInTheDocument();
    const patch = fetchMock.mock.calls.find(([, init]) => init?.method === "PATCH")!;
    expect(JSON.parse(patch[1]!.body as string)).toMatchObject({ status: "acknowledged", actor: "Sam" });
  });

  it("explains an empty list", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => json({ items: [], total: 0, limit: 25, offset: 0 })));
    render(<App />);
    expect(await screen.findByText(/No incidents yet/)).toBeInTheDocument();
  });

  it("says what to do when the API is unreachable", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("down")));
    render(<App />);
    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent(/Can't reach the API/));
  });

  it("asks for a key on 401", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => json({ detail: "no" }, 401)));
    render(<App />);
    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent(/needs a key/));
  });
});
