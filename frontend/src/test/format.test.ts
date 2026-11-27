import { describe, expect, it } from "vitest";
import { clock, duration, NEXT_STATUSES, queueSentence, violationSpan } from "../format";
import { parseHash } from "../hooks";

describe("format", () => {
  it("formats video time", () => {
    expect(clock(7.4)).toBe("0:07.4");
    expect(clock(65)).toBe("1:05.0");
  });
  it("handles missing times", () => {
    expect(clock(null)).toBe("-");
    expect(clock(Number.NaN)).toBe("-");
    expect(clock(-3)).toBe("0:00.0");
  });
  it("formats durations", () => {
    expect(duration(2.34)).toBe("2.3 s");
    expect(duration(125)).toBe("2 min 5 s");
  });
  it("measures the span to the last violation, not to the give-up time", () => {
    expect(violationSpan({ started_s: 1, last_violation_s: 1.8, ended_s: 4.84, opened_s: 4 })).toBeCloseTo(0.8);
    expect(violationSpan({ started_s: 1, last_violation_s: null, ended_s: 3, opened_s: null })).toBe(2);
    expect(violationSpan({ started_s: 1, last_violation_s: null, ended_s: null, opened_s: null })).toBeNull();
  });
  it("describes the queue in plain words", () => {
    expect(queueSentence(0, 0)).toBe("No violations happening now. Nothing is waiting for review.");
    expect(queueSentence(1, 1)).toBe("1 violation happening now. 1 incident is waiting for review.");
    expect(queueSentence(3, 2)).toBe("3 violations happening now. 2 incidents are waiting for review.");
  });
  it("only offers allowed status changes", () => {
    expect(NEXT_STATUSES.open).toContain("acknowledged");
    expect(NEXT_STATUSES.resolved).toEqual(["open"]);
  });
});

describe("routes", () => {
  it("parses the hash", () => {
    expect(parseHash("")).toEqual({ page: "home" });
    expect(parseHash("#/stats")).toEqual({ page: "stats" });
    expect(parseHash("#/incidents/12")).toEqual({ page: "incident", id: 12 });
    expect(parseHash("#/incidents/abc")).toEqual({ page: "home" });
    expect(parseHash("#/incidents/1/../2")).toEqual({ page: "home" });
  });
});
