import transitions from "./statusTransitions.json";
import type { IncidentSummary, Status } from "./types";

/** What a reviewer can do next from each state. The API checks the same table (see the contract test). */
export const NEXT_STATUSES = transitions as Record<Status, Status[]>;

export const STATUS_LABEL: Record<Status, string> = {
  open: "Needs review",
  acknowledged: "Acknowledged",
  resolved: "Resolved",
  false_positive: "False alarm",
};

/** The button for each change names the action; the toast afterwards names the result. */
export const ACTION_LABEL: Record<Status, string> = {
  open: "Reopen",
  acknowledged: "Acknowledge",
  resolved: "Mark resolved",
  false_positive: "Mark as false alarm",
};

export const RULE_LABEL: Record<string, string> = {
  no_helmet: "No helmet",
  no_vest: "No vest",
  zone_intrusion: "Restricted zone",
  forklift_proximity: "Close to forklift",
};

export const END_REASON_LABEL: Record<string, string> = {
  cleared: "violation stopped",
  track_lost: "person left the view",
  stream_end: "video ended",
};

export function ruleLabel(rule: string): string {
  return RULE_LABEL[rule] ?? rule;
}

/** Video seconds as m:ss.s, e.g. 7.4 -> "0:07.4". */
export function clock(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || Number.isNaN(seconds)) return "-";
  const total = Math.max(0, seconds);
  const minutes = Math.floor(total / 60);
  const rest = total - minutes * 60;
  return `${minutes}:${rest.toFixed(1).padStart(4, "0")}`;
}

export function duration(seconds: number): string {
  return seconds < 60 ? `${seconds.toFixed(1)} s` : `${Math.floor(seconds / 60)} min ${Math.round(seconds % 60)} s`;
}

export function when(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString(undefined, { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
}

/** How long the violation lasted, in video seconds (to the last violation, not to when we gave up waiting). */
export function violationSpan(i: Pick<IncidentSummary, "started_s" | "last_violation_s" | "ended_s" | "opened_s">): number | null {
  const end = i.last_violation_s ?? i.ended_s ?? i.opened_s;
  return end === null ? null : Math.max(0, end - i.started_s);
}

export function queueSentence(active: number, needsReview: number): string {
  const now = active === 0 ? "No violations happening now." : active === 1 ? "1 violation happening now." : `${active} violations happening now.`;
  const waiting = needsReview === 0 ? "Nothing is waiting for review." : needsReview === 1 ? "1 incident is waiting for review." : `${needsReview} incidents are waiting for review.`;
  return `${now} ${waiting}`;
}
