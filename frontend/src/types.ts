export type Status = "open" | "acknowledged" | "resolved" | "false_positive";

export interface IncidentSummary {
  id: number;
  run_id: string;
  camera_id: string;
  rule_id: string;
  track_id: number;
  episode: number;
  severity: string;
  status: Status;
  active: boolean;
  started_s: number;
  opened_s: number | null;
  last_violation_s: number | null;
  ended_s: number | null;
  end_reason: string | null;
  violating_frames: number;
  uncertain_frames: number;
  created_at: string;
  updated_at: string;
}

export interface HistoryEntry {
  kind: string;
  from_status: string | null;
  to_status: string | null;
  actor: string;
  note: string;
  at: string;
}

export interface EvidenceLink {
  kind: string;
  url: string;
}

export interface IncidentDetail extends IncidentSummary {
  details: Record<string, unknown>;
  history: HistoryEntry[];
  evidence: EvidenceLink[];
}

export interface IncidentPage {
  items: IncidentSummary[];
  total: number;
  limit: number;
  offset: number;
}

export interface Stats {
  total: number;
  active: number;
  by_rule: Record<string, number>;
  by_status: Record<string, number>;
  by_severity: Record<string, number>;
  by_camera: Record<string, number>;
  by_day: Record<string, number>;
}

export interface Filters {
  status: Status | "";
  rule_id: string;
  camera_id: string;
  active: "" | "true" | "false";
}
