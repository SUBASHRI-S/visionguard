import { useEffect, useState } from "react";
import { api, type ApiSettings } from "../api";
import { clock, ruleLabel, STATUS_LABEL, violationSpan, duration } from "../format";
import { useLoad } from "../hooks";
import type { Filters, IncidentSummary, Status } from "../types";

const PAGE = 25;

interface Props {
  settings: ApiSettings;
  selectedId: number | null;
  refreshKey: number;
  onError: (message: string | null) => void;
}

export const NO_FILTERS: Filters = { status: "", rule_id: "", camera_id: "", active: "" };

export function IncidentList({ settings, selectedId, refreshKey, onError }: Props) {
  const [filters, setFilters] = useState<Filters>(NO_FILTERS);
  const [extra, setExtra] = useState<IncidentSummary[]>([]);
  const first = useLoad(() => api.listIncidents(settings, filters, PAGE, 0), [settings, filters, refreshKey]);
  const [loadingMore, setLoadingMore] = useState(false);

  useEffect(() => setExtra([]), [filters, refreshKey, settings]);
  useEffect(() => onError(first.error ? first.error.message : null), [first.error, onError]);

  const items = [...(first.data?.items ?? []), ...extra];
  const total = first.data?.total ?? 0;

  async function more() {
    setLoadingMore(true);
    try {
      const page = await api.listIncidents(settings, filters, PAGE, items.length);
      setExtra((e) => [...e, ...page.items]);
    } catch (e) {
      onError(e instanceof Error ? e.message : "Could not load more.");
    } finally {
      setLoadingMore(false);
    }
  }

  const set = <K extends keyof Filters>(key: K, value: Filters[K]) => setFilters((f) => ({ ...f, [key]: value }));
  const filtered = Object.values(filters).some((v) => v !== "");

  return (
    <nav className="list" aria-label="Incidents">
      <div className="filters">
        <label>
          Status
          <select value={filters.status} onChange={(e) => set("status", e.target.value as Status | "")}>
            <option value="">All</option>
            {(Object.keys(STATUS_LABEL) as Status[]).map((s) => (
              <option key={s} value={s}>{STATUS_LABEL[s]}</option>
            ))}
          </select>
        </label>
        <label>
          Rule
          <select value={filters.rule_id} onChange={(e) => set("rule_id", e.target.value)}>
            <option value="">All</option>
            <option value="no_helmet">No helmet</option>
            <option value="no_vest">No vest</option>
            <option value="zone_intrusion">Restricted zone</option>
            <option value="forklift_proximity">Close to forklift</option>
          </select>
        </label>
        <label>
          Happening
          <select value={filters.active} onChange={(e) => set("active", e.target.value as Filters["active"])}>
            <option value="">Any time</option>
            <option value="true">Now</option>
            <option value="false">Ended</option>
          </select>
        </label>
      </div>

      {first.loading && !first.data && <p className="muted pad">Loading incidents</p>}
      {first.data && items.length === 0 && (
        <p className="muted pad">
          {filtered ? "No incidents match these filters." : "No incidents yet. Process a video with --db to see them here."}
          {filtered && (
            <> <button type="button" className="link" onClick={() => setFilters(NO_FILTERS)}>Clear filters</button></>
          )}
        </p>
      )}

      <ul>
        {items.map((i) => {
          const span = violationSpan(i);
          return (
            <li key={i.id}>
              <a href={`#/incidents/${i.id}`} aria-current={i.id === selectedId ? "page" : undefined} className={`row row--${i.status}`}>
                <span className="row__title">
                  {ruleLabel(i.rule_id)}
                  {i.active && <span className="live">happening now</span>}
                </span>
                <span className="row__meta">
                  {i.camera_id} · person {i.track_id} · at {clock(i.started_s)}
                  {span !== null && ` · ${duration(span)}`}
                </span>
                <span className="row__status">{STATUS_LABEL[i.status]}</span>
              </a>
            </li>
          );
        })}
      </ul>

      {first.data && items.length < total && (
        <button type="button" className="btn btn--quiet more" disabled={loadingMore} onClick={more}>
          {loadingMore ? "Loading" : `Show more (${total - items.length} left)`}
        </button>
      )}
    </nav>
  );
}
