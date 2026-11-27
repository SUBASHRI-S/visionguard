import { useState } from "react";
import { api, ApiError, type ApiSettings } from "../api";
import { clock, END_REASON_LABEL, ruleLabel, STATUS_LABEL, when } from "../format";
import { useLoad } from "../hooks";
import type { Status } from "../types";
import { DurationBar } from "./DurationBar";
import { EvidenceImage } from "./EvidenceImage";
import { ReviewPanel } from "./ReviewPanel";

interface Props {
  settings: ApiSettings;
  id: number;
  actor: string;
  onActor: (name: string) => void;
  onChanged: () => void;
}

const EVIDENCE_LABEL: Record<string, string> = { opened: "When raised", closed: "When closed" };

export function IncidentDetail({ settings, id, actor, onActor, onChanged }: Props) {
  const incident = useLoad(() => api.getIncident(settings, id), [settings, id]);
  const [busy, setBusy] = useState(false);
  const [reviewError, setReviewError] = useState<string | null>(null);
  const [saved, setSaved] = useState<string | null>(null);

  async function review(status: Status, name: string, note: string) {
    setBusy(true);
    setReviewError(null);
    setSaved(null);
    try {
      await api.review(settings, id, status, name, note);
      onActor(name);
      setSaved(`Status is now ${STATUS_LABEL[status].toLowerCase()}.`);
      incident.reload();
      onChanged();
    } catch (e) {
      setReviewError(e instanceof ApiError ? e.message : "Could not save the change.");
      incident.reload();
    } finally {
      setBusy(false);
    }
  }

  if (incident.error && !incident.data) {
    return (
      <div className="detail" role="alert">
        <h2>Can't show this incident</h2>
        <p>{incident.error.message}</p>
        <button type="button" className="btn btn--quiet" onClick={incident.reload}>Try again</button>
      </div>
    );
  }
  const i = incident.data;
  if (!i) return <div className="detail"><p className="muted">Loading incident</p></div>;

  const detailEntries = Object.entries(i.details).filter(([, v]) => ["string", "number", "boolean"].includes(typeof v));

  return (
    <article className="detail" aria-labelledby="incident-title">
      <header>
        <h2 id="incident-title">{ruleLabel(i.rule_id)}</h2>
        <p className="muted">
          {i.camera_id} · person {i.track_id} · run {i.run_id} · {i.severity} severity
        </p>
        <p className={`badge badge--${i.status}`}>{STATUS_LABEL[i.status]}{i.active ? " · happening now" : ""}</p>
      </header>

      <DurationBar
        startedS={i.started_s}
        openedS={i.opened_s}
        lastViolationS={i.last_violation_s}
        endedS={i.ended_s}
        endReason={i.end_reason}
        active={i.active}
      />
      <p className="muted small">
        {i.violating_frames} frames showed the violation, {i.uncertain_frames} were unclear (the detector could not tell).
      </p>

      {i.evidence.length > 0 ? (
        <div className="evidence-row">
          {i.evidence.map((e) => (
            <EvidenceImage key={e.kind} settings={settings} url={e.url} label={EVIDENCE_LABEL[e.kind] ?? e.kind} />
          ))}
        </div>
      ) : (
        <p className="muted">No pictures were saved for this incident.</p>
      )}

      <ReviewPanel status={i.status} busy={busy} error={reviewError} initialActor={actor} onReview={review} />
      {saved && <p className="saved" role="status">{saved}</p>}

      <section>
        <h3>History</h3>
        <ol className="history">
          {i.history.map((h, n) => (
            <li key={n}>
              <span>{when(h.at)}</span>
              <span>
                {h.kind === "status_change" && h.to_status
                  ? `${h.actor} changed status${h.from_status ? ` from ${STATUS_LABEL[h.from_status as Status] ?? h.from_status}` : ""} to ${STATUS_LABEL[h.to_status as Status] ?? h.to_status}`
                  : h.kind === "opened" ? "Raised by the system" : h.kind === "ended" ? `Violation ended${h.note ? ` (${END_REASON_LABEL[h.note] ?? h.note})` : ""}` : `${h.actor}: ${h.kind.replace(/_/g, " ")}`}
                {h.note && h.kind === "status_change" && <em> – {h.note}</em>}
              </span>
            </li>
          ))}
        </ol>
      </section>

      {detailEntries.length > 0 && (
        <section>
          <h3>Details from the rule</h3>
          <dl className="kv">
            {detailEntries.map(([k, v]) => (
              <div key={k}>
                <dt>{k.replace(/_/g, " ")}</dt>
                <dd>{String(v)}</dd>
              </div>
            ))}
          </dl>
        </section>
      )}
      <p className="muted small">Times are video time: {clock(i.started_s)} is where in the clip it began.</p>
    </article>
  );
}
