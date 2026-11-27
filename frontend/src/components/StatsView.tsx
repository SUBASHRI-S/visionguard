import { api, type ApiSettings } from "../api";
import { queueSentence, ruleLabel, STATUS_LABEL } from "../format";
import { useLoad } from "../hooks";
import type { Status } from "../types";

function Bars({ title, data, label }: { title: string; data: Record<string, number>; label?: (k: string) => string }) {
  const entries = Object.entries(data);
  const max = Math.max(1, ...entries.map(([, v]) => v));
  return (
    <section className="bars">
      <h3>{title}</h3>
      {entries.length === 0 ? <p className="muted">Nothing yet.</p> : (
        <ul>
          {entries.map(([k, v]) => (
            <li key={k}>
              <span>{label ? label(k) : k}</span>
              <span className="bars__track"><span style={{ width: `${(v / max) * 100}%` }} /></span>
              <span className="bars__n">{v}</span>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

export function StatsView({ settings }: { settings: ApiSettings }) {
  const stats = useLoad(() => api.stats(settings, 14), [settings]);
  if (stats.error && !stats.data) {
    return <div className="detail" role="alert"><h2>Can't load the summary</h2><p>{stats.error.message}</p></div>;
  }
  const s = stats.data;
  if (!s) return <div className="detail"><p className="muted">Loading summary</p></div>;
  const waiting = s.by_status["open"] ?? 0;
  return (
    <div className="detail">
      <h2>Summary</h2>
      <p className="lead">{s.total === 0 ? "No incidents have been recorded." : queueSentence(s.active, waiting)}</p>
      <p className="muted small">{s.total} incidents in total.</p>
      <Bars title="By rule" data={s.by_rule} label={ruleLabel} />
      <Bars title="By status" data={s.by_status} label={(k) => STATUS_LABEL[k as Status] ?? k} />
      <Bars title="By camera" data={s.by_camera} />
      <Bars title="Per day, last 14 days" data={s.by_day} />
    </div>
  );
}
