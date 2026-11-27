import { useState } from "react";
import { ACTION_LABEL, NEXT_STATUSES, STATUS_LABEL } from "../format";
import type { Status } from "../types";

interface Props {
  status: Status;
  busy: boolean;
  error: string | null;
  initialActor: string;
  onReview: (status: Status, actor: string, note: string) => void;
}

export function ReviewPanel({ status, busy, error, initialActor, onReview }: Props) {
  const [actor, setActor] = useState(initialActor);
  const [note, setNote] = useState("");
  const next = NEXT_STATUSES[status] ?? [];
  const canSubmit = actor.trim().length > 0 && !busy;

  return (
    <section className="review" aria-labelledby="review-title">
      <h3 id="review-title">Your review</h3>
      <p className="review__now">
        Current status: <strong>{STATUS_LABEL[status]}</strong>
      </p>
      <div className="field">
        <label htmlFor="actor">Your name</label>
        <input id="actor" value={actor} maxLength={80} onChange={(e) => setActor(e.target.value)} autoComplete="name" />
      </div>
      <div className="field">
        <label htmlFor="note">Note (optional)</label>
        <textarea id="note" value={note} maxLength={500} rows={2} onChange={(e) => setNote(e.target.value)} />
      </div>
      {actor.trim().length === 0 && <p className="hint">Enter your name so the change is recorded against you.</p>}
      <div className="review__actions">
        {next.map((target) => (
          <button
            key={target}
            type="button"
            className={target === "false_positive" ? "btn btn--quiet" : target === "open" ? "btn btn--quiet" : "btn"}
            disabled={!canSubmit}
            onClick={() => {
              onReview(target, actor.trim(), note.trim());
              setNote("");
            }}
          >
            {ACTION_LABEL[target]}
          </button>
        ))}
      </div>
      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}
    </section>
  );
}
