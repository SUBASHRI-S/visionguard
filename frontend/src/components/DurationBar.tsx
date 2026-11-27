import { clock, duration, END_REASON_LABEL } from "../format";

interface Props {
  startedS: number;
  openedS: number | null;
  lastViolationS: number | null;
  endedS: number | null;
  endReason: string | null;
  active: boolean;
}

/**
 * One incident drawn on the video's own timeline:
 * solid = violation seen, hatched = system waiting to be sure it had stopped.
 * The bar is scaled to the incident, not to the whole video, so a 2 s and a 20 s incident are both readable.
 */
export function DurationBar({ startedS, openedS, lastViolationS, endedS, endReason, active }: Props) {
  const lastSeen = lastViolationS ?? endedS ?? openedS ?? startedS;
  const end = Math.max(endedS ?? lastSeen, lastSeen);
  const span = Math.max(end - startedS, 0.001);
  const pct = (t: number) => `${(Math.min(Math.max(t - startedS, 0), span) / span) * 100}%`;
  const solidWidth = pct(lastSeen);
  const reason = endReason ? END_REASON_LABEL[endReason] ?? endReason : null;

  const label =
    `Violation from ${clock(startedS)} to ${clock(lastSeen)}` +
    (openedS !== null ? `, raised at ${clock(openedS)}` : ", not raised") +
    (active ? ", still happening" : endedS !== null ? `, closed at ${clock(endedS)}` : "");

  return (
    <figure className="durbar" aria-label={label}>
      <div className="durbar__track" role="img" aria-label={label}>
        <div className="durbar__seen" style={{ width: solidWidth }} />
        {end > lastSeen && <div className="durbar__wait" style={{ left: solidWidth, right: 0 }} />}
        {openedS !== null && <div className="durbar__flag" style={{ left: pct(openedS) }} title={`Raised at ${clock(openedS)}`} />}
      </div>
      <div className="durbar__axis">
        <span>{clock(startedS)}</span>
        <span>{active ? "still happening" : duration(end - startedS)}</span>
        <span>{clock(end)}</span>
      </div>
      <figcaption className="durbar__legend">
        <span className="key key--seen" /> violation seen for {duration(lastSeen - startedS)}
        {end > lastSeen && (
          <>
            <span className="key key--wait" /> waited {duration(end - lastSeen)}
            {reason ? ` (${reason})` : ""}
          </>
        )}
        {openedS !== null && (
          <>
            <span className="key key--flag" /> raised at {clock(openedS)}
          </>
        )}
      </figcaption>
    </figure>
  );
}
