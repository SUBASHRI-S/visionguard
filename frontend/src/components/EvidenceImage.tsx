import { useEffect, useState } from "react";
import { api, type ApiSettings } from "../api";

interface Props {
  settings: ApiSettings;
  url: string;
  label: string;
}

export function EvidenceImage({ settings, url, label }: Props) {
  const [src, setSrc] = useState<string | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let revoked = false;
    let objectUrl: string | null = null;
    setSrc(null);
    setFailed(false);
    api
      .evidenceBlob(settings, url)
      .then((blob) => {
        if (revoked) return;
        objectUrl = URL.createObjectURL(blob);
        setSrc(objectUrl);
      })
      .catch(() => !revoked && setFailed(true));
    return () => {
      revoked = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [settings, url]);

  return (
    <figure className="evidence">
      {src ? <img src={src} alt={`${label} frame from the incident`} /> : <div className="evidence__blank">{failed ? "Picture not available" : "Loading picture"}</div>}
      <figcaption>{label}</figcaption>
    </figure>
  );
}
