import { useState } from "react";
import type { ApiSettings } from "../api";

interface Props {
  settings: ApiSettings;
  onSave: (settings: ApiSettings) => void;
  onClose: () => void;
}

export function SettingsDialog({ settings, onSave, onClose }: Props) {
  const [baseUrl, setBaseUrl] = useState(settings.baseUrl);
  const [apiKey, setApiKey] = useState(settings.apiKey);
  return (
    <div className="scrim" role="presentation" onClick={onClose}>
      <form
        className="dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="settings-title"
        onClick={(e) => e.stopPropagation()}
        onSubmit={(e) => {
          e.preventDefault();
          onSave({ baseUrl: baseUrl.trim() || settings.baseUrl, apiKey: apiKey.trim() });
          onClose();
        }}
      >
        <h2 id="settings-title">Connection</h2>
        <div className="field">
          <label htmlFor="base-url">API address</label>
          <input id="base-url" value={baseUrl} onChange={(e) => setBaseUrl(e.target.value)} />
        </div>
        <div className="field">
          <label htmlFor="api-key">API key</label>
          <input id="api-key" type="password" value={apiKey} onChange={(e) => setApiKey(e.target.value)} autoComplete="off" />
          <p className="hint">Only needed if the API was started with VG_API_KEY. It is kept in this browser only.</p>
        </div>
        <div className="review__actions">
          <button type="submit" className="btn">Save</button>
          <button type="button" className="btn btn--quiet" onClick={onClose}>Cancel</button>
        </div>
      </form>
    </div>
  );
}
