import { useCallback, useState } from "react";
import { loadSettings, saveSettings, type ApiSettings } from "./api";
import { IncidentDetail } from "./components/IncidentDetail";
import { IncidentList } from "./components/IncidentList";
import { SettingsDialog } from "./components/SettingsDialog";
import { StatsView } from "./components/StatsView";
import { useRoute } from "./hooks";

const ACTOR_KEY = "visionguard.actor";

function readActor(): string {
  try {
    return localStorage.getItem(ACTOR_KEY) ?? "";
  } catch {
    return "";
  }
}

export default function App() {
  const route = useRoute();
  const [settings, setSettings] = useState<ApiSettings>(loadSettings);
  const [showSettings, setShowSettings] = useState(false);
  const [refreshKey, setRefreshKey] = useState(0);
  const [listError, setListError] = useState<string | null>(null);
  const [actor, setActor] = useState(readActor);

  const rememberActor = useCallback((name: string) => {
    setActor(name);
    try {
      localStorage.setItem(ACTOR_KEY, name);
    } catch {
      /* optional convenience */
    }
  }, []);

  return (
    <div className="shell">
      <header className="top">
        <a href="#/" className="brand">VisionGuard</a>
        <nav aria-label="Pages">
          <a href="#/" aria-current={route.page !== "stats" ? "page" : undefined}>Review</a>
          <a href="#/stats" aria-current={route.page === "stats" ? "page" : undefined}>Summary</a>
        </nav>
        <button type="button" className="btn btn--quiet" onClick={() => setShowSettings(true)}>Connection</button>
      </header>

      {listError && (
        <div className="banner" role="alert">
          {listError}{" "}
          <button type="button" className="link" onClick={() => setRefreshKey((k) => k + 1)}>Try again</button>
        </div>
      )}

      <main className={`desk desk--${route.page}`}>
        <IncidentList
          settings={settings}
          selectedId={route.page === "incident" ? route.id : null}
          refreshKey={refreshKey}
          onError={setListError}
        />
        <section className="pane">
          {route.page !== "home" && <a href="#/" className="back">Back to the list</a>}
          {route.page === "incident" && (
            <IncidentDetail
              key={route.id}
              settings={settings}
              id={route.id}
              actor={actor}
              onActor={rememberActor}
              onChanged={() => setRefreshKey((k) => k + 1)}
            />
          )}
          {route.page === "stats" && <StatsView settings={settings} />}
          {route.page === "home" && (
            <div className="detail empty">
              <h2>Pick an incident to review</h2>
              <p className="muted">Choose one from the list. Each shows when the violation began, how long it lasted, and the pictures saved.</p>
            </div>
          )}
        </section>
      </main>

      {showSettings && (
        <SettingsDialog
          settings={settings}
          onSave={(s) => {
            saveSettings(s);
            setSettings(s);
          }}
          onClose={() => setShowSettings(false)}
        />
      )}
    </div>
  );
}
