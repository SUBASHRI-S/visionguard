import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError } from "./api";

export type Route = { page: "home" } | { page: "incident"; id: number } | { page: "stats" };

export function parseHash(hash: string): Route {
  const m = /^#\/incidents\/(\d+)$/.exec(hash);
  if (m) return { page: "incident", id: Number(m[1]) };
  if (hash === "#/stats") return { page: "stats" };
  return { page: "home" };
}

export function useRoute(): Route {
  const [route, setRoute] = useState<Route>(() => parseHash(window.location.hash));
  useEffect(() => {
    const onChange = () => setRoute(parseHash(window.location.hash));
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);
  return route;
}

export interface Loaded<T> {
  data: T | null;
  error: ApiError | null;
  loading: boolean;
  reload: () => void;
}

/** Runs `load` whenever `deps` change; ignores answers that arrive after a newer request started. */
export function useLoad<T>(load: () => Promise<T>, deps: unknown[]): Loaded<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [loading, setLoading] = useState(true);
  const [tick, setTick] = useState(0);
  const latest = useRef(0);
  const loadRef = useRef(load);
  loadRef.current = load;

  useEffect(() => {
    const mine = ++latest.current;
    setLoading(true);
    loadRef
      .current()
      .then((value) => {
        if (mine !== latest.current) return;
        setData(value);
        setError(null);
      })
      .catch((e: unknown) => {
        if (mine !== latest.current) return;
        setError(e instanceof ApiError ? e : new ApiError("server", "Something went wrong."));
      })
      .finally(() => {
        if (mine === latest.current) setLoading(false);
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, tick]);

  const reload = useCallback(() => setTick((t) => t + 1), []);
  return { data, error, loading, reload };
}
