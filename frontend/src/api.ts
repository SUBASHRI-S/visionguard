import type { Filters, IncidentDetail, IncidentPage, Stats, Status } from "./types";

export interface ApiSettings {
  baseUrl: string;
  apiKey: string;
}

export const DEFAULT_BASE_URL = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";
const STORAGE_KEY = "visionguard.api";

export class ApiError extends Error {
  constructor(
    public kind: "network" | "unauthorized" | "not_found" | "conflict" | "invalid" | "server",
    message: string,
    public status = 0,
  ) {
    super(message);
  }
}

export function loadSettings(): ApiSettings {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "{}");
    return { baseUrl: saved.baseUrl || DEFAULT_BASE_URL, apiKey: saved.apiKey || "" };
  } catch {
    return { baseUrl: DEFAULT_BASE_URL, apiKey: "" };
  }
}

export function saveSettings(settings: ApiSettings): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
  } catch {
    /* storage can be blocked; the settings then last until the page is closed */
  }
}

function headers(settings: ApiSettings, json = false): HeadersInit {
  const h: Record<string, string> = {};
  if (settings.apiKey) h["X-API-Key"] = settings.apiKey;
  if (json) h["Content-Type"] = "application/json";
  return h;
}

function url(settings: ApiSettings, path: string, query: Record<string, string | number | undefined> = {}): string {
  const u = new URL(path, settings.baseUrl.endsWith("/") ? settings.baseUrl : settings.baseUrl + "/");
  for (const [key, value] of Object.entries(query)) {
    if (value !== undefined && value !== "") u.searchParams.set(key, String(value));
  }
  return u.toString();
}

async function request<T>(settings: ApiSettings, target: string, init: RequestInit = {}): Promise<T> {
  let response: Response;
  try {
    response = await fetch(target, init);
  } catch {
    throw new ApiError("network", `Can't reach the API at ${settings.baseUrl}.`);
  }
  if (response.ok) return (await response.json()) as T;
  let detail = "";
  try {
    const body = await response.json();
    detail = typeof body.detail === "string" ? body.detail : "";
  } catch {
    /* not JSON */
  }
  if (response.status === 401) throw new ApiError("unauthorized", "The API needs a key. Add it in Settings.", 401);
  if (response.status === 404) throw new ApiError("not_found", detail || "Not found.", 404);
  if (response.status === 409) throw new ApiError("conflict", detail || "That change isn't allowed.", 409);
  if (response.status === 422) throw new ApiError("invalid", detail || "The request was not valid.", 422);
  throw new ApiError("server", "The API reported a problem. Check its log.", response.status);
}

export const api = {
  listIncidents(settings: ApiSettings, filters: Filters, limit: number, offset: number): Promise<IncidentPage> {
    return request(
      settings,
      url(settings, "/api/v1/incidents", { ...filters, limit, offset }),
      { headers: headers(settings) },
    );
  },
  getIncident(settings: ApiSettings, id: number): Promise<IncidentDetail> {
    return request(settings, url(settings, `/api/v1/incidents/${id}`), { headers: headers(settings) });
  },
  review(settings: ApiSettings, id: number, status: Status, actor: string, note: string): Promise<IncidentDetail> {
    return request(settings, url(settings, `/api/v1/incidents/${id}`), {
      method: "PATCH",
      headers: headers(settings, true),
      body: JSON.stringify({ status, actor, note }),
    });
  },
  stats(settings: ApiSettings, days = 14): Promise<Stats> {
    return request(settings, url(settings, "/api/v1/stats", { days }), { headers: headers(settings) });
  },
  /** Pictures need the API key too, and an <img> tag cannot send headers, so fetch them as data. */
  async evidenceBlob(settings: ApiSettings, evidenceUrl: string): Promise<Blob> {
    let response: Response;
    try {
      response = await fetch(url(settings, evidenceUrl), { headers: headers(settings) });
    } catch {
      throw new ApiError("network", `Can't reach the API at ${settings.baseUrl}.`);
    }
    if (!response.ok) throw new ApiError(response.status === 401 ? "unauthorized" : "not_found", "Picture not available.", response.status);
    return response.blob();
  },
};
