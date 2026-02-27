const AUTH_TOKEN_KEY = "token";
const CONFIGURED_API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL?.trim() || "";
export const AUTH_TOKEN_EVENT = "bidflow-auth-token-changed";

export function apiUrl(path: string): string {
  return `${getApiBase()}${path.startsWith("/") ? path : `/${path}`}`;
}

function getApiBase(): string {
  if (CONFIGURED_API_BASE) return CONFIGURED_API_BASE.replace(/\/$/, "");

  if (typeof window !== "undefined") {
    const webPort = Number(window.location.port || "3000");
    const apiPort =
      Number.isFinite(webPort) && webPort >= 3100 && webPort <= 3199
        ? 8000 + (webPort - 3000)
        : 8000;
    return `${window.location.protocol}//${window.location.hostname}:${apiPort}`;
  }

  return "http://localhost:8000";
}

export function getStoredAuthToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem(AUTH_TOKEN_KEY);
}

export function clearSessionCache(): void {
  if (typeof window === "undefined") return;

  const keysToRemove: string[] = [];
  for (let i = 0; i < localStorage.length; i += 1) {
    const key = localStorage.key(i);
    if (!key) continue;
    if (
      key.startsWith("cached_") ||
      key.startsWith("analysis_result_") ||
      key.startsWith("validation_result_") ||
      key === "analysis_running_doc_hash" ||
      key === "validation_running_doc_hash"
    ) {
      keysToRemove.push(key);
    }
  }

  keysToRemove.forEach((key) => localStorage.removeItem(key));
}

export function setAuthToken(token: string): void {
  if (typeof window === "undefined") return;
  clearSessionCache();
  localStorage.setItem(AUTH_TOKEN_KEY, token);
  window.dispatchEvent(new Event(AUTH_TOKEN_EVENT));
}

export function clearAuthToken(): void {
  if (typeof window === "undefined") return;
  localStorage.removeItem(AUTH_TOKEN_KEY);
  clearSessionCache();
  window.dispatchEvent(new Event(AUTH_TOKEN_EVENT));
}

export function subscribeAuthToken(onStoreChange: () => void): () => void {
  if (typeof window === "undefined") return () => {};

  const onStorage = (event: StorageEvent) => {
    if (!event.key || event.key === AUTH_TOKEN_KEY) onStoreChange();
  };
  const onTokenChange = () => onStoreChange();

  window.addEventListener("storage", onStorage);
  window.addEventListener(AUTH_TOKEN_EVENT, onTokenChange);

  return () => {
    window.removeEventListener("storage", onStorage);
    window.removeEventListener(AUTH_TOKEN_EVENT, onTokenChange);
  };
}

export function getAuthTokenSnapshot(): string {
  return getStoredAuthToken() ?? "";
}

export function getAuthTokenServerSnapshot(): string {
  return "";
}

export interface RFPDocument {
  id: string;
  doc_hash: string;
  filename: string;
  upload_date: string;
  status?: string;
  owner_name?: string;
  uploaded_by?: string;
  uploaded_by_name?: string;
}

export interface TeamMember {
  username: string;
  name: string;
}

export interface DecisionSummary {
  signal: string;
  recommendation: string;
  n_red: number;
  n_gray: number;
  n_green: number;
}

export interface UploadDocumentResponse {
  status: string;
  doc_id: string;
  doc_hash: string;
  filename: string;
  chunk_count: number;
  table_count: number;
  user_id: string;
  message: string;
}

export interface ExtractionSlotData {
  value?: unknown;
  evidence?: Array<Record<string, unknown>>;
  [key: string]: unknown;
}

export type ExtractionSlotValue =
  | ExtractionSlotData
  | string
  | number
  | boolean
  | null
  | undefined;

export interface ExtractionData {
  g1?: Record<string, ExtractionSlotValue>;
  g2?: Record<string, ExtractionSlotValue>;
  g3?: Record<string, ExtractionSlotValue>;
  g4?: unknown;
  [key: string]: unknown;
}

export interface ExtractionResponse {
  status: string;
  doc_id: string;
  data: ExtractionData;
}

export interface ValidationRequest {
  doc_hash: string;
  slots: Record<string, unknown>;
}

export interface ValidationResult {
  slot_key: string;
  decision: string;
  reasons?: string[];
  evidence?: Array<Record<string, unknown>>;
  risk_level?: string;
  timestamp?: string;
  [key: string]: unknown;
}

export interface UserInfo {
  username: string;
  name: string;
  email: string;
  team: string;
  licenses: string;
  region?: string;
  company_logo?: string | null;
  role: string;
}

function getAuthHeaders(): Record<string, string> {
  const token = getStoredAuthToken();
  if (!token) throw new Error("Not authenticated");
  return {
    Authorization: `Bearer ${token}`,
    "Content-Type": "application/json",
  };
}

function getAuthHeadersMultipart(): Record<string, string> {
  const token = getStoredAuthToken();
  if (!token) throw new Error("Not authenticated");
  return {
    Authorization: `Bearer ${token}`,
  };
}

/**
 * 인증된 fetch 래퍼: 401 응답 시 토큰을 정리하고 로그인 페이지로 이동합니다.
 */
async function authFetch(input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
  const res = await fetch(input, init);
  if (res.status === 401) {
    clearAuthToken();
    if (typeof window !== "undefined") {
      window.location.href = "/";
    }
  }
  return res;
}

// ── Documents ────────────────────────────────────────────────────

export async function getDocuments(): Promise<RFPDocument[]> {
  const res = await authFetch(apiUrl("/api/v1/ingest/documents"), {
    headers: getAuthHeaders(),
  });
  if (!res.ok) throw new Error(`Failed to fetch documents: ${res.status}`);
  return res.json();
}

export async function uploadDocument(file: File): Promise<UploadDocumentResponse> {
  const formData = new FormData();
  formData.append("file", file);
  const res = await authFetch(apiUrl("/api/v1/ingest/upload"), {
    method: "POST",
    headers: getAuthHeadersMultipart(),
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Upload failed: ${res.status}`);
  }
  return res.json();
}

// ── Extraction ───────────────────────────────────────────────────

export async function runExtraction(
  docHash: string
): Promise<ExtractionResponse> {
  const res = await authFetch(apiUrl(`/api/v1/extract/${docHash}`), {
    method: "POST",
    headers: getAuthHeaders(),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Extraction failed: ${res.status}`);
  }
  return res.json();
}

export async function getExtractionResult(
  docHash: string
): Promise<ExtractionResponse | null> {
  try {
    const res = await authFetch(apiUrl(`/api/v1/extract/${docHash}`), {
      headers: getAuthHeaders(),
    });
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}

// ── Validation ───────────────────────────────────────────────────

export async function runValidation(
  matrix: ValidationRequest
): Promise<ValidationResult[]> {
  const res = await authFetch(apiUrl("/api/v1/validate"), {
    method: "POST",
    headers: getAuthHeaders(),
    body: JSON.stringify(matrix),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Validation failed: ${res.status}`);
  }
  return res.json();
}

// ── Team ─────────────────────────────────────────────────────────

export async function getTeamMembers(): Promise<TeamMember[]> {
  const res = await authFetch(apiUrl("/api/v1/team/members"), {
    headers: getAuthHeaders(),
  });
  if (!res.ok) return [];
  return res.json();
}

export async function getTeamDocuments(): Promise<RFPDocument[]> {
  const res = await authFetch(apiUrl("/api/v1/team/documents"), {
    headers: getAuthHeaders(),
  });
  if (!res.ok) return [];
  return res.json();
}

export async function getDecisionSummary(
  docHash: string
): Promise<DecisionSummary | null> {
  try {
    const res = await authFetch(apiUrl(`/api/v1/team/decision/${docHash}`), {
      headers: getAuthHeaders(),
    });
    if (!res.ok) return null;
    const data = await res.json();
    return data;
  } catch {
    return null;
  }
}

// ── User ─────────────────────────────────────────────────────────

export async function getCurrentUser(): Promise<UserInfo | null> {
  try {
    const res = await authFetch(apiUrl("/auth/me"), {
      headers: getAuthHeaders(),
    });
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}

export async function updateUserProfile(data: {
  team?: string;
  licenses?: string;
  region?: string;
  company_logo?: string;
}): Promise<UserInfo> {
  const res = await authFetch(apiUrl("/auth/me"), {
    method: "PUT",
    headers: getAuthHeaders(),
    body: JSON.stringify(data),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Update failed: ${res.status}`);
  }
  return res.json();
}
