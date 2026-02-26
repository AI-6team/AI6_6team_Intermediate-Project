export const API_BASE = (process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000").replace(/\/$/, "");

export function apiUrl(path: string): string {
  return `${API_BASE}${path.startsWith("/") ? path : `/${path}`}`;
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
  const token = localStorage.getItem("token");
  if (!token) throw new Error("Not authenticated");
  return {
    Authorization: `Bearer ${token}`,
    "Content-Type": "application/json",
  };
}

function getAuthHeadersMultipart(): Record<string, string> {
  const token = localStorage.getItem("token");
  if (!token) throw new Error("Not authenticated");
  return {
    Authorization: `Bearer ${token}`,
  };
}

// ── Documents ────────────────────────────────────────────────────

export async function getDocuments(): Promise<RFPDocument[]> {
  const res = await fetch(apiUrl("/api/v1/ingest/documents"), {
    headers: getAuthHeaders(),
  });
  if (!res.ok) throw new Error(`Failed to fetch documents: ${res.status}`);
  return res.json();
}

export async function uploadDocument(file: File): Promise<UploadDocumentResponse> {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(apiUrl("/api/v1/ingest/upload"), {
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
  const res = await fetch(apiUrl(`/api/v1/extract/${docHash}`), {
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
    const res = await fetch(apiUrl(`/api/v1/extract/${docHash}`), {
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
  const res = await fetch(apiUrl("/api/v1/validate"), {
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
  const res = await fetch(apiUrl("/api/v1/team/members"), {
    headers: getAuthHeaders(),
  });
  if (!res.ok) return [];
  return res.json();
}

export async function getTeamDocuments(): Promise<RFPDocument[]> {
  const res = await fetch(apiUrl("/api/v1/team/documents"), {
    headers: getAuthHeaders(),
  });
  if (!res.ok) return [];
  return res.json();
}

export async function getDecisionSummary(
  docHash: string
): Promise<DecisionSummary | null> {
  try {
    const res = await fetch(apiUrl(`/api/v1/team/decision/${docHash}`), {
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
    const res = await fetch(apiUrl("/auth/me"), {
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
  const res = await fetch(apiUrl("/auth/me"), {
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
