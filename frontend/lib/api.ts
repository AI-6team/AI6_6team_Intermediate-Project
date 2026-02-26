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

export async function uploadDocument(file: File): Promise<any> {
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
): Promise<{ status: string; doc_id: string; data: any }> {
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
): Promise<{ status: string; doc_id: string; data: any } | null> {
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
  matrix: any
): Promise<any[]> {
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
