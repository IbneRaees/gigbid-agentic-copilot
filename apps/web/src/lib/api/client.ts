import type {
  ApiHealth,
  AuditLogEntry,
  ClientThread,
  EvalResultsBundle,
  FreelancerPersona,
  MarketplaceJob,
  PortfolioItem,
  RunSummary,
  SemanticMemoryItem,
} from "./types";

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL ||
  process.env.NEXT_PUBLIC_API_BASE_URL ||
  "http://localhost:8000";

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers || {}),
    },
    cache: "no-store",
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`API ${res.status}: ${detail}`);
  }
  return (await res.json()) as T;
}

export const gigbidApi = {
  getHealth: () => apiFetch<ApiHealth>("/health"),

  listJobs: (params?: { q?: string; domain?: string; freelancer_id?: string }) => {
    const qs = new URLSearchParams();
    if (params?.q) qs.set("q", params.q);
    if (params?.domain) qs.set("domain", params.domain);
    if (params?.freelancer_id) qs.set("freelancer_id", params.freelancer_id);
    const suffix = qs.toString() ? `?${qs.toString()}` : "";
    return apiFetch<{ total: number; jobs: MarketplaceJob[]; marketplace_mode: string }>(
      `/jobs${suffix}`
    );
  },

  listFreelancers: () =>
    apiFetch<{ total: number; freelancers: FreelancerPersona[] }>("/freelancers"),

  listThreads: () =>
    apiFetch<{ total: number; threads: ClientThread[] }>("/threads"),

  listRuns: (pendingOnly = false) =>
    apiFetch<RunSummary[]>(`/runs${pendingOnly ? "?pending_approval=true" : ""}`),

  getRun: (threadId: string) => apiFetch<RunSummary>(`/runs/${encodeURIComponent(threadId)}`),

  startBidRun: (payload: {
    job_id: string;
    freelancer_id: string;
    rag_variant?: string;
    model_profile?: string;
  }) =>
    apiFetch<RunSummary>("/runs/bid", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  startConversationRun: (payload: {
    thread_id: string;
    freelancer_id: string;
    incoming_message?: string;
  }) =>
    apiFetch<RunSummary>("/runs/conversation", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  approveRun: (
    threadId: string,
    payload: {
      approved: boolean;
      reason?: string;
      cover_letter?: string;
      proposed_rate?: number;
      reply_text?: string;
      selected_slot?: string;
      create_email_draft?: boolean;
    }
  ) =>
    apiFetch<RunSummary>(`/runs/${encodeURIComponent(threadId)}/approve`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  getMemory: (freelancerId?: string) =>
    apiFetch<{
      stats: Record<string, unknown>;
      semantic_memories: SemanticMemoryItem[];
      episodic_events: Array<Record<string, unknown>>;
      portfolio_items: PortfolioItem[];
    }>(`/memory${freelancerId ? `?freelancer_id=${encodeURIComponent(freelancerId)}` : ""}`),

  getAuditLogs: (server?: string) =>
    apiFetch<{ total: number; log_path: string; entries: AuditLogEntry[] }>(
      `/audit-logs${server ? `?server=${encodeURIComponent(server)}` : ""}`
    ),

  getEvalResults: () => apiFetch<EvalResultsBundle>("/evals/results"),

  streamUrl: (threadId: string) => `${API_BASE_URL}/runs/${encodeURIComponent(threadId)}/stream`,
};
