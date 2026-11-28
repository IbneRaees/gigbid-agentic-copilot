export interface BudgetRange {
  min: number;
  max: number;
  currency: string;
  budget_type: "fixed" | "hourly";
}

export interface MarketplaceJob {
  job_id: string;
  client_id: string;
  title: string;
  description: string;
  required_skills: string[];
  nice_to_have_skills: string[];
  budget: BudgetRange;
  duration_days: number;
  experience_level: string;
  screening_questions: string[];
  connects_cost: number;
  domain_category: string;
  eval_tag?: string;
  gold_decision?: string;
}

export interface FreelancerPersona {
  freelancer_id: string;
  name: string;
  headline: string;
  primary_domain: string;
  hourly_rate_usd: number;
  min_fixed_budget_usd: number;
  connects_balance: number;
  skills: string[];
}

export interface TraceEvent {
  event_id: string;
  node: string;
  phase: string;
  timestamp: string;
  duration_ms: number;
  summary: string;
  handoff_type?: string | null;
  handoff_payload?: Record<string, unknown> | null;
  rag_evidence_ids?: string[];
  verification_score?: number | null;
  model_used?: string | null;
}

export interface RunSummary {
  run_id: string;
  thread_id: string;
  graph_type: "bid" | "conversation";
  job_id: string | null;
  freelancer_id: string;
  status: string;
  awaiting_approval: boolean;
  created_at: string;
  updated_at: string;
  interrupt_payload: Record<string, unknown> | null;
  state_snapshot: Record<string, unknown>;
}

export interface SSEEnvelope {
  event: string;
  timestamp: string;
  thread_id: string;
  data: Record<string, unknown>;
}

export interface ClientThread {
  thread_id: string;
  job_id: string;
  client_id: string;
  freelancer_id: string;
  status: string;
  messages: Array<{
    message_id: string;
    sender: "freelancer" | "client" | "system";
    text: string;
    timestamp: string;
  }>;
}

export interface PortfolioItem {
  evidence_id: string;
  freelancer_id: string;
  type: string;
  title: string;
  client_industry: string;
  date: string;
  outcome: string;
  skills: string[];
  summary: string;
  metrics: Record<string, unknown>;
  repo_url?: string | null;
}

export interface SemanticMemoryItem {
  memory_id: string;
  freelancer_id: string;
  memory_type: string;
  title: string;
  content: string;
  evidence_ids: string[];
  tags: string[];
  confidence: number;
  usage_count: number;
  updated_at: string;
}

export interface AuditLogEntry {
  timestamp: string;
  trace_id: string;
  user_id: string;
  server_name: string;
  tool_name: string;
  status: string;
  duration_ms: number;
  sanitized_args: Record<string, unknown>;
  result_summary: string;
}

export interface RagComparisonRow {
  variant: string;
  precision_at_3: number;
  recall_at_3: number;
  mrr: number;
  ndcg_at_3: number;
  grounding_fact_coverage: number;
  p50_latency_ms: number;
  p95_latency_ms: number;
}

export interface EvalResultsBundle {
  rag_comparison: {
    benchmark: string;
    embedding_model: string;
    reranker_model: string;
    recommended_variant: string;
    results: RagComparisonRow[];
  } | null;
  full_eval_report: Record<string, unknown> | null;
  extension_ablation: Record<string, unknown> | null;
  judge_calibration: Record<string, unknown> | null;
}

export interface ApiHealth {
  status: string;
  app: string;
  version: string;
  marketplace_mode: string;
  model_profile: string;
  active_runs: number;
  pending_approvals: number;
}
