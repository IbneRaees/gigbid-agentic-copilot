"use client";

import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  Activity,
  BarChart3,
  Brain,
  Briefcase,
  CheckSquare,
  Lock,
  MessageSquare,
  Shield,
} from "lucide-react";
import { gigbidApi } from "@/lib/api/client";
import type {
  ApiHealth,
  AuditLogEntry,
  ClientThread,
  EvalResultsBundle,
  FreelancerPersona,
  MarketplaceJob,
  PortfolioItem,
  RunSummary,
  SSEEnvelope,
  SemanticMemoryItem,
  TraceEvent,
} from "@/lib/api/types";
import { ApprovalQueueView } from "@/components/ApprovalQueueView";
import { AuditLogsView } from "@/components/AuditLogsView";
import { ConversationView } from "@/components/ConversationView";
import { EvalsDashboardView } from "@/components/EvalsDashboardView";
import { JobFeedView } from "@/components/JobFeedView";
import { LiveTraceView } from "@/components/LiveTraceView";
import { PortfolioMemoryView } from "@/components/PortfolioMemoryView";

type TabId =
  | "jobs"
  | "trace"
  | "approvals"
  | "conversations"
  | "memory"
  | "evals"
  | "audit";

export default function HomePage() {
  const [activeTab, setActiveTab] = useState<TabId>("jobs");
  const [health, setHealth] = useState<ApiHealth | null>(null);
  const [jobs, setJobs] = useState<MarketplaceJob[]>([]);
  const [freelancers, setFreelancers] = useState<FreelancerPersona[]>([]);
  const [selectedFreelancer, setSelectedFreelancer] = useState("persona_aiml_01");
  const [ragVariant, setRagVariant] = useState("v4");

  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [activeThreadId, setActiveThreadId] = useState<string | null>(null);
  const [eventsByThread, setEventsByThread] = useState<Record<string, TraceEvent[]>>({});
  const [launchingJobId, setLaunchingJobId] = useState<string | null>(null);
  const [launchingConvId, setLaunchingConvId] = useState<string | null>(null);
  const [submittingApproval, setSubmittingApproval] = useState(false);

  const [threads, setThreads] = useState<ClientThread[]>([]);
  const [portfolioItems, setPortfolioItems] = useState<PortfolioItem[]>([]);
  const [semanticMemories, setSemanticMemories] = useState<SemanticMemoryItem[]>([]);
  const [episodicCount, setEpisodicCount] = useState(0);
  const [evals, setEvals] = useState<EvalResultsBundle | null>(null);
  const [auditEntries, setAuditEntries] = useState<AuditLogEntry[]>([]);
  const [auditLogPath, setAuditLogPath] = useState("logs/mcp_audit.jsonl");
  const [auditServerFilter, setAuditServerFilter] = useState("");
  const [errorBanner, setErrorBanner] = useState<string | null>(null);

  const sseRef = useRef<EventSource | null>(null);

  const refreshCoreData = useCallback(async () => {
    try {
      const [h, j, f, r, th, mem, ev] = await Promise.all([
        gigbidApi.getHealth(),
        gigbidApi.listJobs({ freelancer_id: selectedFreelancer }),
        gigbidApi.listFreelancers(),
        gigbidApi.listRuns(),
        gigbidApi.listThreads(),
        gigbidApi.getMemory(),
        gigbidApi.getEvalResults(),
      ]);
      setHealth(h);
      setJobs(j.jobs);
      setFreelancers(f.freelancers);
      setRuns(r);
      setThreads(th.threads);
      setPortfolioItems(mem.portfolio_items);
      setSemanticMemories(mem.semantic_memories);
      setEpisodicCount(mem.episodic_events.length);
      setEvals(ev);
      setErrorBanner(null);
    } catch (err) {
      setErrorBanner(err instanceof Error ? err.message : "Failed to reach GigBid API");
    }
  }, [selectedFreelancer]);

  const refreshAuditLogs = useCallback(async () => {
    try {
      const res = await gigbidApi.getAuditLogs(auditServerFilter || undefined);
      setAuditEntries(res.entries);
      setAuditLogPath(res.log_path);
    } catch {
      // ignore if offline
    }
  }, [auditServerFilter]);

  useEffect(() => {
    void refreshCoreData();
    void refreshAuditLogs();
  }, [refreshCoreData, refreshAuditLogs]);

  const subscribeToRun = useCallback((threadId: string) => {
    if (sseRef.current) {
      sseRef.current.close();
      sseRef.current = null;
    }
    const es = new EventSource(gigbidApi.streamUrl(threadId));
    sseRef.current = es;

    const handleNodeUpdate = (e: MessageEvent) => {
      try {
        const env = JSON.parse(e.data) as SSEEnvelope;
        const traceEvt = env.data?.trace_event as TraceEvent | undefined;
        if (traceEvt) {
          setEventsByThread((prev) => {
            const existing = prev[threadId] || [];
            if (existing.some((x) => x.event_id === traceEvt.event_id)) return prev;
            return { ...prev, [threadId]: [...existing, traceEvt] };
          });
        }
      } catch {
        // ignore parse errors
      }
    };

    const handleTerminalOrInterrupt = async () => {
      try {
        const updatedRun = await gigbidApi.getRun(threadId);
        setRuns((prev) => {
          const others = prev.filter((r) => r.thread_id !== threadId);
          return [updatedRun, ...others];
        });
        await refreshCoreData();
        await refreshAuditLogs();
        if (updatedRun.awaiting_approval) {
          setActiveTab("approvals");
        }
      } catch {
        // ignore
      }
    };

    es.addEventListener("node_update", handleNodeUpdate);
    es.addEventListener("interrupt", () => {
      void handleTerminalOrInterrupt();
    });
    es.addEventListener("run_completed", () => {
      void handleTerminalOrInterrupt();
      es.close();
    });
    es.onerror = () => {
      es.close();
    };
  }, [refreshCoreData, refreshAuditLogs]);

  useEffect(() => {
    return () => {
      if (sseRef.current) sseRef.current.close();
    };
  }, []);

  const handleLaunchBidRun = async (jobId: string) => {
    setLaunchingJobId(jobId);
    try {
      const run = await gigbidApi.startBidRun({
        job_id: jobId,
        freelancer_id: selectedFreelancer,
        rag_variant: ragVariant,
      });
      setRuns((prev) => [run, ...prev.filter((r) => r.thread_id !== run.thread_id)]);
      setActiveThreadId(run.thread_id);
      setActiveTab("trace");
      subscribeToRun(run.thread_id);
    } catch (err) {
      setErrorBanner(err instanceof Error ? err.message : "Failed to start bid run");
    } finally {
      setLaunchingJobId(null);
    }
  };

  const handleLaunchConversation = async (threadId: string, freelancerId: string) => {
    setLaunchingConvId(threadId);
    try {
      const run = await gigbidApi.startConversationRun({
        thread_id: threadId,
        freelancer_id: freelancerId,
      });
      setRuns((prev) => [run, ...prev.filter((r) => r.thread_id !== run.thread_id)]);
      setActiveThreadId(run.thread_id);
      setActiveTab("trace");
      subscribeToRun(run.thread_id);
    } catch (err) {
      setErrorBanner(err instanceof Error ? err.message : "Failed to start conversation run");
    } finally {
      setLaunchingConvId(null);
    }
  };

  const handleApproveRun = async (
    threadId: string,
    payload: {
      approved: boolean;
      reason?: string;
      cover_letter?: string;
      proposed_rate?: number;
      reply_text?: string;
      selected_slot?: string;
    }
  ) => {
    setSubmittingApproval(true);
    try {
      const updated = await gigbidApi.approveRun(threadId, payload);
      setRuns((prev) => [updated, ...prev.filter((r) => r.thread_id !== threadId)]);
      setActiveThreadId(updated.thread_id);
      setActiveTab("trace");
      await refreshCoreData();
      await refreshAuditLogs();
    } catch (err) {
      setErrorBanner(err instanceof Error ? err.message : "Approval failed");
    } finally {
      setSubmittingApproval(false);
    }
  };

  const pendingApprovals = runs.filter((r) => r.awaiting_approval);

  const navItems: Array<{ id: TabId; label: string; icon: React.ReactNode; badge?: number }> = [
    { id: "jobs", label: "Job Feed & Scout", icon: <Briefcase className="h-4 w-4" /> },
    { id: "trace", label: "Live Trace & Hand-offs", icon: <Activity className="h-4 w-4" /> },
    {
      id: "approvals",
      label: "HITL Approval Queue",
      icon: <CheckSquare className="h-4 w-4" />,
      badge: pendingApprovals.length,
    },
    { id: "conversations", label: "Liaison & Calendar", icon: <MessageSquare className="h-4 w-4" /> },
    { id: "memory", label: "Portfolio & Memory", icon: <Brain className="h-4 w-4" /> },
    { id: "evals", label: "RAG & Evals", icon: <BarChart3 className="h-4 w-4" /> },
    { id: "audit", label: "MCP Audit Logs", icon: <Lock className="h-4 w-4" /> },
  ];

  return (
    <div className="min-h-screen bg-[#090d16] text-slate-100">
      {/* Top Bar */}
      <header className="sticky top-0 z-20 border-b border-slate-800/80 bg-[#090d16]/90 backdrop-blur">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-4 px-4 py-3">
          <div className="flex items-center gap-2.5">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-indigo-600 text-white shadow-lg shadow-indigo-600/20">
              <Shield className="h-5 w-5" />
            </div>
            <div>
              <h1 className="text-base font-bold tracking-tight text-white">
                GigBid <span className="font-normal text-indigo-400">Copilot</span>
              </h1>
              <p className="text-[11px] text-slate-400">
                LangGraph • Hybrid RAG + ClaimVerifier • MCP Guardrails • Mandatory HITL
              </p>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2 text-xs">
            <span className="rounded-md border border-slate-800 bg-slate-900 px-2.5 py-1 text-slate-300">
              Mode: <strong className="text-emerald-400">{health?.marketplace_mode || "gigsim"}</strong>
            </span>
            <span className="rounded-md border border-slate-800 bg-slate-900 px-2.5 py-1 text-slate-300">
              Profile: <strong className="text-indigo-300">{health?.model_profile || "open"}</strong>
            </span>
            <span className="rounded-md border border-slate-800 bg-slate-900 px-2.5 py-1 text-slate-300">
              Auto-Send: <strong className="text-rose-400">OFF</strong>
            </span>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav className="mx-auto flex max-w-7xl gap-1 overflow-x-auto px-4 pb-2">
          {navItems.map((tab) => {
            const active = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                type="button"
                data-testid={`tab-${tab.id}`}
                onClick={() => setActiveTab(tab.id)}
                className={`inline-flex shrink-0 items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium transition ${
                  active
                    ? "bg-indigo-600 text-white"
                    : "text-slate-400 hover:bg-slate-800/80 hover:text-slate-200"
                }`}
              >
                {tab.icon}
                {tab.label}
                {tab.badge !== undefined && tab.badge > 0 && (
                  <span className="ml-1 rounded-full bg-amber-400 px-1.5 py-0.2 text-[10px] font-bold text-slate-950">
                    {tab.badge}
                  </span>
                )}
              </button>
            );
          })}
        </nav>
      </header>

      <main className="mx-auto max-w-7xl px-4 py-6">
        {errorBanner && (
          <div className="mb-4 rounded-xl border border-rose-500/40 bg-rose-500/10 px-4 py-2.5 text-xs text-rose-300">
            {errorBanner}
          </div>
        )}

        {activeTab === "jobs" && (
          <JobFeedView
            jobs={jobs}
            freelancers={freelancers}
            selectedFreelancer={selectedFreelancer}
            onSelectFreelancer={setSelectedFreelancer}
            ragVariant={ragVariant}
            onSelectRagVariant={setRagVariant}
            onLaunchBidRun={handleLaunchBidRun}
            launchingJobId={launchingJobId}
          />
        )}

        {activeTab === "trace" && (
          <LiveTraceView
            runs={runs}
            activeThreadId={activeThreadId}
            onSelectRun={(tid) => {
              setActiveThreadId(tid);
              subscribeToRun(tid);
            }}
            eventsByThread={eventsByThread}
            onGoToApproval={() => setActiveTab("approvals")}
          />
        )}

        {activeTab === "approvals" && (
          <ApprovalQueueView
            pendingRuns={pendingApprovals}
            onApprove={handleApproveRun}
            submitting={submittingApproval}
          />
        )}

        {activeTab === "conversations" && (
          <ConversationView
            threads={threads}
            onLaunchConversation={handleLaunchConversation}
            launchingThreadId={launchingConvId}
          />
        )}

        {activeTab === "memory" && (
          <PortfolioMemoryView
            portfolioItems={portfolioItems}
            semanticMemories={semanticMemories}
            episodicCount={episodicCount}
          />
        )}

        {activeTab === "evals" && <EvalsDashboardView evals={evals} />}

        {activeTab === "audit" && (
          <AuditLogsView
            entries={auditEntries}
            logPath={auditLogPath}
            serverFilter={auditServerFilter}
            onServerFilterChange={setAuditServerFilter}
            onRefresh={() => void refreshAuditLogs()}
          />
        )}
      </main>
    </div>
  );
}
