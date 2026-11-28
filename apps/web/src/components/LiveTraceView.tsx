"use client";

import React, { useState } from "react";
import { Activity, CheckCircle2, ChevronDown, ChevronRight, Clock, FileCheck, PauseCircle, XCircle } from "lucide-react";
import type { RunSummary, TraceEvent } from "@/lib/api/types";

interface LiveTraceViewProps {
  runs: RunSummary[];
  activeThreadId: string | null;
  onSelectRun: (threadId: string) => void;
  eventsByThread: Record<string, TraceEvent[]>;
  onGoToApproval: () => void;
}

export function LiveTraceView({
  runs,
  activeThreadId,
  onSelectRun,
  eventsByThread,
  onGoToApproval,
}: LiveTraceViewProps) {
  const [expandedEvent, setExpandedEvent] = useState<string | null>(null);
  const activeRun = runs.find((r) => r.thread_id === activeThreadId) || runs[0] || null;
  const snapshot = activeRun?.state_snapshot || {};
  const snapshotEvents = (snapshot.trace_events as TraceEvent[] | undefined) || [];
  const liveEvents = activeRun ? eventsByThread[activeRun.thread_id] || [] : [];
  const events = liveEvents.length >= snapshotEvents.length ? liveEvents : snapshotEvents;
  const budgets = (snapshot.budgets as Record<string, number> | undefined) || {};

  return (
    <div className="grid grid-cols-1 gap-5 lg:grid-cols-3">
      {/* Left Run Selector */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-4">
        <h2 className="mb-3 flex items-center gap-2 text-sm font-semibold text-slate-200">
          <Activity className="h-4 w-4 text-indigo-400" />
          Agent Runs ({runs.length})
        </h2>
        <div className="space-y-2">
          {runs.length === 0 && (
            <p className="text-xs text-slate-500">
              Launch a Bid Run from the Job Feed or a Conversation Liaison run to inspect live SSE traces.
            </p>
          )}
          {runs.map((run) => {
            const selected = activeRun?.thread_id === run.thread_id;
            return (
              <button
                key={run.thread_id}
                type="button"
                onClick={() => onSelectRun(run.thread_id)}
                className={`w-full rounded-lg border p-3 text-left transition ${
                  selected
                    ? "border-indigo-500 bg-indigo-500/10"
                    : "border-slate-800 bg-slate-950/60 hover:border-slate-700"
                }`}
              >
                <div className="flex items-center justify-between text-xs">
                  <span className="font-mono font-semibold text-slate-200">
                    {run.job_id || run.graph_type}
                  </span>
                  <span
                    className={`rounded px-1.5 py-0.5 text-[11px] font-medium ${
                      run.awaiting_approval
                        ? "bg-amber-500/20 text-amber-300"
                        : run.status === "skipped" || run.status === "rejected_by_human"
                        ? "bg-slate-700 text-slate-300"
                        : run.status === "error" || run.status === "blocked_by_compliance"
                        ? "bg-rose-500/20 text-rose-300"
                        : "bg-emerald-500/20 text-emerald-300"
                    }`}
                  >
                    {run.status}
                  </span>
                </div>
                <div className="mt-1 truncate font-mono text-[11px] text-slate-500">
                  {run.thread_id}
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* Right Timeline + Hand-off Inspector */}
      <div className="space-y-4 lg:col-span-2">
        {!activeRun ? (
          <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-8 text-center text-sm text-slate-400">
            Select or launch a run to inspect live node transitions and Pydantic hand-off payloads.
          </div>
        ) : (
          <>
            <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-slate-800 bg-slate-900/70 p-4">
              <div>
                <div className="text-xs font-mono text-indigo-400">{activeRun.thread_id}</div>
                <h2 className="text-base font-semibold text-slate-100">
                  {activeRun.graph_type === "bid" ? "Bid-or-Skip & Proposal Pipeline" : "Client Liaison Pipeline"}
                </h2>
                <div className="mt-1 flex flex-wrap gap-3 text-xs text-slate-400">
                  <span>Steps: {budgets.steps_used ?? events.length}/{budgets.max_steps ?? 24}</span>
                  <span>Tokens: {budgets.tokens_used ?? 0}</span>
                  <span>Revisions: {budgets.revision_count ?? 0}/{budgets.max_revisions ?? 2}</span>
                  <span>Cost: ${(budgets.estimated_cost_usd ?? 0).toFixed(4)}</span>
                </div>
              </div>
              {activeRun.awaiting_approval && (
                <button
                  type="button"
                  onClick={onGoToApproval}
                  className="inline-flex items-center gap-1.5 rounded-lg bg-amber-500 px-3.5 py-2 text-xs font-semibold text-slate-950 hover:bg-amber-400"
                >
                  <PauseCircle className="h-4 w-4" />
                  Review Interrupt & Approve
                </button>
              )}
            </div>

            <div className="space-y-2.5">
              {events.map((evt, idx) => {
                const key = evt.event_id || `${evt.node}-${idx}`;
                const isOpen = expandedEvent === key;
                return (
                  <div
                    key={key}
                    className="rounded-xl border border-slate-800 bg-slate-900/60 p-3.5"
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div className="flex items-start gap-2.5">
                        {evt.phase === "error" ? (
                          <XCircle className="mt-0.5 h-4 w-4 shrink-0 text-rose-400" />
                        ) : evt.phase === "skipped" ? (
                          <PauseCircle className="mt-0.5 h-4 w-4 shrink-0 text-amber-400" />
                        ) : (
                          <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-emerald-400" />
                        )}
                        <div>
                          <div className="flex flex-wrap items-center gap-2">
                            <span className="font-mono text-xs font-semibold text-indigo-300">
                              {evt.node}
                            </span>
                            {evt.handoff_type && (
                              <span className="rounded bg-indigo-500/20 px-1.5 py-0.5 font-mono text-[11px] text-indigo-200">
                                {evt.handoff_type}
                              </span>
                            )}
                            {evt.verification_score !== null && evt.verification_score !== undefined && (
                              <span className="inline-flex items-center gap-1 rounded bg-emerald-500/20 px-1.5 py-0.5 text-[11px] text-emerald-300">
                                <FileCheck className="h-3 w-3" />
                                Grounding {(evt.verification_score * 100).toFixed(0)}%
                              </span>
                            )}
                            {evt.model_used && (
                              <span className="font-mono text-[11px] text-slate-500">
                                {evt.model_used}
                              </span>
                            )}
                          </div>
                          <p className="mt-1 text-sm text-slate-300">{evt.summary}</p>
                          {evt.rag_evidence_ids && evt.rag_evidence_ids.length > 0 && (
                            <div className="mt-1.5 flex flex-wrap gap-1">
                              {evt.rag_evidence_ids.map((eid) => (
                                <span
                                  key={eid}
                                  className="rounded bg-slate-800 px-1.5 py-0.5 font-mono text-[11px] text-cyan-300"
                                >
                                  {eid}
                                </span>
                              ))}
                            </div>
                          )}
                        </div>
                      </div>

                      <div className="flex shrink-0 items-center gap-2 text-xs text-slate-400">
                        <span className="inline-flex items-center gap-1">
                          <Clock className="h-3 w-3" />
                          {evt.duration_ms.toFixed(1)}ms
                        </span>
                        {evt.handoff_payload && (
                          <button
                            type="button"
                            onClick={() => setExpandedEvent(isOpen ? null : key)}
                            className="rounded border border-slate-700 px-2 py-0.5 text-[11px] text-slate-300 hover:bg-slate-800"
                          >
                            {isOpen ? (
                              <span className="inline-flex items-center gap-0.5">
                                Hide JSON <ChevronDown className="h-3 w-3" />
                              </span>
                            ) : (
                              <span className="inline-flex items-center gap-0.5">
                                Hand-off JSON <ChevronRight className="h-3 w-3" />
                              </span>
                            )}
                          </button>
                        )}
                      </div>
                    </div>

                    {isOpen && evt.handoff_payload && (
                      <pre className="mt-3 max-h-72 overflow-auto rounded-lg border border-slate-800 bg-slate-950 p-3 font-mono text-xs text-slate-300">
                        {JSON.stringify(evt.handoff_payload, null, 2)}
                      </pre>
                    )}
                  </div>
                );
              })}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
