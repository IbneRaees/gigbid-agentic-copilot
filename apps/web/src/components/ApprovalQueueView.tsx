"use client";

import React, { useEffect, useState } from "react";
import { Check, Coins, ShieldCheck, ThumbsDown, ThumbsUp } from "lucide-react";
import type { RunSummary } from "@/lib/api/types";

interface ApprovalQueueViewProps {
  pendingRuns: RunSummary[];
  onApprove: (
    threadId: string,
    payload: {
      approved: boolean;
      reason?: string;
      cover_letter?: string;
      proposed_rate?: number;
      reply_text?: string;
      selected_slot?: string;
    }
  ) => Promise<void>;
  submitting: boolean;
}

export function ApprovalQueueView({
  pendingRuns,
  onApprove,
  submitting,
}: ApprovalQueueViewProps) {
  const [selectedThreadId, setSelectedThreadId] = useState<string | null>(null);
  const active =
    pendingRuns.find((r) => r.thread_id === selectedThreadId) || pendingRuns[0] || null;

  const payload = active?.interrupt_payload || {};
  const isBid = active?.graph_type === "bid";
  const draft = (payload.proposal_draft as Record<string, unknown> | undefined) || {};
  const replyDraft = (payload.reply_draft as Record<string, unknown> | undefined) || {};
  const dryRun = (payload.dry_run_preview as Record<string, unknown> | undefined) || {};
  const previewSummary =
    (dryRun.dry_run_preview as Record<string, unknown> | undefined) || {};
  const snapshot = active?.state_snapshot || {};
  const verReport = (snapshot.verification_report as Record<string, unknown> | undefined) || {};
  const claims = (verReport.claims as Array<Record<string, unknown>> | undefined) || [];

  const [editedText, setEditedText] = useState("");
  const [editedRate, setEditedRate] = useState<string>("");
  const [rejectReason, setRejectReason] = useState("Hold Connects for higher priority role");

  useEffect(() => {
    if (!active) return;
    if (isBid) {
      setEditedText(String(draft.cover_letter || ""));
      setEditedRate(String(draft.proposed_rate || ""));
    } else {
      setEditedText(String(replyDraft.reply_text || ""));
      setEditedRate("");
    }
  }, [active?.thread_id, isBid, draft.cover_letter, draft.proposed_rate, replyDraft.reply_text]);

  if (pendingRuns.length === 0) {
    return (
      <div
        data-testid="approval-empty"
        className="rounded-xl border border-slate-800 bg-slate-900/60 p-10 text-center"
      >
        <ShieldCheck className="mx-auto h-10 w-10 text-emerald-400" />
        <h3 className="mt-3 text-base font-semibold text-slate-100">
          Human-in-the-Loop Approval Queue is Clear
        </h3>
        <p className="mt-1 text-sm text-slate-400">
          No proposals or client replies are waiting at an interrupt gate. Launch a Bid Run on a valid
          job (e.g. <span className="font-mono text-indigo-300">job_001</span>) to review a dry-run preview here.
        </p>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 gap-5 lg:grid-cols-3">
      {/* Queue List */}
      <div className="space-y-2 rounded-xl border border-slate-800 bg-slate-900/70 p-4">
        <h2 className="text-sm font-semibold text-slate-200">
          Awaiting Human Approval ({pendingRuns.length})
        </h2>
        {pendingRuns.map((run) => (
          <button
            key={run.thread_id}
            type="button"
            onClick={() => setSelectedThreadId(run.thread_id)}
            className={`w-full rounded-lg border p-3 text-left ${
              active?.thread_id === run.thread_id
                ? "border-amber-500 bg-amber-500/10"
                : "border-slate-800 bg-slate-950/60"
            }`}
          >
            <div className="flex items-center justify-between text-xs">
              <span className="font-mono font-semibold text-amber-300">
                {run.job_id || run.graph_type}
              </span>
              <span className="rounded bg-amber-500/20 px-1.5 py-0.5 text-[11px] text-amber-200">
                INTERRUPT
              </span>
            </div>
            <div className="mt-1 truncate font-mono text-[11px] text-slate-400">
              {run.thread_id}
            </div>
          </button>
        ))}
      </div>

      {/* Review & Edit Pane */}
      {active && (
        <div className="space-y-4 lg:col-span-2">
          {/* Dry-Run Preview & Grounding Header */}
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
            <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-3.5">
              <div className="text-xs text-slate-400">ClaimVerifier Grounding</div>
              <div className="mt-1 flex items-center gap-2 text-lg font-bold text-emerald-400">
                <ShieldCheck className="h-5 w-5" />
                {(((payload.grounding_score as number | undefined) ?? 1.0) * 100).toFixed(0)}%
              </div>
              <div className="text-xs text-slate-500">
                POL-001..009: {payload.compliance_passed !== false ? "PASSED" : "BLOCKED"}
              </div>
            </div>

            <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-3.5">
              <div className="text-xs text-slate-400">Dry-Run Connects Preview</div>
              <div className="mt-1 flex items-center gap-2 text-lg font-bold text-indigo-300">
                <Coins className="h-5 w-5" />
                {String(previewSummary.connects_cost ?? draft.connects_cost ?? 0)} Connects
              </div>
              <div className="text-xs text-slate-500">
                Charged so far: 0 (mode={String(payload.marketplace_mode || "gigsim")})
              </div>
            </div>

            <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-3.5">
              <div className="text-xs text-slate-400">Net Earnings (After POL-007 Fee)</div>
              <div className="mt-1 text-lg font-bold text-slate-100">
                ${Number(previewSummary.estimated_net_usd ?? editedRate ?? 0).toLocaleString()}
              </div>
              <div className="text-xs text-slate-500">
                Fee: ${Number(previewSummary.estimated_fee_usd ?? 0).toLocaleString()}
              </div>
            </div>
          </div>

          {/* Editable Draft */}
          <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-4 space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <h3 className="text-sm font-semibold text-slate-100">
                {isBid
                  ? `Editable Proposal Draft — ${String(payload.job_title || active.job_id)}`
                  : `Editable Client Reply — ${active.thread_id}`}
              </h3>
              {isBid && (
                <label className="flex items-center gap-2 text-xs text-slate-300">
                  Proposed Rate (USD):
                  <input
                    type="number"
                    value={editedRate}
                    onChange={(e) => setEditedRate(e.target.value)}
                    className="w-28 rounded border border-slate-700 bg-slate-950 px-2 py-1 font-mono text-xs text-slate-100"
                  />
                </label>
              )}
            </div>

            <textarea
              rows={7}
              value={editedText}
              onChange={(e) => setEditedText(e.target.value)}
              className="w-full rounded-lg border border-slate-700 bg-slate-950 p-3 text-sm text-slate-100 focus:border-indigo-500 focus:outline-none"
            />

            {/* Claim Grounding Badges */}
            {claims.length > 0 && (
              <div className="space-y-1.5 border-t border-slate-800 pt-3">
                <div className="text-xs font-semibold text-slate-300">
                  Sentence-Level Claim Verification ({claims.length} sentences)
                </div>
                <div className="max-h-40 space-y-1.5 overflow-auto pr-1">
                  {claims.map((c, idx) => (
                    <div
                      key={idx}
                      className="flex items-start justify-between gap-2 rounded-lg border border-slate-800 bg-slate-950/70 px-2.5 py-1.5 text-xs"
                    >
                      <span className="text-slate-300">{String(c.claim_text)}</span>
                      <span
                        className={`shrink-0 rounded px-1.5 py-0.5 font-mono text-[10px] ${
                          c.status === "SUPPORTED"
                            ? "bg-emerald-500/20 text-emerald-300"
                            : c.status === "GENERIC"
                            ? "bg-slate-800 text-slate-400"
                            : "bg-rose-500/20 text-rose-300"
                        }`}
                      >
                        {String(c.status)}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Approve / Reject Actions */}
            <div className="flex flex-wrap items-center justify-between gap-3 border-t border-slate-800 pt-3">
              <div className="flex items-center gap-2">
                <input
                  type="text"
                  value={rejectReason}
                  onChange={(e) => setRejectReason(e.target.value)}
                  placeholder="Rejection reason..."
                  className="w-56 rounded-lg border border-slate-700 bg-slate-950 px-2.5 py-1.5 text-xs text-slate-200"
                />
                <button
                  type="button"
                  disabled={submitting}
                  onClick={() =>
                    onApprove(active.thread_id, {
                      approved: false,
                      reason: rejectReason,
                    })
                  }
                  className="inline-flex items-center gap-1.5 rounded-lg border border-rose-500/40 bg-rose-500/10 px-3 py-1.5 text-xs font-semibold text-rose-300 hover:bg-rose-500/20 disabled:opacity-50"
                >
                  <ThumbsDown className="h-3.5 w-3.5" />
                  Reject
                </button>
              </div>

              <button
                type="button"
                disabled={submitting}
                data-testid="approve-submit-btn"
                onClick={() =>
                  onApprove(
                    active.thread_id,
                    isBid
                      ? {
                          approved: true,
                          cover_letter: editedText,
                          proposed_rate: Number(editedRate) || undefined,
                        }
                      : {
                          approved: true,
                          reply_text: editedText,
                          selected_slot: String(payload.selected_slot || ""),
                        }
                  )
                }
                className="inline-flex items-center gap-1.5 rounded-lg bg-emerald-600 px-4 py-2 text-xs font-semibold text-white hover:bg-emerald-500 disabled:opacity-50"
              >
                <ThumbsUp className="h-3.5 w-3.5" />
                {submitting ? "Submitting..." : isBid ? "Approve & Submit Proposal" : "Approve & Send Reply"}
                <Check className="h-3.5 w-3.5" />
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
