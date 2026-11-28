"use client";

import React from "react";
import { BarChart3, CheckCircle2, Sparkles } from "lucide-react";
import type { EvalResultsBundle } from "@/lib/api/types";

interface EvalsDashboardViewProps {
  evals: EvalResultsBundle | null;
}

export function EvalsDashboardView({ evals }: EvalsDashboardViewProps) {
  const rag = evals?.rag_comparison;
  const full = evals?.full_eval_report as Record<string, unknown> | null;
  const ablation = evals?.extension_ablation as Record<string, unknown> | null;
  const kappa = evals?.judge_calibration as Record<string, unknown> | null;

  return (
    <div className="space-y-6">
      {/* Summary KPIs */}
      {full && (
        <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
          <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-4">
            <div className="text-xs text-slate-400">25-Case Decision Accuracy</div>
            <div className="mt-1 text-xl font-bold text-emerald-400">
              {((Number(full.decision_accuracy ?? 0)) * 100).toFixed(1)}%
            </div>
          </div>
          <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-4">
            <div className="text-xs text-slate-400">Policy Trap Block Rate</div>
            <div className="mt-1 text-xl font-bold text-emerald-400">
              {((Number(full.policy_block_rate ?? 0)) * 100).toFixed(1)}%
            </div>
          </div>
          <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-4">
            <div className="text-xs text-slate-400">Mean Grounding Score</div>
            <div className="mt-1 text-xl font-bold text-indigo-300">
              {((Number(full.mean_grounding_score ?? 0)) * 100).toFixed(1)}%
            </div>
          </div>
          <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-4">
            <div className="text-xs text-slate-400">Judge Cohen&apos;s Kappa</div>
            <div className="mt-1 text-xl font-bold text-purple-300">
              {kappa ? Number(kappa.cohens_kappa ?? 0).toFixed(3) : "—"}
            </div>
          </div>
        </div>
      )}

      {/* RAG V0-V4 Table */}
      <section className="rounded-xl border border-slate-800 bg-slate-900/70 p-4">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
          <h2 className="flex items-center gap-2 text-sm font-semibold text-slate-100">
            <BarChart3 className="h-4 w-4 text-indigo-400" />
            RAG Variant Benchmark (V0–V4)
          </h2>
          {rag && (
            <span className="text-xs text-slate-400">
              Embed: <span className="font-mono text-slate-200">{rag.embedding_model}</span> •
              Rerank: <span className="font-mono text-slate-200">{rag.reranker_model}</span>
            </span>
          )}
        </div>

        {!rag ? (
          <p className="text-xs text-slate-400">
            No RAG comparison report found. Run <span className="font-mono">make eval</span> to generate results.
          </p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="border-b border-slate-800 text-slate-400">
                <tr>
                  <th className="py-2 pr-4">Variant</th>
                  <th className="py-2 pr-4">P@3</th>
                  <th className="py-2 pr-4">R@3</th>
                  <th className="py-2 pr-4">MRR</th>
                  <th className="py-2 pr-4">nDCG@3</th>
                  <th className="py-2 pr-4">Fact Cov</th>
                  <th className="py-2 pr-4">p50 (ms)</th>
                  <th className="py-2">p95 (ms)</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/70">
                {rag.results.map((row) => (
                  <tr key={row.variant} className="text-slate-200">
                    <td className="py-2 pr-4 font-mono font-semibold text-indigo-300">
                      {row.variant.toUpperCase()}
                    </td>
                    <td className="py-2 pr-4">{row.precision_at_3.toFixed(4)}</td>
                    <td className="py-2 pr-4">{row.recall_at_3.toFixed(4)}</td>
                    <td className="py-2 pr-4">{row.mrr.toFixed(4)}</td>
                    <td className="py-2 pr-4">{row.ndcg_at_3.toFixed(4)}</td>
                    <td className="py-2 pr-4">{row.grounding_fact_coverage.toFixed(4)}</td>
                    <td className="py-2 pr-4">{row.p50_latency_ms.toFixed(1)}</td>
                    <td className="py-2">{row.p95_latency_ms.toFixed(1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {/* Extension Ablation */}
      {ablation && (
        <section className="rounded-xl border border-slate-800 bg-slate-900/70 p-4">
          <h2 className="mb-2 flex items-center gap-2 text-sm font-semibold text-slate-100">
            <Sparkles className="h-4 w-4 text-amber-400" />
            Signature Extension Ablation: Bayesian Connects ROI & Bid-Boost Optimizer
          </h2>
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <pre className="overflow-auto rounded-lg border border-slate-800 bg-slate-950 p-3 font-mono text-xs text-slate-300">
              {JSON.stringify(ablation, null, 2)}
            </pre>
            {full && (
              <div className="rounded-lg border border-slate-800 bg-slate-950 p-3 text-xs text-slate-300">
                <div className="mb-2 flex items-center gap-1.5 font-semibold text-emerald-400">
                  <CheckCircle2 className="h-4 w-4" />
                  Full 25-Case Evaluation Summary
                </div>
                <pre className="max-h-64 overflow-auto font-mono text-xs">
                  {JSON.stringify(full, null, 2)}
                </pre>
              </div>
            )}
          </div>
        </section>
      )}
    </div>
  );
}
