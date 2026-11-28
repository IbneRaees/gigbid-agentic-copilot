"use client";

import React from "react";
import { Brain, Database, FolderGit2 } from "lucide-react";
import type { PortfolioItem, SemanticMemoryItem } from "@/lib/api/types";

interface PortfolioMemoryViewProps {
  portfolioItems: PortfolioItem[];
  semanticMemories: SemanticMemoryItem[];
  episodicCount: number;
}

export function PortfolioMemoryView({
  portfolioItems,
  semanticMemories,
  episodicCount,
}: PortfolioMemoryViewProps) {
  return (
    <div className="space-y-6">
      {/* Promoted Semantic Memories */}
      <section className="rounded-xl border border-slate-800 bg-slate-900/70 p-4">
        <div className="mb-3 flex items-center justify-between">
          <h2 className="flex items-center gap-2 text-sm font-semibold text-slate-100">
            <Brain className="h-4 w-4 text-purple-400" />
            Promoted Semantic Memories ({semanticMemories.length})
          </h2>
          <span className="text-xs text-slate-400">
            Episodic events captured: {episodicCount}
          </span>
        </div>

        {semanticMemories.length === 0 ? (
          <p className="text-xs text-slate-400">
            No semantic memories promoted yet. Approve a grounded proposal or process a skipped
            policy trap to populate promoted patterns & Q&A snippets.
          </p>
        ) : (
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
            {semanticMemories.map((mem) => (
              <div
                key={mem.memory_id}
                className="rounded-lg border border-slate-800 bg-slate-950/70 p-3"
              >
                <div className="flex items-center justify-between text-xs">
                  <span className="rounded bg-purple-500/20 px-1.5 py-0.5 font-mono text-purple-300">
                    {mem.memory_type}
                  </span>
                  <span className="text-slate-400">
                    Conf {(mem.confidence * 100).toFixed(0)}% • Uses {mem.usage_count}
                  </span>
                </div>
                <div className="mt-1.5 text-sm font-semibold text-slate-200">{mem.title}</div>
                <p className="mt-1 text-xs text-slate-300">{mem.content}</p>
                {mem.evidence_ids.length > 0 && (
                  <div className="mt-2 flex flex-wrap gap-1">
                    {mem.evidence_ids.map((eid) => (
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
            ))}
          </div>
        )}
      </section>

      {/* Verified Portfolio Corpus */}
      <section className="rounded-xl border border-slate-800 bg-slate-900/70 p-4">
        <h2 className="mb-3 flex items-center gap-2 text-sm font-semibold text-slate-100">
          <Database className="h-4 w-4 text-cyan-400" />
          Verified Portfolio Evidence Corpus ({portfolioItems.length} items)
        </h2>
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
          {portfolioItems.map((item) => (
            <div
              key={item.evidence_id}
              className="rounded-lg border border-slate-800 bg-slate-950/60 p-3.5"
            >
              <div className="flex items-center justify-between text-xs">
                <span className="font-mono font-semibold text-cyan-300">{item.evidence_id}</span>
                <span
                  className={`rounded px-1.5 py-0.5 text-[11px] font-medium ${
                    item.outcome === "won"
                      ? "bg-emerald-500/20 text-emerald-300"
                      : "bg-slate-800 text-slate-300"
                  }`}
                >
                  {item.type} • {item.outcome}
                </span>
              </div>
              <h3 className="mt-1.5 text-sm font-semibold text-slate-100">{item.title}</h3>
              <p className="mt-1 text-xs text-slate-300">{item.summary}</p>
              <div className="mt-2 flex flex-wrap gap-1">
                {item.skills.slice(0, 6).map((sk) => (
                  <span
                    key={sk}
                    className="rounded bg-slate-800 px-1.5 py-0.5 text-[11px] text-slate-300"
                  >
                    {sk}
                  </span>
                ))}
              </div>
              {item.repo_url && (
                <div className="mt-2 flex items-center gap-1 font-mono text-[11px] text-indigo-300">
                  <FolderGit2 className="h-3.5 w-3.5" />
                  {item.repo_url}
                </div>
              )}
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
