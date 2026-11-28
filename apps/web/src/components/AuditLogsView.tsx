"use client";

import React from "react";
import { Lock, RefreshCw } from "lucide-react";
import type { AuditLogEntry } from "@/lib/api/types";

interface AuditLogsViewProps {
  entries: AuditLogEntry[];
  logPath: string;
  serverFilter: string;
  onServerFilterChange: (server: string) => void;
  onRefresh: () => void;
}

export function AuditLogsView({
  entries,
  logPath,
  serverFilter,
  onServerFilterChange,
  onRefresh,
}: AuditLogsViewProps) {
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-slate-800 bg-slate-900/70 p-4">
        <div>
          <h2 className="flex items-center gap-2 text-sm font-semibold text-slate-100">
            <Lock className="h-4 w-4 text-emerald-400" />
            MCP Security & PII-Scrubbed Audit Log ({entries.length} events)
          </h2>
          <div className="mt-0.5 font-mono text-xs text-slate-400">{logPath}</div>
        </div>
        <div className="flex items-center gap-2">
          <select
            value={serverFilter}
            onChange={(e) => onServerFilterChange(e.target.value)}
            aria-label="Filter MCP server"
            className="rounded-lg border border-slate-700 bg-slate-950 px-3 py-1.5 text-xs text-slate-100"
          >
            <option value="">All MCP Servers</option>
            <option value="marketplace">marketplace (:8092)</option>
            <option value="portfolio">portfolio (:8091)</option>
            <option value="workspace">workspace (:8093)</option>
          </select>
          <button
            type="button"
            onClick={onRefresh}
            className="inline-flex items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-800 px-3 py-1.5 text-xs font-medium text-slate-200 hover:bg-slate-700"
          >
            <RefreshCw className="h-3.5 w-3.5" />
            Refresh
          </button>
        </div>
      </div>

      <div className="overflow-x-auto rounded-xl border border-slate-800 bg-slate-900/60">
        <table className="w-full text-left text-xs">
          <thead className="border-b border-slate-800 bg-slate-950/60 text-slate-400">
            <tr>
              <th className="px-3 py-2.5">Timestamp</th>
              <th className="px-3 py-2.5">Server / Tool</th>
              <th className="px-3 py-2.5">Status</th>
              <th className="px-3 py-2.5">Trace ID</th>
              <th className="px-3 py-2.5">Result Summary</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/70">
            {entries.length === 0 ? (
              <tr>
                <td colSpan={5} className="px-3 py-6 text-center text-slate-500">
                  No MCP audit log entries recorded yet.
                </td>
              </tr>
            ) : (
              entries.map((entry, idx) => (
                <tr key={`${entry.trace_id}-${idx}`} className="text-slate-300">
                  <td className="whitespace-nowrap px-3 py-2 font-mono text-[11px] text-slate-400">
                    {entry.timestamp}
                  </td>
                  <td className="px-3 py-2 font-mono">
                    <span className="text-indigo-300">{entry.server_name}</span>.
                    <span className="text-slate-100">{entry.tool_name}</span>
                  </td>
                  <td className="px-3 py-2">
                    <span
                      className={`rounded px-1.5 py-0.5 font-mono text-[11px] ${
                        entry.status === "ok"
                          ? "bg-emerald-500/20 text-emerald-300"
                          : entry.status === "dry_run"
                          ? "bg-cyan-500/20 text-cyan-300"
                          : "bg-rose-500/20 text-rose-300"
                      }`}
                    >
                      {entry.status}
                    </span>
                  </td>
                  <td className="px-3 py-2 font-mono text-[11px] text-slate-400">
                    {entry.trace_id}
                  </td>
                  <td className="px-3 py-2">{entry.result_summary}</td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
