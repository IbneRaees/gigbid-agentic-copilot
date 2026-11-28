"use client";

import React, { useState } from "react";
import { CalendarClock, MessageSquare, Send } from "lucide-react";
import type { ClientThread } from "@/lib/api/types";

interface ConversationViewProps {
  threads: ClientThread[];
  onLaunchConversation: (threadId: string, freelancerId: string) => Promise<void>;
  launchingThreadId: string | null;
}

export function ConversationView({
  threads,
  onLaunchConversation,
  launchingThreadId,
}: ConversationViewProps) {
  const [selectedId, setSelectedId] = useState<string>(threads[0]?.thread_id || "");
  const activeThread =
    threads.find((t) => t.thread_id === selectedId) || threads[0] || null;

  return (
    <div className="grid grid-cols-1 gap-5 lg:grid-cols-3">
      <div className="space-y-2 rounded-xl border border-slate-800 bg-slate-900/70 p-4">
        <h2 className="mb-2 flex items-center gap-2 text-sm font-semibold text-slate-200">
          <MessageSquare className="h-4 w-4 text-indigo-400" />
          Client Threads ({threads.length})
        </h2>
        {threads.map((t) => (
          <button
            key={t.thread_id}
            type="button"
            onClick={() => setSelectedId(t.thread_id)}
            className={`w-full rounded-lg border p-3 text-left transition ${
              activeThread?.thread_id === t.thread_id
                ? "border-indigo-500 bg-indigo-500/10"
                : "border-slate-800 bg-slate-950/60 hover:border-slate-700"
            }`}
          >
            <div className="flex items-center justify-between text-xs">
              <span className="font-mono font-semibold text-slate-200">{t.thread_id}</span>
              <span className="rounded bg-slate-800 px-1.5 py-0.5 text-[11px] text-slate-300">
                {t.status}
              </span>
            </div>
            <div className="mt-1 text-xs text-slate-400">
              Job: {t.job_id} • Client: {t.client_id}
            </div>
          </button>
        ))}
      </div>

      <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-4 lg:col-span-2">
        {!activeThread ? (
          <p className="text-sm text-slate-400">No client threads found.</p>
        ) : (
          <div className="space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800 pb-3">
              <div>
                <h3 className="text-base font-semibold text-slate-100">
                  Thread {activeThread.thread_id} ({activeThread.job_id})
                </h3>
                <p className="text-xs text-slate-400">
                  Freelancer: {activeThread.freelancer_id} • Client: {activeThread.client_id}
                </p>
              </div>
              <button
                type="button"
                disabled={launchingThreadId === activeThread.thread_id}
                onClick={() =>
                  onLaunchConversation(activeThread.thread_id, activeThread.freelancer_id)
                }
                className="inline-flex items-center gap-1.5 rounded-lg bg-indigo-600 px-3.5 py-2 text-xs font-semibold text-white hover:bg-indigo-500 disabled:opacity-50"
              >
                <CalendarClock className="h-4 w-4" />
                {launchingThreadId === activeThread.thread_id
                  ? "Drafting Liaison Reply..."
                  : "Run Liaison & Calendar Check"}
                <Send className="h-3.5 w-3.5" />
              </button>
            </div>

            <div className="space-y-3">
              {activeThread.messages.map((m) => (
                <div
                  key={m.message_id}
                  className={`rounded-xl border p-3.5 ${
                    m.sender === "client"
                      ? "border-indigo-500/30 bg-indigo-950/25"
                      : "border-slate-800 bg-slate-950/60"
                  }`}
                >
                  <div className="flex items-center justify-between text-xs text-slate-400">
                    <span className="font-semibold uppercase tracking-wide text-slate-200">
                      {m.sender}
                    </span>
                    <span>{m.timestamp}</span>
                  </div>
                  <p className="mt-1.5 text-sm text-slate-200">{m.text}</p>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
