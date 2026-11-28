"use client";

import React, { useMemo, useState } from "react";
import { Play, Search, ShieldAlert, Sparkles, Tag } from "lucide-react";
import type { FreelancerPersona, MarketplaceJob } from "@/lib/api/types";

interface JobFeedViewProps {
  jobs: MarketplaceJob[];
  freelancers: FreelancerPersona[];
  selectedFreelancer: string;
  onSelectFreelancer: (id: string) => void;
  ragVariant: string;
  onSelectRagVariant: (v: string) => void;
  onLaunchBidRun: (jobId: string) => Promise<void>;
  launchingJobId: string | null;
}

export function JobFeedView({
  jobs,
  freelancers,
  selectedFreelancer,
  onSelectFreelancer,
  ragVariant,
  onSelectRagVariant,
  onLaunchBidRun,
  launchingJobId,
}: JobFeedViewProps) {
  const [searchQuery, setSearchQuery] = useState("");
  const [domainFilter, setDomainFilter] = useState("all");

  const filteredJobs = useMemo(() => {
    return jobs.filter((job) => {
      if (domainFilter !== "all" && job.domain_category !== domainFilter) return false;
      if (!searchQuery.trim()) return true;
      const q = searchQuery.toLowerCase();
      return (
        job.title.toLowerCase().includes(q) ||
        job.description.toLowerCase().includes(q) ||
        job.required_skills.some((s) => s.toLowerCase().includes(q)) ||
        job.job_id.toLowerCase().includes(q)
      );
    });
  }, [jobs, searchQuery, domainFilter]);

  const activePersona = freelancers.find((f) => f.freelancer_id === selectedFreelancer);

  return (
    <div className="space-y-5">
      {/* Controls Bar */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-4">
        <div className="grid grid-cols-1 gap-3 md:grid-cols-4">
          <div className="relative md:col-span-2">
            <Search className="absolute left-3 top-2.5 h-4 w-4 text-slate-400" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search GigSim jobs by title, skill, or job_id..."
              className="w-full rounded-lg border border-slate-700 bg-slate-950 py-2 pl-9 pr-3 text-sm text-slate-100 placeholder-slate-500 focus:border-indigo-500 focus:outline-none"
            />
          </div>
          <select
            value={selectedFreelancer}
            onChange={(e) => onSelectFreelancer(e.target.value)}
            aria-label="Select freelancer persona"
            className="rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100 focus:border-indigo-500 focus:outline-none"
          >
            {freelancers.map((f) => (
              <option key={f.freelancer_id} value={f.freelancer_id}>
                {f.name} ({f.primary_domain})
              </option>
            ))}
          </select>
          <div className="flex gap-2">
            <select
              value={domainFilter}
              onChange={(e) => setDomainFilter(e.target.value)}
              aria-label="Filter domain"
              className="w-1/2 rounded-lg border border-slate-700 bg-slate-950 px-2.5 py-2 text-sm text-slate-100 focus:border-indigo-500 focus:outline-none"
            >
              <option value="all">All Domains</option>
              <option value="ai_ml">AI / ML</option>
              <option value="fullstack_web">Full-Stack</option>
              <option value="game_dev">Game Dev</option>
            </select>
            <select
              value={ragVariant}
              onChange={(e) => onSelectRagVariant(e.target.value)}
              aria-label="RAG variant"
              className="w-1/2 rounded-lg border border-slate-700 bg-slate-950 px-2.5 py-2 text-sm text-slate-100 focus:border-indigo-500 focus:outline-none"
            >
              <option value="v4">RAG V4 (Hier+Policy)</option>
              <option value="v3">RAG V3 (Rerank)</option>
              <option value="v2">RAG V2 (HyDE)</option>
              <option value="v1">RAG V1 (Hybrid RRF)</option>
              <option value="v0">RAG V0 (Naive)</option>
            </select>
          </div>
        </div>

        {activePersona && (
          <div className="mt-3 flex flex-wrap items-center justify-between gap-2 border-t border-slate-800/80 pt-3 text-xs text-slate-400">
            <span>
              <strong className="text-slate-200">{activePersona.name}</strong> —{" "}
              {activePersona.headline}
            </span>
            <div className="flex items-center gap-3">
              <span className="rounded-md bg-slate-800 px-2 py-1 text-slate-200">
                Floor: ${activePersona.min_fixed_budget_usd} | Rate: ${activePersona.hourly_rate_usd}/hr
              </span>
              <span className="rounded-md bg-indigo-500/20 px-2 py-1 font-medium text-indigo-300">
                {activePersona.connects_balance} Connects
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Job Cards */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        {filteredJobs.map((job) => {
          const isTrap =
            job.eval_tag?.includes("scam") ||
            job.eval_tag?.includes("injection") ||
            job.eval_tag?.includes("lowball");
          return (
            <div
              key={job.job_id}
              data-testid={`job-card-${job.job_id}`}
              className="flex flex-col justify-between rounded-xl border border-slate-800 bg-slate-900/60 p-4 transition hover:border-slate-700"
            >
              <div>
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <div className="flex flex-wrap items-center gap-2 text-xs text-slate-400">
                      <span className="font-mono text-indigo-400">{job.job_id}</span>
                      <span>•</span>
                      <span className="uppercase tracking-wide">{job.domain_category}</span>
                      {job.eval_tag && (
                        <span
                          className={`inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[11px] font-medium ${
                            isTrap
                              ? "bg-amber-500/15 text-amber-300"
                              : "bg-emerald-500/15 text-emerald-300"
                          }`}
                        >
                          {isTrap ? <ShieldAlert className="h-3 w-3" /> : <Sparkles className="h-3 w-3" />}
                          {job.eval_tag}
                        </span>
                      )}
                    </div>
                    <h3 className="mt-1.5 text-base font-semibold text-slate-100">{job.title}</h3>
                  </div>
                  <div className="shrink-0 text-right">
                    <div className="text-sm font-semibold text-emerald-400">
                      ${job.budget.min.toLocaleString()}–${job.budget.max.toLocaleString()}
                    </div>
                    <div className="text-xs text-slate-400">
                      {job.connects_cost} Connects • {job.duration_days}d
                    </div>
                  </div>
                </div>

                <p className="mt-2.5 line-clamp-3 text-sm text-slate-300">{job.description}</p>

                <div className="mt-3 flex flex-wrap gap-1.5">
                  {job.required_skills.map((skill) => (
                    <span
                      key={skill}
                      className="inline-flex items-center gap-1 rounded-md border border-slate-700/80 bg-slate-800/70 px-2 py-0.5 text-xs text-slate-300"
                    >
                      <Tag className="h-3 w-3 text-slate-400" />
                      {skill}
                    </span>
                  ))}
                </div>
              </div>

              <div className="mt-4 flex items-center justify-between border-t border-slate-800/80 pt-3">
                <span className="text-xs text-slate-400">
                  Client: <span className="font-mono text-slate-300">{job.client_id}</span> •{" "}
                  {job.screening_questions.length} screening Qs
                </span>
                <button
                  type="button"
                  onClick={() => onLaunchBidRun(job.job_id)}
                  disabled={launchingJobId === job.job_id}
                  data-testid={`launch-bid-${job.job_id}`}
                  className="inline-flex items-center gap-1.5 rounded-lg bg-indigo-600 px-3.5 py-1.5 text-xs font-semibold text-white transition hover:bg-indigo-500 disabled:opacity-50"
                >
                  <Play className="h-3.5 w-3.5" />
                  {launchingJobId === job.job_id ? "Launching..." : "Run Copilot"}
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
