"""Fit Matcher Node (Parallel Fan-Out): grounds job requirements in Portfolio RAG + MCP."""

from __future__ import annotations

import time
from typing import Any

from agent.handoffs import FitReport, RequirementMatch
from agent.llm import ModelRouter
from agent.nodes._runtime import get_mcp_client_manager
from agent.state import BidGraphState, make_budget_delta, make_trace_event
from rag.index.hybrid_store import tokenize_technical
from rag.retriever import EvidenceChunk, GigBidRetriever, VariantId

_SHARED_RETRIEVER: GigBidRetriever | None = None


def _get_retriever() -> GigBidRetriever:
    global _SHARED_RETRIEVER
    if _SHARED_RETRIEVER is None:
        _SHARED_RETRIEVER = GigBidRetriever(auto_build=True)
    return _SHARED_RETRIEVER


async def _fetch_portfolio_mcp_skills(freelancer_id: str, trace_id: str) -> set[str]:
    """Query Portfolio MCP `list_skills` if MCP server is running."""
    mcp_mgr = get_mcp_client_manager(trace_id=trace_id, user_id=freelancer_id)
    if mcp_mgr is None:
        return set()
    try:
        async with mcp_mgr as mcp:
            res = await mcp.call_tool("portfolio", "list_skills", {"freelancer_id": freelancer_id})
            return {str(s["skill"]).lower() for s in res.get("skills", []) if "skill" in s}
    except Exception:
        return set()


def _heuristic_fit(
    *,
    raw_job: dict[str, Any],
    freelancer_profile: dict[str, Any],
    evidence_chunks: list[EvidenceChunk],
    mcp_skills: set[str],
) -> FitReport:
    job_id = str(raw_job.get("job_id", "unknown"))
    freelancer_id = str(freelancer_profile.get("freelancer_id", "persona_aiml_01"))
    req_skills = [str(s) for s in raw_job.get("required_skills", [])]
    nice_skills = [str(s) for s in raw_job.get("nice_to_have_skills", [])]

    profile_skills = {str(s).lower() for s in freelancer_profile.get("skills", [])} | mcp_skills
    chunk_skill_map: list[tuple[str, set[str], str]] = []
    for ch in evidence_chunks:
        ev_id = ch.evidence_id or ch.chunk_id
        if ch.doc_type == "freelancer_profile":
            continue
        toks = set(tokenize_technical(f"{ch.title} {ch.parent_content or ch.content}"))
        toks |= {s.lower() for s in ch.skills}
        chunk_skill_map.append((ev_id, toks, (ch.content or "")[:180]))

    req_matches: list[RequirementMatch] = []
    matched_count = 0
    missing_skills: list[str] = []
    matched_ev_ids: list[str] = []

    for skill in req_skills:
        sk_lower = skill.lower()
        sk_tokens = set(tokenize_technical(sk_lower))
        supporting_evs: list[str] = []
        best_snippet = ""

        for ev_id, ev_toks, snippet in chunk_skill_map:
            if sk_lower in ev_toks or (
                sk_tokens and len(sk_tokens & ev_toks) >= max(1, len(sk_tokens) // 2)
            ):
                supporting_evs.append(ev_id)
                if not best_snippet:
                    best_snippet = snippet

        in_profile = sk_lower in profile_skills or any(
            sk_lower in ps or ps in sk_lower for ps in profile_skills
        )
        matched = bool(supporting_evs) or in_profile
        if matched:
            matched_count += 1
            for eid in supporting_evs:
                if eid not in matched_ev_ids:
                    matched_ev_ids.append(eid)
        else:
            missing_skills.append(skill)

        conf = 0.92 if supporting_evs else (0.76 if in_profile else 0.15)
        req_matches.append(
            RequirementMatch(
                requirement=skill,
                matched=matched,
                confidence=conf,
                evidence_ids=supporting_evs[:3],
                supporting_snippet=best_snippet,
                gap_note=None if matched else f"No verified portfolio evidence for '{skill}'.",
            )
        )

    skill_coverage = (matched_count / len(req_skills)) if req_skills else 0.5
    job_domain = str(raw_job.get("domain_category", "general"))
    freelancer_domain = str(freelancer_profile.get("primary_domain", ""))
    domain_relevance = (
        0.95
        if job_domain == freelancer_domain
        else (0.55 if job_domain in {"ai_ml", "fullstack_web"} else 0.15)
    )

    nice_bonus = 0.0
    if nice_skills:
        nice_hits = sum(1 for sk in nice_skills if sk.lower() in profile_skills)
        nice_bonus = 0.05 * (nice_hits / len(nice_skills))

    ev_bonus = 0.08 if matched_ev_ids else 0.0
    fit_score = round(
        min(1.0, 0.65 * skill_coverage + 0.27 * domain_relevance + nice_bonus + ev_bonus),
        3,
    )
    if not matched_ev_ids and skill_coverage < 0.25:
        fit_score = round(min(fit_score, 0.22), 3)

    # Ensure top retrieved portfolio case studies are retained even if matched primarily via semantic search
    for ch in evidence_chunks:
        if (
            ch.evidence_id
            and ch.doc_type != "freelancer_profile"
            and ch.evidence_id not in matched_ev_ids
        ):
            if fit_score >= 0.45:
                matched_ev_ids.append(ch.evidence_id)

    strengths = [
        f"Verified evidence for {m.requirement} ({', '.join(m.evidence_ids)})"
        for m in req_matches
        if m.matched and m.evidence_ids
    ][:4]
    weaknesses = [f"Missing direct portfolio item for {sk}" for sk in missing_skills[:3]]

    return FitReport(
        job_id=job_id,
        freelancer_id=freelancer_id,
        fit_score=fit_score,
        skill_coverage=round(skill_coverage, 3),
        domain_relevance=round(domain_relevance, 3),
        requirement_matches=req_matches,
        matched_evidence_ids=matched_ev_ids[:5],
        missing_skills=missing_skills,
        strengths=strengths or ["Profile domain aligns with core deliverable."],
        weaknesses=weaknesses,
        rationale=(
            f"SkillCoverage={skill_coverage:.0%}, DomainRelevance={domain_relevance:.2f}, "
            f"Evidence={matched_ev_ids[:4]} -> FitScore={fit_score:.2f}."
        ),
    )


async def fit_matcher_node(state: BidGraphState) -> dict[str, Any]:
    """Retrieve grounded portfolio evidence (`GigBidRetriever`) and score freelancer fit."""
    t0 = time.perf_counter()
    raw_job = state.get("raw_job") or {}
    freelancer_profile = state.get("freelancer_profile") or {}
    freelancer_id = str(
        freelancer_profile.get("freelancer_id") or state.get("freelancer_id", "persona_aiml_01")
    )
    variant: VariantId = state.get("rag_variant", "v4")  # type: ignore[assignment]
    trace_id = state.get("trace_id", "trace-local")

    req_skills = [str(s) for s in raw_job.get("required_skills", [])]
    query = f"{raw_job.get('title', '')}. {raw_job.get('description', '')[:300]} Skills: {', '.join(req_skills)}"

    retriever = _get_retriever()
    evidence_chunks = retriever.retrieve_portfolio(
        query=query,
        k=5,
        freelancer_id=freelancer_id,
        required_skills=req_skills,
        variant=variant,
    )
    # Also fetch cross-persona evidence only if same-persona yielded nothing, keeping persona strict otherwise
    if not evidence_chunks:
        evidence_chunks = retriever.retrieve_portfolio(
            query=query,
            k=4,
            freelancer_id=freelancer_id,
            variant=variant,
        )

    mcp_skills = await _fetch_portfolio_mcp_skills(freelancer_id, trace_id)
    heuristic_fit = _heuristic_fit(
        raw_job=raw_job,
        freelancer_profile=freelancer_profile,
        evidence_chunks=evidence_chunks,
        mcp_skills=mcp_skills,
    )

    router = ModelRouter(profile=state.get("model_profile"))
    system_prompt = (
        "You are GigBid's Fit Matcher. Produce a FitReport strictly grounded in the retrieved "
        "portfolio evidence_ids. Never invent evidence IDs or unverified skills."
    )
    ev_summary = [
        {
            "evidence_id": c.evidence_id,
            "title": c.title,
            "skills": c.skills,
            "outcome": c.outcome_status,
            "summary": c.content[:220],
        }
        for c in evidence_chunks
        if c.doc_type != "freelancer_profile"
    ]
    user_prompt = (
        f"Freelancer Profile: {freelancer_profile}\n"
        f"Required Skills: {req_skills}\n"
        f"Retrieved Evidence: {ev_summary}"
    )

    report, usage = await router.invoke_structured(
        node_name="fit_matcher",
        schema=FitReport,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        fallback_fn=lambda: heuristic_fit,
        state=state,
    )

    # Keep evidence_ids strictly anchored to real retrieved IDs
    valid_ids = {c.evidence_id for c in evidence_chunks if c.evidence_id}
    cleaned_ev_ids = [eid for eid in report.matched_evidence_ids if eid in valid_ids]
    if not cleaned_ev_ids and heuristic_fit.matched_evidence_ids:
        cleaned_ev_ids = heuristic_fit.matched_evidence_ids
    if cleaned_ev_ids != report.matched_evidence_ids:
        report = report.model_copy(update={"matched_evidence_ids": cleaned_ev_ids})

    duration_ms = (time.perf_counter() - t0) * 1000.0
    dumped = report.model_dump(mode="json")
    ev_dicts = [c.model_dump(mode="json") for c in evidence_chunks]

    event = make_trace_event(
        node="fit_matcher",
        phase="completed",
        duration_ms=duration_ms,
        summary=(
            f"FitScore={report.fit_score:.2f} (skill_cov={report.skill_coverage:.0%}) | "
            f"evidence={report.matched_evidence_ids}"
        ),
        handoff_type="FitReport",
        handoff_payload=dumped,
        rag_evidence_ids=report.matched_evidence_ids,
        model_used=usage["model"],
    )
    return {
        "fit_report": dumped,
        "retrieved_evidence": ev_dicts,
        "cited_evidence_ids": report.matched_evidence_ids,
        "budgets": make_budget_delta(
            steps=1,
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            estimated_cost_usd=usage["estimated_cost_usd"],
        ),
        "node_timings": {"fit_matcher": round(duration_ms, 2)},
        "trace_events": [event],
    }


__all__ = ["fit_matcher_node"]
