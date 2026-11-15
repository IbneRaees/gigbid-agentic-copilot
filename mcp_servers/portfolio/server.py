"""Strictly Read-Only Portfolio MCP Server (`port 8091`) built with official MCP SDK.

Exposes:
  Tools (`readOnlyHint=True, destructiveHint=False, idempotentHint=True`):
    1. `search_evidence(requirement: str, k: int = 5, filters: dict | None = None)`
    2. `get_case_study(case_id: str)`
    3. `list_skills(freelancer_id: str | None = None)`

  Resource Templates:
    - `portfolio://case/{id}`
    - `portfolio://proposal/{id}`

  Prompt Templates:
    - `answer_screening_question`
    - `draft_evidence_anchored_opener`

Backed by `data/portfolio_items.json` and `data/freelancers.json`, with Bearer
authentication, per-user rate limiting, and JSONL audit logging.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

import uvicorn

from mcp_servers.security import (
    MCPAuditLogger,
    MCPServer,
    SlidingWindowRateLimiter,
    build_secured_mcp_asgi_app,
    make_tool_annotations,
)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_DIR = Path(os.getenv("GIGSIM_DATA_DIR", str(ROOT_DIR / "data")))

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_+#.\-]+")
_OUTCOME_MULTIPLIERS = {
    "won": 1.35,
    "interview": 1.20,
    "replied": 1.10,
    "viewed": 1.02,
    "ignored": 0.85,
}


def _tokenize(text: str) -> set[str]:
    return {tok.lower() for tok in _TOKEN_RE.findall(text) if len(tok) > 1}


class PortfolioRepository:
    """Read-only repository over `data/portfolio_items.json` and `data/freelancers.json`."""

    def __init__(self, data_dir: Path = DEFAULT_DATA_DIR) -> None:
        self.data_dir = Path(data_dir)
        self._items: list[dict[str, Any]] = []
        self._freelancers: list[dict[str, Any]] = []
        self.reload()

    def reload(self) -> None:
        portfolio_path = self.data_dir / "portfolio_items.json"
        freelancers_path = self.data_dir / "freelancers.json"
        if portfolio_path.exists():
            self._items = json.loads(portfolio_path.read_text(encoding="utf-8"))
        else:
            self._items = []
        if freelancers_path.exists():
            self._freelancers = json.loads(freelancers_path.read_text(encoding="utf-8"))
        else:
            self._freelancers = []

    def search_evidence(
        self,
        requirement: str,
        k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        query_tokens = _tokenize(requirement)
        req_lower = requirement.lower().strip()
        active_filters = filters or {}

        filter_freelancer = active_filters.get("freelancer_id")
        filter_type = active_filters.get("type")
        filter_outcome = active_filters.get("outcome")
        filter_skill = (
            str(active_filters["skill"]).lower().strip()
            if active_filters.get("skill")
            else None
        )
        exclude_ignored = bool(active_filters.get("exclude_ignored", False))

        scored: list[tuple[float, dict[str, Any]]] = []
        for item in self._items:
            if filter_freelancer and item.get("freelancer_id") != filter_freelancer:
                continue
            if filter_type and item.get("type") != filter_type:
                continue
            if filter_outcome and item.get("outcome") != filter_outcome:
                continue
            if exclude_ignored and item.get("outcome") == "ignored":
                continue

            item_skills = [str(s) for s in item.get("skills", [])]
            item_skills_lower = {s.lower() for s in item_skills}
            if filter_skill and not any(filter_skill in s for s in item_skills_lower):
                continue

            title = str(item.get("title", ""))
            summary = str(item.get("summary", ""))
            full_text = str(item.get("full_text", ""))
            industry = str(item.get("client_industry", ""))

            doc_tokens = _tokenize(f"{title} {summary} {full_text} {industry} {' '.join(item_skills)}")
            overlap = len(query_tokens & doc_tokens)
            lexical_score = overlap / max(1, len(query_tokens))

            skill_hits = sum(
                1 for sk in item_skills_lower if sk in req_lower or _tokenize(sk) <= query_tokens
            )
            skill_bonus = 0.25 * min(3, skill_hits)

            outcome = str(item.get("outcome", "viewed"))
            outcome_mult = _OUTCOME_MULTIPLIERS.get(outcome, 1.0)

            raw_score = (lexical_score + skill_bonus) * outcome_mult
            if raw_score <= 0.0 and query_tokens:
                # Keep minimal fallback score only when filtering narrowly
                raw_score = 0.01 * outcome_mult

            scored.append(
                (
                    round(raw_score, 4),
                    {
                        "evidence_id": item["evidence_id"],
                        "freelancer_id": item.get("freelancer_id", ""),
                        "type": item.get("type", "case_study"),
                        "title": title,
                        "client_industry": industry,
                        "date": item.get("date", ""),
                        "outcome": outcome,
                        "skills": item_skills,
                        "summary": summary,
                        "metrics": item.get("metrics", {}),
                        "repo_url": item.get("repo_url"),
                        "relevance_score": round(raw_score, 4),
                    },
                )
            )

        scored.sort(key=lambda pair: (-pair[0], pair[1]["evidence_id"]))
        top_k = max(1, min(int(k), 25))
        return [entry for _, entry in scored[:top_k]]

    def get_case_study(self, case_id: str) -> dict[str, Any]:
        for item in self._items:
            if item.get("evidence_id") == case_id:
                return dict(item)
        raise KeyError(f"Portfolio evidence item '{case_id}' not found.")

    def list_skills(self, freelancer_id: str | None = None) -> dict[str, Any]:
        skill_map: dict[str, dict[str, Any]] = {}

        for fl in self._freelancers:
            if freelancer_id and fl.get("freelancer_id") != freelancer_id:
                continue
            for sk in fl.get("skills", []):
                entry = skill_map.setdefault(
                    sk,
                    {"skill": sk, "evidence_ids": [], "project_count": 0, "won_count": 0},
                )

        for item in self._items:
            if freelancer_id and item.get("freelancer_id") != freelancer_id:
                continue
            ev_id = item["evidence_id"]
            is_won = item.get("outcome") == "won"
            for sk in item.get("skills", []):
                entry = skill_map.setdefault(
                    sk,
                    {"skill": sk, "evidence_ids": [], "project_count": 0, "won_count": 0},
                )
                if ev_id not in entry["evidence_ids"]:
                    entry["evidence_ids"].append(ev_id)
                    entry["project_count"] += 1
                    if is_won:
                        entry["won_count"] += 1

        skills_list = sorted(
            skill_map.values(),
            key=lambda x: (-x["project_count"], -x["won_count"], x["skill"].lower()),
        )
        return {
            "freelancer_id": freelancer_id or "all",
            "total_skills": len(skills_list),
            "skills": skills_list,
        }

    def get_freelancer_profile(self, freelancer_id: str) -> dict[str, Any]:
        for fl in self._freelancers:
            if fl.get("freelancer_id") == freelancer_id:
                return dict(fl)
        return self._freelancers[0] if self._freelancers else {"freelancer_id": freelancer_id}


def build_portfolio_mcp_server(
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
    audit_logger: MCPAuditLogger | None = None,
) -> MCPServer:
    """Construct and configure the strictly read-only Portfolio MCP server."""
    repo = PortfolioRepository(data_dir=data_dir)
    audit = audit_logger or MCPAuditLogger()
    mcp = MCPServer("gigbid-portfolio-mcp")

    # ------------------------------------------------------------------
    # 3 STRICTLY READ-ONLY TOOLS
    # ------------------------------------------------------------------

    @mcp.tool(
        name="search_evidence",
        description=(
            "Search verified freelancer portfolio case studies, GitHub repositories, CVs, "
            "and past winning proposals matching a job requirement. Strictly read-only."
        ),
        annotations=make_tool_annotations(
            read_only=True, destructive=False, idempotent=True, open_world=False
        ),
    )
    async def search_evidence(
        requirement: str,
        k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        raw_args = {"requirement": requirement, "k": k, "filters": filters}
        try:
            matches = repo.search_evidence(requirement=requirement, k=k, filters=filters)
            result = {
                "requirement": requirement,
                "k": k,
                "total_returned": len(matches),
                "matches": matches,
            }
            audit.record(
                server_name="portfolio",
                tool_name="search_evidence",
                raw_args=raw_args,
                result_summary=f"returned={len(matches)}, top={matches[0]['evidence_id'] if matches else 'none'}",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
            )
            return result
        except Exception as exc:
            audit.record(
                server_name="portfolio",
                tool_name="search_evidence",
                raw_args=raw_args,
                result_summary=f"error: {exc}",
                status="error",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
            )
            raise

    @mcp.tool(
        name="get_case_study",
        description=(
            "Retrieve full verifiable details, metrics, repository URL, and narrative text "
            "for a specific portfolio evidence_id (e.g., 'ev_aiml_001'). Strictly read-only."
        ),
        annotations=make_tool_annotations(
            read_only=True, destructive=False, idempotent=True, open_world=False
        ),
    )
    async def get_case_study(case_id: str) -> dict[str, Any]:
        t0 = time.perf_counter()
        raw_args = {"case_id": case_id}
        try:
            item = repo.get_case_study(case_id)
            audit.record(
                server_name="portfolio",
                tool_name="get_case_study",
                raw_args=raw_args,
                result_summary=f"case_id={case_id}, outcome={item.get('outcome')}",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
            )
            return item
        except Exception as exc:
            audit.record(
                server_name="portfolio",
                tool_name="get_case_study",
                raw_args=raw_args,
                result_summary=f"error: {exc}",
                status="error",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
            )
            raise

    @mcp.tool(
        name="list_skills",
        description=(
            "List verified technical skills for a freelancer persona along with supporting "
            "portfolio evidence_ids and project counts. Strictly read-only."
        ),
        annotations=make_tool_annotations(
            read_only=True, destructive=False, idempotent=True, open_world=False
        ),
    )
    async def list_skills(freelancer_id: str | None = None) -> dict[str, Any]:
        t0 = time.perf_counter()
        raw_args = {"freelancer_id": freelancer_id}
        try:
            result = repo.list_skills(freelancer_id=freelancer_id)
            audit.record(
                server_name="portfolio",
                tool_name="list_skills",
                raw_args=raw_args,
                result_summary=f"freelancer_id={result['freelancer_id']}, total_skills={result['total_skills']}",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
            )
            return result
        except Exception as exc:
            audit.record(
                server_name="portfolio",
                tool_name="list_skills",
                raw_args=raw_args,
                result_summary=f"error: {exc}",
                status="error",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
            )
            raise

    # ------------------------------------------------------------------
    # RESOURCE TEMPLATES (`portfolio://case/{id}`, `portfolio://proposal/{id}`)
    # ------------------------------------------------------------------

    @mcp.resource(
        "portfolio://case/{id}",
        name="portfolio_case_study",
        description="Read a verified portfolio case study or repository record by evidence ID.",
        mime_type="application/json",
    )
    def read_case_resource(id: str) -> str:
        item = repo.get_case_study(id)
        return json.dumps(item, indent=2, ensure_ascii=False)

    @mcp.resource(
        "portfolio://proposal/{id}",
        name="portfolio_past_proposal",
        description="Read a historical proposal record by evidence ID.",
        mime_type="application/json",
    )
    def read_proposal_resource(id: str) -> str:
        item = repo.get_case_study(id)
        return json.dumps(item, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # PROMPT TEMPLATES
    # ------------------------------------------------------------------

    @mcp.prompt(
        name="answer_screening_question",
        description=(
            "Generate a grounded screening-question answer prompt backed by retrieved "
            "portfolio evidence items and freelancer style rules."
        ),
    )
    def answer_screening_question_prompt(
        question: str,
        job_context: str = "",
        freelancer_id: str = "persona_aiml_01",
    ) -> str:
        profile = repo.get_freelancer_profile(freelancer_id)
        evidence = repo.search_evidence(
            requirement=f"{question} {job_context}".strip(),
            k=3,
            filters={"freelancer_id": freelancer_id, "exclude_ignored": True},
        )
        evidence_block = "\n".join(
            f"- [{ev['evidence_id']}] {ev['title']} (outcome={ev['outcome']}): "
            f"{ev['summary']} | Metrics: {json.dumps(ev['metrics'])}"
            for ev in evidence
        )
        style_rules = "\n".join(
            f"- {rule}" for rule in profile.get("style_preferences", [])
        )
        return (
            f"You are drafting a client screening question response for {profile.get('name', freelancer_id)} "
            f"({profile.get('headline', '')}).\n\n"
            f"SCREENING QUESTION:\n{question}\n\n"
            f"JOB CONTEXT:\n{job_context or 'N/A'}\n\n"
            f"VERIFIED PORTFOLIO EVIDENCE (Cite ONLY from these evidence_ids; do NOT invent metrics):\n"
            f"{evidence_block or '- No matching evidence found; answer conservatively.'}\n\n"
            f"STYLE GUIDELINES:\n{style_rules}\n\n"
            f"Write a concise, direct 3-5 sentence answer citing exact evidence_ids in brackets."
        )

    @mcp.prompt(
        name="draft_evidence_anchored_opener",
        description=(
            "Generate a 2-sentence proposal opening paragraph anchored in the top matching "
            "portfolio case study without generic filler."
        ),
    )
    def draft_evidence_anchored_opener_prompt(
        job_title: str,
        key_requirement: str,
        freelancer_id: str = "persona_aiml_01",
    ) -> str:
        profile = repo.get_freelancer_profile(freelancer_id)
        evidence = repo.search_evidence(
            requirement=f"{job_title} {key_requirement}",
            k=2,
            filters={"freelancer_id": freelancer_id, "exclude_ignored": True},
        )
        evidence_block = "\n".join(
            f"- [{ev['evidence_id']}] {ev['title']}: {ev['summary']} (Metrics: {json.dumps(ev['metrics'])})"
            for ev in evidence
        )
        return (
            f"Write a crisp 2-sentence opening hook for '{job_title}' addressing '{key_requirement}'.\n"
            f"Use ONLY verified facts from:\n{evidence_block}\n"
            f"Forbidden phrases: 'I am passionate', 'Dear Sir/Madam', 'I read your job post carefully'."
        )

    return mcp


def create_app(
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
    expected_token: str | None = None,
    rate_limit_rpm: int | None = None,
    rate_limiter: SlidingWindowRateLimiter | None = None,
    audit_logger: MCPAuditLogger | None = None,
):
    """Create the secured Starlette ASGI application for Portfolio MCP."""
    server = build_portfolio_mcp_server(data_dir=data_dir, audit_logger=audit_logger)
    return build_secured_mcp_asgi_app(
        server,
        server_name="portfolio",
        expected_token=expected_token,
        rate_limit_rpm=rate_limit_rpm,
        rate_limiter=rate_limiter,
    )


def main() -> None:
    host = os.getenv("PORTFOLIO_MCP_HOST", "0.0.0.0")
    port = int(os.getenv("PORTFOLIO_MCP_PORT", "8091"))
    asgi_app = create_app()
    uvicorn.run(asgi_app, host=host, port=port)


if __name__ == "__main__":
    main()