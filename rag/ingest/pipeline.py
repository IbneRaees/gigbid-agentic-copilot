"""Structure-aware ingestion and chunking pipeline for GigBid.

Ingests:
  1. data/portfolio_items.json (case studies, GitHub projects, past proposals)
  2. data/freelancers.json (freelancer personas, rates, skills, bios)
  3. data/policy_pack/marketplace_policy.md (rule-delimited policy pack)
  4. Optional GitHub repository READMEs / Markdown docs (data/github_repos/*.md)

Every emitted DocumentChunk preserves parent linkage (`parent_id`, `parent_text`),
citation identifiers (`evidence_id`, `policy_rule_id`), skills metadata, and
strict `data_provenance` ("kaggle_real" | "synthetic").
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

ProvenanceType = Literal["kaggle_real", "synthetic"]
NamespaceType = Literal["portfolio", "policy"]
DocType = Literal[
    "case_study",
    "github_repo",
    "past_proposal",
    "freelancer_profile",
    "policy_rule",
]


class DocumentChunk(BaseModel):
    """Normalized, structure-aware chunk ready for hybrid dense + sparse indexing."""

    chunk_id: str = Field(..., description="Deterministic unique chunk identifier")
    parent_id: str = Field(..., description="Parent document ID for hierarchical expansion")
    evidence_id: str | None = Field(
        default=None,
        description="Citable portfolio evidence ID (e.g., ev_aiml_01) when applicable",
    )
    namespace: NamespaceType = Field(..., description="Retrieval namespace: portfolio or policy")
    doc_type: DocType = Field(..., description="Semantic document category")
    title: str = Field(..., description="Parent document or section title")
    section_title: str = Field(default="", description="Sub-section or aspect heading")
    text: str = Field(..., description="Chunk text used for embedding and lexical indexing")
    parent_text: str = Field(
        ...,
        description="Complete parent document/rule text returned by hierarchical retrieval (V4)",
    )
    freelancer_id: str | None = Field(
        default=None,
        description="Owning freelancer persona ID (None for shared policy documents)",
    )
    skills: list[str] = Field(default_factory=list, description="Associated normalized skills")
    outcome_status: str | None = Field(
        default=None,
        description="Historical outcome if applicable (won, replied, interviewed, completed, etc.)",
    )
    policy_rule_id: str | None = Field(
        default=None,
        description="Structured marketplace policy rule code (e.g., POL-001)",
    )
    data_provenance: ProvenanceType = Field(
        default="synthetic",
        description="Dataset lineage tag: kaggle_real or synthetic",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestResult(BaseModel):
    """Summary statistics produced by an ingestion run."""

    chunks: list[DocumentChunk]
    total_chunks: int
    portfolio_chunks: int
    policy_chunks: int
    parent_documents: int
    sources_loaded: list[str]


def _stable_id(prefix: str, *parts: str) -> str:
    raw = "::".join(str(p) for p in parts)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _normalize_skills(raw_skills: Any) -> list[str]:
    if not raw_skills:
        return []
    if isinstance(raw_skills, str):
        items = [s.strip() for s in raw_skills.split(",")]
    elif isinstance(raw_skills, list):
        items = [str(s).strip() for s in raw_skills]
    else:
        items = []
    seen: set[str] = set()
    cleaned: list[str] = []
    for item in items:
        if item and item.lower() not in seen:
            seen.add(item.lower())
            cleaned.append(item)
    return cleaned


def _normalize_provenance(value: Any) -> ProvenanceType:
    if str(value).strip().lower() == "kaggle_real":
        return "kaggle_real"
    return "synthetic"


def _builtin_fallback_portfolio() -> list[dict[str, Any]]:
    """Deterministic fallback portfolio items if data/portfolio_items.json is absent."""
    return [
        {
            "evidence_id": "ev_aiml_01",
            "freelancer_id": "persona_aiml_01",
            "type": "case_study",
            "title": "Enterprise Legal Contract Copilot with LangGraph & pgvector",
            "problem": "Law firm associates spent 6 hours per NDA reviewing indemnity and liability clauses manually.",
            "solution": "Built a multi-agent LangGraph state machine with hybrid pgvector + BM25 clause retrieval, PostgresSaver checkpointing, and human-in-the-loop redline approval.",
            "outcome": "Reduced contract review turnaround by 62% with 99.1% citation grounding across 14,000+ agreements.",
            "skills": ["Python", "LangGraph", "FastAPI", "PostgreSQL", "pgvector", "RAG"],
            "client_industry": "LegalTech",
            "date": "2026-06-15",
            "source": "https://github.com/alex-chen-ai/legal-contract-copilot",
            "outcome_status": "won",
            "data_provenance": "synthetic",
        },
        {
            "evidence_id": "ev_aiml_02",
            "freelancer_id": "persona_aiml_01",
            "type": "github_repo",
            "title": "High-Throughput Async FastAPI + Celery Document Ingestion Engine",
            "problem": "Financial analytics startup faced 45-second API timeouts when ingesting multi-hundred-page PDF filings.",
            "solution": "Architected an async FastAPI gateway backed by Redis + Celery workers, idempotency keys, and BGE cross-encoder reranking.",
            "outcome": "Cut p95 ingestion latency from 45s to 2.8s and processed 120,000 SEC filings with zero duplicate writes.",
            "skills": ["Python", "FastAPI", "Redis", "Celery", "Docker", "Reranking"],
            "client_industry": "FinTech",
            "date": "2026-04-20",
            "source": "https://github.com/alex-chen-ai/async-doc-pipeline",
            "outcome_status": "completed",
            "data_provenance": "synthetic",
        },
        {
            "evidence_id": "ev_web_01",
            "freelancer_id": "persona_web_02",
            "type": "case_study",
            "title": "Real-Time B2B SaaS Operations Portal in Next.js, TypeScript & FastAPI",
            "problem": "Logistics platform suffered from stale dashboards and manual CSV reconciliation across 35 enterprise tenants.",
            "solution": "Delivered a Next.js 15 App Router + TypeScript frontend connected via SSE streaming to a multi-tenant FastAPI + PostgreSQL backend.",
            "outcome": "Eliminated manual CSV workflows, saving 18 hours/week per operations manager with 99.95% uptime.",
            "skills": ["Next.js", "TypeScript", "React", "FastAPI", "PostgreSQL", "SSE"],
            "client_industry": "Supply Chain",
            "date": "2026-07-10",
            "source": "https://github.com/priya-nair-dev/ops-pulse-portal",
            "outcome_status": "won",
            "data_provenance": "synthetic",
        },
    ]


def _builtin_fallback_freelancers() -> list[dict[str, Any]]:
    """Deterministic fallback freelancer profiles if data/freelancers.json is absent."""
    return [
        {
            "freelancer_id": "persona_aiml_01",
            "name": "Alex Chen",
            "headline": "Senior AI Systems & Agentic Backend Engineer (LangGraph, RAG, FastAPI)",
            "bio": "6 years building production LLM agents, hybrid RAG search engines, and fault-tolerant FastAPI backends on Ubuntu and Docker.",
            "years_experience": 6,
            "hourly_rate": 95.0,
            "min_fixed_budget": 1500.0,
            "skills": [
                "Python",
                "LangGraph",
                "FastAPI",
                "PostgreSQL",
                "pgvector",
                "RAG",
                "Redis",
                "Docker",
            ],
            "data_provenance": "synthetic",
        },
        {
            "freelancer_id": "persona_web_02",
            "name": "Priya Nair",
            "headline": "Full-Stack TypeScript, Next.js & Python Product Engineer",
            "bio": "5 years shipping end-to-end SaaS applications, real-time streaming UIs, and Playwright-tested web workflows.",
            "years_experience": 5,
            "hourly_rate": 85.0,
            "min_fixed_budget": 1200.0,
            "skills": [
                "TypeScript",
                "Next.js",
                "React",
                "Python",
                "FastAPI",
                "PostgreSQL",
                "Playwright",
            ],
            "data_provenance": "synthetic",
        },
    ]


def _builtin_fallback_policy() -> str:
    """Deterministic fallback policy pack if data/policy_pack/marketplace_policy.md is absent."""
    return """# GigBid Marketplace Policy Pack

## POL-001: Off-Platform Contact & Anti-Circumvention
Freelancers and clients must never share personal email addresses, phone numbers, Telegram/WhatsApp handles, or external payment links prior to an active on-platform contract.

## POL-002: On-Platform Payment & Escrow Compliance
All milestone and hourly payments must be processed exclusively through the marketplace escrow system. Proposals must never offer PayPal, crypto, wire transfer, or direct invoicing discounts.

## POL-003: Prohibition on Unpaid Speculative Work
Freelancers must not submit completed production code, proprietary datasets, or unpaid custom prototypes inside initial proposals.

## POL-004: Truthful Experience & Grounded Claims
Every quantitative metric, client outcome, and technical skill claimed in a proposal must be backed by verifiable portfolio evidence. Fabricated credentials or unverified guarantees are strictly prohibited.

## POL-005: Human-in-the-Loop Approval Gate
Automated agents may draft proposals and replies, but every outbound submission and Connects expenditure requires explicit human approval.
"""


class IngestionPipeline:
    """Structure-aware document loader and hierarchical chunker."""

    def __init__(
        self,
        data_dir: str | Path = "data",
        github_repos_dir: str | Path | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.github_repos_dir = (
            Path(github_repos_dir)
            if github_repos_dir is not None
            else self.data_dir / "github_repos"
        )

    def run(self) -> IngestResult:
        """Load all configured data sources and return structured chunks."""
        chunks: list[DocumentChunk] = []
        sources_loaded: list[str] = []

        # 1. Portfolio items
        portfolio_path = self.data_dir / "portfolio_items.json"
        if portfolio_path.exists():
            raw_portfolio = json.loads(portfolio_path.read_text(encoding="utf-8"))
            if isinstance(raw_portfolio, dict) and "items" in raw_portfolio:
                raw_portfolio = raw_portfolio["items"]
            sources_loaded.append(str(portfolio_path))
        else:
            logger.warning(
                "Portfolio file %s not found; using built-in fallback items.", portfolio_path
            )
            raw_portfolio = _builtin_fallback_portfolio()
            sources_loaded.append("builtin://portfolio_items")

        for item in raw_portfolio:
            chunks.extend(self._chunk_portfolio_item(item))

        # 2. Freelancer personas
        freelancers_path = self.data_dir / "freelancers.json"
        if freelancers_path.exists():
            raw_freelancers = json.loads(freelancers_path.read_text(encoding="utf-8"))
            if isinstance(raw_freelancers, dict) and "freelancers" in raw_freelancers:
                raw_freelancers = raw_freelancers["freelancers"]
            sources_loaded.append(str(freelancers_path))
        else:
            logger.warning(
                "Freelancers file %s not found; using built-in fallback profiles.", freelancers_path
            )
            raw_freelancers = _builtin_fallback_freelancers()
            sources_loaded.append("builtin://freelancers")

        for profile in raw_freelancers:
            chunks.extend(self._chunk_freelancer_profile(profile))

        # 3. Marketplace policy pack
        policy_path = self.data_dir / "policy_pack" / "marketplace_policy.md"
        if policy_path.exists():
            policy_md = policy_path.read_text(encoding="utf-8")
            sources_loaded.append(str(policy_path))
        else:
            logger.warning("Policy pack %s not found; using built-in fallback policy.", policy_path)
            policy_md = _builtin_fallback_policy()
            sources_loaded.append("builtin://marketplace_policy")

        chunks.extend(self._chunk_policy_markdown(policy_md, source_name=str(policy_path)))

        # 4. Optional GitHub repo markdown docs
        if self.github_repos_dir.exists() and self.github_repos_dir.is_dir():
            for md_file in sorted(self.github_repos_dir.glob("*.md")):
                md_text = md_file.read_text(encoding="utf-8")
                chunks.extend(self._chunk_github_markdown(md_file.stem, md_text))
                sources_loaded.append(str(md_file))

        portfolio_count = sum(1 for c in chunks if c.namespace == "portfolio")
        policy_count = sum(1 for c in chunks if c.namespace == "policy")
        unique_parents = len({c.parent_id for c in chunks})

        return IngestResult(
            chunks=chunks,
            total_chunks=len(chunks),
            portfolio_chunks=portfolio_count,
            policy_chunks=policy_count,
            parent_documents=unique_parents,
            sources_loaded=sources_loaded,
        )

    def _chunk_portfolio_item(self, item: dict[str, Any]) -> list[DocumentChunk]:
        evidence_id = str(
            item.get("evidence_id")
            or item.get("id")
            or _stable_id("ev", json.dumps(item, sort_keys=True))
        )
        freelancer_id = str(
            item.get("freelancer_id") or item.get("persona_id") or "persona_aiml_01"
        )
        raw_type = str(item.get("type") or item.get("item_type") or "case_study").lower()
        if "github" in raw_type or "repo" in raw_type:
            doc_type: DocType = "github_repo"
        elif "proposal" in raw_type:
            doc_type = "past_proposal"
        else:
            doc_type = "case_study"

        title = str(item.get("title") or f"Portfolio Evidence {evidence_id}").strip()
        problem = str(item.get("problem") or item.get("context") or "").strip()
        solution = str(
            item.get("solution")
            or item.get("approach")
            or item.get("summary")
            or item.get("description")
            or ""
        ).strip()
        outcome = str(
            item.get("outcome") or item.get("metrics") or item.get("impact") or ""
        ).strip()
        skills = _normalize_skills(item.get("skills") or item.get("tags"))
        industry = str(item.get("client_industry") or item.get("industry") or "General").strip()
        date_str = str(item.get("date") or item.get("completed_at") or "2026-01-01").strip()
        source_url = str(item.get("source") or item.get("repo_url") or "").strip()
        outcome_status = (
            str(item.get("outcome_status") or item.get("status") or "completed").strip().lower()
        )
        provenance = _normalize_provenance(item.get("data_provenance"))

        skills_str = ", ".join(skills) if skills else "General Software Engineering"
        parent_lines = [
            f"[{evidence_id}] {title} ({doc_type.replace('_', ' ').title()})",
            f"Client Industry: {industry} | Date: {date_str} | Outcome Status: {outcome_status}",
            f"Skills & Stack: {skills_str}",
        ]
        if problem:
            parent_lines.append(f"Problem: {problem}")
        if solution:
            parent_lines.append(f"Solution & Architecture: {solution}")
        if outcome:
            parent_lines.append(f"Measured Outcome & Metrics: {outcome}")
        if source_url:
            parent_lines.append(f"Artifact Reference: {source_url}")
        parent_text = "\n".join(parent_lines)

        parent_id = f"parent_{evidence_id}"
        base_meta = {
            "client_industry": industry,
            "date": date_str,
            "source": source_url,
            "raw_type": raw_type,
        }

        emitted: list[DocumentChunk] = []

        # Aspect 1: Overview & Problem Context
        overview_text = (
            f"{title}. Industry: {industry}. Skills: {skills_str}. "
            f"{problem} {solution[:240]}".strip()
        )
        emitted.append(
            DocumentChunk(
                chunk_id=f"{evidence_id}__overview",
                parent_id=parent_id,
                evidence_id=evidence_id,
                namespace="portfolio",
                doc_type=doc_type,
                title=title,
                section_title="Overview & Problem",
                text=overview_text,
                parent_text=parent_text,
                freelancer_id=freelancer_id,
                skills=skills,
                outcome_status=outcome_status,
                data_provenance=provenance,
                metadata=base_meta,
            )
        )

        # Aspect 2: Technical Implementation & Stack
        if solution:
            impl_text = f"{title} — Technical Implementation ({skills_str}): {solution}"
            emitted.append(
                DocumentChunk(
                    chunk_id=f"{evidence_id}__solution",
                    parent_id=parent_id,
                    evidence_id=evidence_id,
                    namespace="portfolio",
                    doc_type=doc_type,
                    title=title,
                    section_title="Solution & Architecture",
                    text=impl_text,
                    parent_text=parent_text,
                    freelancer_id=freelancer_id,
                    skills=skills,
                    outcome_status=outcome_status,
                    data_provenance=provenance,
                    metadata=base_meta,
                )
            )

        # Aspect 3: Measured Outcome & Quantitative Impact
        if outcome:
            outcome_text = (
                f"{title} — Verified Outcome & Metrics ({industry}, {skills_str}): {outcome}"
            )
            emitted.append(
                DocumentChunk(
                    chunk_id=f"{evidence_id}__outcome",
                    parent_id=parent_id,
                    evidence_id=evidence_id,
                    namespace="portfolio",
                    doc_type=doc_type,
                    title=title,
                    section_title="Measured Outcome",
                    text=outcome_text,
                    parent_text=parent_text,
                    freelancer_id=freelancer_id,
                    skills=skills,
                    outcome_status=outcome_status,
                    data_provenance=provenance,
                    metadata=base_meta,
                )
            )

        return emitted

    def _chunk_freelancer_profile(self, profile: dict[str, Any]) -> list[DocumentChunk]:
        freelancer_id = str(profile.get("freelancer_id") or profile.get("id") or "persona_aiml_01")
        name = str(profile.get("name") or freelancer_id).strip()
        headline = str(profile.get("headline") or profile.get("title") or "").strip()
        bio = str(profile.get("bio") or profile.get("summary") or "").strip()
        years_exp = profile.get("years_experience")
        hourly_rate = profile.get("hourly_rate")
        skills = _normalize_skills(profile.get("skills"))
        provenance = _normalize_provenance(profile.get("data_provenance"))

        skills_str = ", ".join(skills)
        exp_str = f"{years_exp} years of professional experience. " if years_exp is not None else ""
        rate_str = f"Standard rate: ${hourly_rate}/hr. " if hourly_rate is not None else ""

        parent_text = (
            f"Freelancer Profile: {name} ({freelancer_id})\n"
            f"Headline: {headline}\n"
            f"{exp_str}{rate_str}\n"
            f"Verified Core Skills: {skills_str}\n"
            f"Summary: {bio}"
        ).strip()

        parent_id = f"parent_profile_{freelancer_id}"
        return [
            DocumentChunk(
                chunk_id=f"profile_{freelancer_id}__bio",
                parent_id=parent_id,
                evidence_id=f"profile_{freelancer_id}",
                namespace="portfolio",
                doc_type="freelancer_profile",
                title=f"{name} — {headline}",
                section_title="Freelancer Bio & Core Skills",
                text=parent_text,
                parent_text=parent_text,
                freelancer_id=freelancer_id,
                skills=skills,
                outcome_status="verified_profile",
                data_provenance=provenance,
                metadata={
                    "name": name,
                    "years_experience": years_exp,
                    "hourly_rate": hourly_rate,
                },
            )
        ]

    def _chunk_policy_markdown(self, markdown_text: str, source_name: str) -> list[DocumentChunk]:
        """Split marketplace policy markdown by H2/H3 headers while preserving rule IDs."""
        lines = markdown_text.splitlines()
        sections: list[tuple[str, str | None, list[str]]] = []
        current_title = "Marketplace Policy Overview"
        current_rule_id: str | None = None
        current_body: list[str] = []

        rule_pattern = re.compile(r"\b(POL-\d{3})\b", re.IGNORECASE)

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("## ") or stripped.startswith("### "):
                if current_body:
                    body_str = "\n".join(current_body).strip()
                    if body_str:
                        sections.append((current_title, current_rule_id, current_body))
                heading_clean = stripped.lstrip("#").strip()
                match = rule_pattern.search(heading_clean)
                current_rule_id = match.group(1).upper() if match else None
                current_title = heading_clean
                current_body = []
            elif stripped.startswith("# "):
                continue
            else:
                current_body.append(line)

        if current_body:
            body_str = "\n".join(current_body).strip()
            if body_str:
                sections.append((current_title, current_rule_id, current_body))

        chunks: list[DocumentChunk] = []
        for idx, (sec_title, rule_id, body_lines) in enumerate(sections, start=1):
            full_section_text = f"{sec_title}\n" + "\n".join(body_lines).strip()
            effective_rule_id = rule_id or f"POL-SEC-{idx:02d}"
            parent_id = f"parent_policy_{effective_rule_id.lower()}"

            # Split long sections into paragraph-level child chunks while keeping parent_text complete
            paragraphs = [
                p.strip() for p in re.split(r"\n\s*\n", "\n".join(body_lines).strip()) if p.strip()
            ]
            if not paragraphs:
                paragraphs = [full_section_text]

            for p_idx, para in enumerate(paragraphs):
                chunk_text = f"[{effective_rule_id}] {sec_title}: {para}"
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"policy_{effective_rule_id.lower()}__p{p_idx}",
                        parent_id=parent_id,
                        evidence_id=effective_rule_id,
                        namespace="policy",
                        doc_type="policy_rule",
                        title=sec_title,
                        section_title=f"Clause {p_idx + 1}",
                        text=chunk_text,
                        parent_text=full_section_text,
                        freelancer_id=None,
                        skills=[],
                        outcome_status=None,
                        policy_rule_id=effective_rule_id,
                        data_provenance="synthetic",
                        metadata={"source": source_name},
                    )
                )

        return chunks

    def _chunk_github_markdown(self, repo_slug: str, markdown_text: str) -> list[DocumentChunk]:
        """Ingest optional GitHub README markdown files into portfolio chunks."""
        clean_text = markdown_text.strip()
        if not clean_text:
            return []
        evidence_id = f"ev_gh_{repo_slug}"
        parent_id = f"parent_{evidence_id}"
        title = f"GitHub Repository: {repo_slug}"
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", clean_text) if p.strip()]
        chunks: list[DocumentChunk] = []
        for idx, para in enumerate(paragraphs[:8]):
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{evidence_id}__sec{idx}",
                    parent_id=parent_id,
                    evidence_id=evidence_id,
                    namespace="portfolio",
                    doc_type="github_repo",
                    title=title,
                    section_title=f"README Section {idx + 1}",
                    text=f"{title}: {para[:600]}",
                    parent_text=clean_text[:3000],
                    freelancer_id="persona_aiml_01",
                    skills=[],
                    outcome_status="open_source",
                    data_provenance="synthetic",
                    metadata={"repo_slug": repo_slug},
                )
            )
        return chunks
