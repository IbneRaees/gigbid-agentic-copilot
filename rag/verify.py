"""Atomic Claim Splitter & Grounding Verification Engine (Part B3).

Decomposes generated proposal cover letters and screening answers into atomic claims,
distinguishing concrete experience/metric/skill claims from benign conversational phrasing,
and verifies every claim against retrieved `EvidenceChunk`s and `FreelancerProfile` data.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from rag.index.hybrid_store import DEFAULT_RERANKER_MODEL, CrossEncoderBackend, tokenize_technical
from rag.retriever import EvidenceChunk

ClaimType = Literal["EXPERIENCE", "METRIC", "SKILL_TOOL", "DELIVERABLE_ACTION", "GENERIC"]
ClaimStatus = Literal["SUPPORTED", "UNSUPPORTED", "CONTRADICTED", "GENERIC"]


class AtomicClaim(BaseModel):
    """Single atomic statement extracted from a proposal cover letter or screening answer."""

    claim_id: str
    source_field: str = Field(default="cover_letter")
    text: str
    claim_type: ClaimType
    extracted_numbers: list[str] = Field(default_factory=list)
    extracted_years: list[int] = Field(default_factory=list)
    extracted_entities: list[str] = Field(default_factory=list)
    cited_evidence_ids: list[str] = Field(default_factory=list)


class ClaimVerificationResult(BaseModel):
    """Verification outcome for one atomic claim."""

    claim_id: str
    source_field: str
    claim_text: str
    claim_type: ClaimType
    status: ClaimStatus
    confidence: float
    matched_evidence_ids: list[str] = Field(default_factory=list)
    matched_snippets: list[str] = Field(default_factory=list)
    reason: str


class VerificationReport(BaseModel):
    """Full grounding verification report consumed by Proposal Writer, Critic, and UI."""

    claims: list[ClaimVerificationResult]
    grounding_score: float = Field(..., ge=0.0, le=1.0)
    total_claims: int
    verifiable_claims: int
    supported_count: int
    unsupported_count: int
    contradicted_count: int
    generic_count: int
    unsupported_claims: list[str] = Field(default_factory=list)
    passed: bool


# Regex patterns for claim classification & fact extraction
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])|\n+")
_CITATION_TAG_RE = re.compile(r"\b(ev_[a-z0-9_]+|POL-\d{3})\b", re.IGNORECASE)
_NUMBER_RE = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*(?:%|x|k|m|b|ms|s|sec|seconds|mins|minutes|hours|hrs|days|weeks|months|years|yrs|\+)?\b",
    re.IGNORECASE,
)
_YEARS_EXP_RE = re.compile(
    r"\b(\d{1,2})\+?\s*(?:years?|yrs?)\s+(?:of\s+)?(?:experience|building|working|developing|in)?",
    re.IGNORECASE,
)

_EXPERIENCE_VERBS_RE = re.compile(
    r"\b(?:built|architected|designed|deployed|delivered|implemented|reduced|cut|improved|"
    r"increased|scaled|migrated|integrated|engineered|developed|created|maintained|led|"
    r"processed|handled|shipped|achieved|saved|eliminated|automated|trained|fine-tuned)\b",
    re.IGNORECASE,
)

_GENERIC_PREFIXES = (
    "hello",
    "hi ",
    "hi,",
    "dear ",
    "thank you",
    "thanks for",
    "i read your",
    "i reviewed your",
    "i would love",
    "i'd love",
    "looking forward",
    "please let me know",
    "happy to hop on",
    "available to start",
    "best regards",
    "sincerely",
    "kind regards",
)

# Curated technical vocabulary to detect explicit skill/tool claims even without past-tense verbs
_KNOWN_TECH_TERMS = {
    "python",
    "fastapi",
    "langgraph",
    "langchain",
    "postgresql",
    "postgres",
    "pgvector",
    "redis",
    "celery",
    "docker",
    "kubernetes",
    "aws",
    "gcp",
    "azure",
    "nextjs",
    "next.js",
    "typescript",
    "react",
    "playwright",
    "bm25",
    "rag",
    "llm",
    "llms",
    "openai",
    "ollama",
    "qwen",
    "pytorch",
    "tensorflow",
    "graphql",
    "kafka",
    "elasticsearch",
    "qdrant",
    "pinecone",
    "weaviate",
    "chromadb",
    "sqlite",
    "sqlalchemy",
    "pydantic",
    "mcp",
    "oauth",
    "sse",
}


def _normalize_numeric_token(raw: str) -> tuple[float, str] | None:
    """Parse a numeric token like '62%', '50k', '2.8s', '6 years' into (canonical_value, unit_class)."""
    s = raw.strip().lower().rstrip("+")
    m = re.match(r"^(\d+(?:[.,]\d+)?)\s*([a-z%]*)$", s)
    if not m:
        return None
    val = float(m.group(1).replace(",", ""))
    unit = m.group(2)
    if unit == "k":
        return (val * 1_000.0, "count")
    if unit == "m":
        return (val * 1_000_000.0, "count")
    if unit == "b":
        return (val * 1_000_000_000.0, "count")
    if unit == "%":
        return (val, "percent")
    if unit in {"year", "years", "yr", "yrs"}:
        return (val, "years")
    if unit in {"ms"}:
        return (val, "time_ms")
    if unit in {"s", "sec", "seconds"}:
        return (val, "time_s")
    return (val, unit or "count")


class ClaimSplitter:
    """Decomposes proposal text and screening answers into typed atomic claims."""

    def split(
        self,
        cover_letter: str,
        screening_answers: dict[str, str] | list[str] | None = None,
    ) -> list[AtomicClaim]:
        segments: list[tuple[str, str]] = []

        for sent in _SENTENCE_SPLIT_RE.split(cover_letter.strip()):
            clean = sent.strip()
            if clean:
                segments.append(("cover_letter", clean))

        if isinstance(screening_answers, dict):
            for q_key, ans_val in screening_answers.items():
                for sent in _SENTENCE_SPLIT_RE.split(str(ans_val).strip()):
                    clean = sent.strip()
                    if clean:
                        segments.append((f"screening:{q_key}", clean))
        elif isinstance(screening_answers, list):
            for idx, ans_val in enumerate(screening_answers):
                for sent in _SENTENCE_SPLIT_RE.split(str(ans_val).strip()):
                    clean = sent.strip()
                    if clean:
                        segments.append((f"screening_{idx}", clean))

        claims: list[AtomicClaim] = []
        for idx, (source_field, sentence) in enumerate(segments, start=1):
            claims.append(self._classify_sentence(f"claim_{idx:02d}", source_field, sentence))
        return claims

    def _classify_sentence(self, claim_id: str, source_field: str, sentence: str) -> AtomicClaim:
        lower = sentence.lower().strip()
        cited_ids = [m.group(1) for m in _CITATION_TAG_RE.finditer(sentence)]

        raw_numbers = [
            m.group(0).strip()
            for m in _NUMBER_RE.finditer(sentence)
            if not re.match(r"^\d{4}$", m.group(0).strip())  # Ignore calendar years like 2026
        ]
        years_found = [int(m.group(1)) for m in _YEARS_EXP_RE.finditer(sentence)]

        tokens = tokenize_technical(sentence)
        entities = sorted({t for t in tokens if t in _KNOWN_TECH_TERMS})

        has_exp_verb = bool(_EXPERIENCE_VERBS_RE.search(sentence))
        is_first_person_past = bool(
            re.search(r"\b(?:i|we|my|our)\b", lower) and (has_exp_verb or years_found or cited_ids)
        )
        is_forward_plan = bool(
            re.match(
                r"^(?:for your|to solve|my plan|step \d|phase \d|i propose|we can|i can|i will)\b",
                lower,
            )
        )

        # Determine claim category
        if any(lower.startswith(prefix) for prefix in _GENERIC_PREFIXES) and not (
            raw_numbers or cited_ids or has_exp_verb
        ):
            claim_type: ClaimType = "GENERIC"
        elif raw_numbers and (has_exp_verb or years_found or cited_ids or is_first_person_past):
            claim_type = "METRIC"
        elif is_first_person_past or cited_ids:
            claim_type = "EXPERIENCE"
        elif entities and not is_forward_plan:
            claim_type = "SKILL_TOOL"
        elif is_forward_plan and entities:
            # Forward-looking technical proposal mentioning specific stack tools implies skill ownership
            claim_type = "SKILL_TOOL"
        elif is_forward_plan:
            claim_type = "DELIVERABLE_ACTION"
        else:
            claim_type = "GENERIC"

        return AtomicClaim(
            claim_id=claim_id,
            source_field=source_field,
            text=sentence,
            claim_type=claim_type,
            extracted_numbers=raw_numbers,
            extracted_years=years_found,
            extracted_entities=entities,
            cited_evidence_ids=cited_ids,
        )


class ClaimVerifier:
    """Verifies atomic claims against retrieved portfolio evidence and freelancer profile."""

    def __init__(
        self,
        reranker_model: str = DEFAULT_RERANKER_MODEL,
        entailment_threshold: float = 0.42,
        min_grounding_score: float = 0.85,
    ) -> None:
        self.splitter = ClaimSplitter()
        self.reranker = CrossEncoderBackend.get_shared(reranker_model)
        self.entailment_threshold = entailment_threshold
        self.min_grounding_score = min_grounding_score

    def verify(
        self,
        cover_letter: str,
        evidence_chunks: list[EvidenceChunk],
        freelancer_profile: dict[str, Any] | None = None,
        screening_answers: dict[str, str] | list[str] | None = None,
    ) -> VerificationReport:
        """Split and verify all claims in a proposal against retrieved evidence."""
        claims = self.splitter.split(cover_letter=cover_letter, screening_answers=screening_answers)

        # Build unified evidence corpus: retrieved chunks + optional freelancer profile summary
        evidence_corpus: list[tuple[str, str, set[str]]] = []
        known_skills: set[str] = set()
        profile_years: int | None = None

        for chunk in evidence_chunks:
            ev_id = chunk.evidence_id or chunk.chunk_id
            full_text = chunk.parent_content or chunk.content
            ev_tokens = set(tokenize_technical(full_text))
            for sk in chunk.skills:
                known_skills.add(sk.lower())
                ev_tokens.add(sk.lower())
            evidence_corpus.append((ev_id, full_text, ev_tokens))

        if freelancer_profile:
            prof_id = str(freelancer_profile.get("freelancer_id") or "profile")
            prof_skills = [str(s).strip() for s in freelancer_profile.get("skills", [])]
            for sk in prof_skills:
                known_skills.add(sk.lower())
            if freelancer_profile.get("years_experience") is not None:
                profile_years = int(freelancer_profile["years_experience"])
            prof_text = (
                f"Freelancer {freelancer_profile.get('name', prof_id)}: "
                f"{freelancer_profile.get('headline', '')}. "
                f"{freelancer_profile.get('bio', '')}. "
                f"Years experience: {profile_years}. "
                f"Skills: {', '.join(prof_skills)}."
            )
            evidence_corpus.append(
                (f"profile_{prof_id}", prof_text, set(tokenize_technical(prof_text)) | known_skills)
            )

        results: list[ClaimVerificationResult] = []
        for claim in claims:
            results.append(
                self._verify_single_claim(
                    claim=claim,
                    evidence_corpus=evidence_corpus,
                    known_skills=known_skills,
                    profile_years=profile_years,
                )
            )

        supported_count = sum(1 for r in results if r.status == "SUPPORTED")
        unsupported_count = sum(1 for r in results if r.status == "UNSUPPORTED")
        contradicted_count = sum(1 for r in results if r.status == "CONTRADICTED")
        generic_count = sum(1 for r in results if r.status == "GENERIC")
        verifiable_claims = supported_count + unsupported_count + contradicted_count

        if verifiable_claims == 0:
            grounding_score = 1.0
        else:
            grounding_score = round(supported_count / verifiable_claims, 4)

        unsupported_feedback = [
            f'[{r.claim_id}] ({r.status}) "{r.claim_text}" -> {r.reason}'
            for r in results
            if r.status in {"UNSUPPORTED", "CONTRADICTED"}
        ]

        passed = (contradicted_count == 0) and (grounding_score >= self.min_grounding_score)

        return VerificationReport(
            claims=results,
            grounding_score=grounding_score,
            total_claims=len(results),
            verifiable_claims=verifiable_claims,
            supported_count=supported_count,
            unsupported_count=unsupported_count,
            contradicted_count=contradicted_count,
            generic_count=generic_count,
            unsupported_claims=unsupported_feedback,
            passed=passed,
        )

    def _verify_single_claim(
        self,
        claim: AtomicClaim,
        evidence_corpus: list[tuple[str, str, set[str]]],
        known_skills: set[str],
        profile_years: int | None,
    ) -> ClaimVerificationResult:
        if claim.claim_type == "GENERIC":
            return ClaimVerificationResult(
                claim_id=claim.claim_id,
                source_field=claim.source_field,
                claim_text=claim.text,
                claim_type=claim.claim_type,
                status="GENERIC",
                confidence=1.0,
                matched_evidence_ids=[],
                matched_snippets=[],
                reason="Conversational or transitional phrasing requiring no external citation.",
            )

        if not evidence_corpus:
            return ClaimVerificationResult(
                claim_id=claim.claim_id,
                source_field=claim.source_field,
                claim_text=claim.text,
                claim_type=claim.claim_type,
                status="UNSUPPORTED",
                confidence=0.95,
                matched_evidence_ids=[],
                matched_snippets=[],
                reason="No retrieved portfolio evidence available to verify factual claim.",
            )

        # 1. Check explicit years-of-experience contradiction against FreelancerProfile
        if claim.extracted_years and profile_years is not None:
            for claimed_yrs in claim.extracted_years:
                if claimed_yrs > profile_years + 1:
                    return ClaimVerificationResult(
                        claim_id=claim.claim_id,
                        source_field=claim.source_field,
                        claim_text=claim.text,
                        claim_type=claim.claim_type,
                        status="CONTRADICTED",
                        confidence=0.96,
                        matched_evidence_ids=[],
                        matched_snippets=[],
                        reason=(
                            f"Claimed {claimed_yrs} years experience exceeds verified profile "
                            f"experience ({profile_years} years)."
                        ),
                    )

        # 2. Check if cited evidence IDs actually exist in retrieved evidence
        available_ev_ids = {ev_id.lower() for ev_id, _, _ in evidence_corpus}
        for cited in claim.cited_evidence_ids:
            if cited.lower() not in available_ev_ids:
                return ClaimVerificationResult(
                    claim_id=claim.claim_id,
                    source_field=claim.source_field,
                    claim_text=claim.text,
                    claim_type=claim.claim_type,
                    status="UNSUPPORTED",
                    confidence=0.94,
                    matched_evidence_ids=[],
                    matched_snippets=[],
                    reason=f"Cited evidence ID '{cited}' was not found in retrieved portfolio items.",
                )

        # 3. Check unowned technical entities/skills
        if claim.extracted_entities and known_skills:
            missing_entities = [
                ent
                for ent in claim.extracted_entities
                if ent not in known_skills
                and not any(ent in ev_toks for _, _, ev_toks in evidence_corpus)
            ]
            if missing_entities and claim.claim_type in {"SKILL_TOOL", "EXPERIENCE"}:
                return ClaimVerificationResult(
                    claim_id=claim.claim_id,
                    source_field=claim.source_field,
                    claim_text=claim.text,
                    claim_type=claim.claim_type,
                    status="UNSUPPORTED",
                    confidence=0.88,
                    matched_evidence_ids=[],
                    matched_snippets=[],
                    reason=(
                        f"Claimed technology/skill(s) {missing_entities} not present in verified "
                        "portfolio evidence or freelancer profile."
                    ),
                )

        # 4. Score semantic entailment against every evidence item using cross-encoder + lexical overlap
        pairs = [(claim.text, ev_text) for _, ev_text, _ in evidence_corpus]
        ce_scores = self.reranker.predict(pairs)

        claim_tokens = {
            t
            for t in tokenize_technical(claim.text)
            if len(t) >= 3
            and t not in {"the", "and", "for", "with", "that", "from", "have", "built"}
        }

        best_score = 0.0
        matched_ids: list[str] = []
        matched_snippets: list[str] = []

        for (ev_id, ev_text, ev_tokens), ce_score in zip(evidence_corpus, ce_scores):
            lex_overlap = (
                len(claim_tokens & ev_tokens) / max(1, len(claim_tokens)) if claim_tokens else 0.0
            )
            combined = 0.60 * ce_score + 0.40 * lex_overlap
            if combined > best_score:
                best_score = combined
            if combined >= self.entailment_threshold or ev_id.lower() in {
                c.lower() for c in claim.cited_evidence_ids
            }:
                if ev_id not in matched_ids:
                    matched_ids.append(ev_id)
                    matched_snippets.append(ev_text[:220].replace("\n", " "))

        # 5. Strict quantitative metric verification for METRIC claims
        if claim.extracted_numbers:
            all_evidence_text = "\n".join(ev_text for _, ev_text, _ in evidence_corpus)
            ev_parsed_nums = [
                parsed
                for m in _NUMBER_RE.finditer(all_evidence_text)
                if (parsed := _normalize_numeric_token(m.group(0))) is not None
            ]

            unverified_nums: list[str] = []
            for raw_num in claim.extracted_numbers:
                parsed_claim = _normalize_numeric_token(raw_num)
                if parsed_claim is None:
                    continue
                c_val, c_unit = parsed_claim
                # Check if an equivalent number exists in evidence (within 5% tolerance)
                found_match = any(
                    (c_unit == e_unit or {c_unit, e_unit} <= {"count", "years"})
                    and abs(c_val - e_val) <= max(0.05 * max(abs(c_val), abs(e_val)), 1e-6)
                    for e_val, e_unit in ev_parsed_nums
                )
                if not found_match:
                    unverified_nums.append(raw_num)

            if unverified_nums:
                return ClaimVerificationResult(
                    claim_id=claim.claim_id,
                    source_field=claim.source_field,
                    claim_text=claim.text,
                    claim_type=claim.claim_type,
                    status="UNSUPPORTED",
                    confidence=0.92,
                    matched_evidence_ids=matched_ids[:3],
                    matched_snippets=matched_snippets[:2],
                    reason=(
                        f"Quantitative metric(s) {unverified_nums} could not be verified against "
                        "retrieved portfolio metrics."
                    ),
                )

        # 6. Deliverable / architectural alignment check
        if claim.claim_type == "DELIVERABLE_ACTION":
            if matched_ids or best_score >= (self.entailment_threshold * 0.85):
                return ClaimVerificationResult(
                    claim_id=claim.claim_id,
                    source_field=claim.source_field,
                    claim_text=claim.text,
                    claim_type=claim.claim_type,
                    status="SUPPORTED",
                    confidence=round(min(0.99, max(0.65, best_score)), 4),
                    matched_evidence_ids=matched_ids[:3],
                    matched_snippets=matched_snippets[:2],
                    reason="Proposed implementation approach aligns with verified portfolio stack.",
                )

        if matched_ids and best_score >= self.entailment_threshold:
            return ClaimVerificationResult(
                claim_id=claim.claim_id,
                source_field=claim.source_field,
                claim_text=claim.text,
                claim_type=claim.claim_type,
                status="SUPPORTED",
                confidence=round(min(0.99, best_score), 4),
                matched_evidence_ids=matched_ids[:3],
                matched_snippets=matched_snippets[:2],
                reason=f"Grounded in retrieved portfolio evidence ({', '.join(matched_ids[:3])}).",
            )

        # Structured skill-metadata check for SKILL_TOOL claims
        if claim.claim_type == "SKILL_TOOL" and claim.extracted_entities:
            all_verified = all(
                ent in known_skills or any(ent in ev_toks for _, _, ev_toks in evidence_corpus)
                for ent in claim.extracted_entities
            )
            if all_verified:
                skill_matched_ids = [
                    ev_id
                    for ev_id, _, ev_toks in evidence_corpus
                    if ev_id != "freelancer_profile"
                    and any(ent in ev_toks for ent in claim.extracted_entities)
                ]
                skill_snippets = [
                    ev_text[:220].replace("\n", " ")
                    for ev_id, ev_text, _ in evidence_corpus
                    if ev_id in skill_matched_ids
                ]
                return ClaimVerificationResult(
                    claim_id=claim.claim_id,
                    source_field=claim.source_field,
                    claim_text=claim.text,
                    claim_type=claim.claim_type,
                    status="SUPPORTED",
                    confidence=0.85,
                    matched_evidence_ids=skill_matched_ids[:3],
                    matched_snippets=skill_snippets[:2],
                    reason=(
                        f"Skill/tool claim ({', '.join(claim.extracted_entities)}) grounded in "
                        "verified freelancer profile and portfolio metadata."
                    ),
                )

        return ClaimVerificationResult(
            claim_id=claim.claim_id,
            source_field=claim.source_field,
            claim_text=claim.text,
            claim_type=claim.claim_type,
            status="UNSUPPORTED",
            confidence=round(max(0.55, 1.0 - best_score), 4),
            matched_evidence_ids=[],
            matched_snippets=[],
            reason="Insufficient semantic or factual overlap with retrieved portfolio evidence.",
        )

    def sanitize_proposal(self, cover_letter: str, report: VerificationReport) -> str:
        """Remove UNSUPPORTED or CONTRADICTED sentences from a draft before Critic review."""
        bad_texts = {
            r.claim_text.strip()
            for r in report.claims
            if r.source_field == "cover_letter" and r.status in {"UNSUPPORTED", "CONTRADICTED"}
        }
        if not bad_texts:
            return cover_letter

        kept_sentences: list[str] = []
        for sent in _SENTENCE_SPLIT_RE.split(cover_letter.strip()):
            clean = sent.strip()
            if clean and clean not in bad_texts:
                kept_sentences.append(clean)
        return " ".join(kept_sentences)
