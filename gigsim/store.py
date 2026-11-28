"""SQLite Persistence, Idempotency Engine, Policy Checker & Client Simulator for GigSim."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import HTTPException, status

from gigsim.models import (
    BudgetInfo,
    ClientRecord,
    ConnectsBalanceResponse,
    ConnectsLedgerEntry,
    DryRunPreview,
    JobListResponse,
    JobRecord,
    MessageItem,
    ProposalDraftRequest,
    ProposalResponse,
    ProposalSubmitRequest,
    SendMessageRequest,
    SendMessageResponse,
    ThreadResponse,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = Path(os.getenv("GIGSIM_DB_PATH", str(ROOT_DIR / "gigsim" / "gigsim.db")))
DEFAULT_DATA_DIR = Path(os.getenv("GIGSIM_DATA_DIR", str(ROOT_DIR / "data")))

# Policy regex patterns aligned with data/policy_pack/marketplace_policy.md
EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
PHONE_PATTERN = re.compile(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}")
OFF_PLATFORM_TERMS = (
    "whatsapp",
    "telegram",
    "discord",
    "skype",
    "signal app",
    "pay via paypal",
    "pay via crypto",
    "usdt",
    "outside upwork",
    "outside the platform",
    "off-platform",
    "work for free",
    "free sample project",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _compute_marketplace_fee(amount: float) -> float:
    """Compute tiered service fee matching POL-007 (15% <= $500, 10% <= $10k, 5% > $10k)."""
    if amount <= 500.0:
        return round(amount * 0.15, 2)
    if amount <= 10000.0:
        return round(500.0 * 0.15 + (amount - 500.0) * 0.10, 2)
    return round(500.0 * 0.15 + 9500.0 * 0.10 + (amount - 10000.0) * 0.05, 2)


class GigSimStore:
    """SQLite-backed state manager for GigSim marketplace simulation."""

    def __init__(
        self,
        db_path: Path | None = None, data_dir: Path = DEFAULT_DATA_DIR) -> None:
        self.db_path = Path(db_path or os.getenv("GIGSIM_DB_PATH", str(ROOT_DIR / "gigsim" / "gigsim.db")))
        self.data_dir = Path(data_dir)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()
        self.seed_if_empty()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS clients (
                    client_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    country TEXT NOT NULL,
                    payment_verified INTEGER NOT NULL,
                    rating REAL NOT NULL,
                    reviews_count INTEGER NOT NULL,
                    jobs_posted INTEGER NOT NULL,
                    hire_rate REAL NOT NULL,
                    total_spent_usd REAL NOT NULL,
                    avg_hourly_paid REAL NOT NULL,
                    member_since TEXT NOT NULL,
                    past_reviews_json TEXT NOT NULL,
                    data_provenance TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    client_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    required_skills_json TEXT NOT NULL,
                    nice_to_have_skills_json TEXT NOT NULL,
                    budget_min REAL NOT NULL,
                    budget_max REAL NOT NULL,
                    currency TEXT NOT NULL,
                    budget_type TEXT NOT NULL,
                    duration_days INTEGER NOT NULL,
                    experience_level TEXT NOT NULL,
                    screening_questions_json TEXT NOT NULL,
                    connects_cost INTEGER NOT NULL,
                    posted_at TEXT NOT NULL,
                    domain_category TEXT NOT NULL,
                    data_provenance TEXT NOT NULL,
                    FOREIGN KEY (client_id) REFERENCES clients(client_id)
                );

                CREATE TABLE IF NOT EXISTS accounts (
                    freelancer_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    connects_balance INTEGER NOT NULL,
                    weekly_budget INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS connects_ledger (
                    entry_id TEXT PRIMARY KEY,
                    freelancer_id TEXT NOT NULL,
                    delta INTEGER NOT NULL,
                    balance_after INTEGER NOT NULL,
                    reason TEXT NOT NULL,
                    job_id TEXT,
                    proposal_id TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS proposals (
                    proposal_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    freelancer_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    cover_letter TEXT NOT NULL,
                    screening_answers_json TEXT NOT NULL,
                    proposed_rate REAL NOT NULL,
                    currency TEXT NOT NULL,
                    cited_evidence_ids_json TEXT NOT NULL,
                    connects_charged INTEGER NOT NULL,
                    thread_id TEXT,
                    simulated_outcome TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS threads (
                    thread_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    client_id TEXT NOT NULL,
                    freelancer_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    message_id TEXT PRIMARY KEY,
                    thread_id TEXT NOT NULL,
                    sender TEXT NOT NULL,
                    text TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (thread_id) REFERENCES threads(thread_id)
                );

                CREATE TABLE IF NOT EXISTS idempotency_records (
                    idempotency_key TEXT PRIMARY KEY,
                    endpoint TEXT NOT NULL,
                    request_hash TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    def seed_if_empty(self) -> None:
        """Populate SQLite from `data/*.json` if the jobs table is empty."""
        with self._connect() as conn:
            count = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
            if count > 0:
                return

            clients_file = self.data_dir / "client_history.json"
            jobs_file = self.data_dir / "jobs.json"
            freelancers_file = self.data_dir / "freelancers.json"
            messages_file = self.data_dir / "messages.json"

            # 1. Seed clients
            clients_data: list[dict[str, Any]] = []
            if clients_file.exists():
                clients_data = json.loads(clients_file.read_text(encoding="utf-8"))
            else:
                clients_data = [
                    {
                        "client_id": "cli_fallback_01",
                        "name": "Fallback Tech Client",
                        "country": "United States",
                        "payment_verified": True,
                        "rating": 4.9,
                        "reviews_count": 10,
                        "jobs_posted": 12,
                        "hire_rate": 0.83,
                        "total_spent_usd": 25000.0,
                        "avg_hourly_paid": 80.0,
                        "member_since": "2023-01-01",
                        "past_reviews": ["Great client."],
                        "data_provenance": "synthetic",
                    }
                ]

            for cli in clients_data:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO clients (
                        client_id, name, country, payment_verified, rating, reviews_count,
                        jobs_posted, hire_rate, total_spent_usd, avg_hourly_paid,
                        member_since, past_reviews_json, data_provenance
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        cli["client_id"],
                        cli["name"],
                        cli.get("country", "United States"),
                        1 if cli.get("payment_verified", True) else 0,
                        float(cli.get("rating", 4.5)),
                        int(cli.get("reviews_count", 0)),
                        int(cli.get("jobs_posted", 1)),
                        float(cli.get("hire_rate", 0.5)),
                        float(cli.get("total_spent_usd", 0.0)),
                        float(cli.get("avg_hourly_paid", 0.0)),
                        cli.get("member_since", "2024-01-01"),
                        json.dumps(cli.get("past_reviews", [])),
                        cli.get("data_provenance", "synthetic"),
                    ),
                )

            # 2. Seed jobs
            jobs_data: list[dict[str, Any]] = []
            if jobs_file.exists():
                jobs_data = json.loads(jobs_file.read_text(encoding="utf-8"))
            else:
                jobs_data = [
                    {
                        "job_id": "job_fallback_01",
                        "client_id": "cli_fallback_01",
                        "title": "Build FastAPI + LangGraph Agent Pipeline",
                        "description": "Need a Python engineer for LangGraph orchestration and RAG.",
                        "required_skills": ["Python", "LangGraph", "FastAPI"],
                        "nice_to_have_skills": ["Docker"],
                        "budget": {"min": 1500.0, "max": 2500.0, "currency": "USD", "budget_type": "fixed"},
                        "duration_days": 14,
                        "experience_level": "expert",
                        "screening_questions": ["Share a similar system you built."],
                        "connects_cost": 8,
                        "posted_at": _utc_now(),
                        "domain_category": "ai_ml",
                        "data_provenance": "synthetic",
                    }
                ]

            for job in jobs_data:
                budget = job.get("budget", {})
                conn.execute(
                    """
                    INSERT OR IGNORE INTO jobs (
                        job_id, client_id, title, description, required_skills_json,
                        nice_to_have_skills_json, budget_min, budget_max, currency,
                        budget_type, duration_days, experience_level,
                        screening_questions_json, connects_cost, posted_at,
                        domain_category, data_provenance
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        job["job_id"],
                        job["client_id"],
                        job["title"],
                        job["description"],
                        json.dumps(job.get("required_skills", [])),
                        json.dumps(job.get("nice_to_have_skills", [])),
                        float(budget.get("min", 500.0)),
                        float(budget.get("max", 1000.0)),
                        budget.get("currency", "USD"),
                        budget.get("budget_type", "fixed"),
                        int(job.get("duration_days", 14)),
                        job.get("experience_level", "intermediate"),
                        json.dumps(job.get("screening_questions", [])),
                        int(job.get("connects_cost", 6)),
                        job.get("posted_at", _utc_now()),
                        job.get("domain_category", "general"),
                        job.get("data_provenance", "synthetic"),
                    ),
                )

            # 3. Seed freelancer accounts
            freelancers_data: list[dict[str, Any]] = []
            if freelancers_file.exists():
                freelancers_data = json.loads(freelancers_file.read_text(encoding="utf-8"))
            else:
                freelancers_data = [
                    {
                        "freelancer_id": "persona_aiml_01",
                        "name": "Dr. Ayesha Tariq",
                        "connects_balance": 84,
                        "weekly_connects_budget": 60,
                    }
                ]

            for fl in freelancers_data:
                fid = fl["freelancer_id"]
                bal = int(fl.get("connects_balance", 80))
                w_bud = int(fl.get("weekly_connects_budget", 60))
                conn.execute(
                    "INSERT OR IGNORE INTO accounts (freelancer_id, name, connects_balance, weekly_budget) VALUES (?, ?, ?, ?)",
                    (fid, fl.get("name", fid), bal, w_bud),
                )
                conn.execute(
                    """
                    INSERT OR IGNORE INTO connects_ledger (
                        entry_id, freelancer_id, delta, balance_after, reason, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (f"led_init_{fid}", fid, bal, bal, "Initial seeded Connects balance", _utc_now()),
                )

            # 4. Seed conversation threads & messages
            if messages_file.exists():
                threads_data: list[dict[str, Any]] = json.loads(messages_file.read_text(encoding="utf-8"))
                for th in threads_data:
                    tid = th["thread_id"]
                    msgs = th.get("messages", [])
                    last_ts = msgs[-1]["timestamp"] if msgs else _utc_now()
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO threads (
                            thread_id, job_id, client_id, freelancer_id, status, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            tid,
                            th["job_id"],
                            th["client_id"],
                            th.get("freelancer_id", "persona_aiml_01"),
                            th.get("status", "replied"),
                            last_ts,
                        ),
                    )
                    for m in msgs:
                        conn.execute(
                            """
                            INSERT OR IGNORE INTO messages (
                                message_id, thread_id, sender, text, timestamp
                            ) VALUES (?, ?, ?, ?, ?)
                            """,
                            (
                                m["message_id"],
                                tid,
                                m["sender"],
                                m["text"],
                                m["timestamp"],
                            ),
                        )

    def get_counts(self) -> tuple[int, int, int]:
        with self._connect() as conn:
            j = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
            c = conn.execute("SELECT COUNT(*) FROM clients").fetchone()[0]
            t = conn.execute("SELECT COUNT(*) FROM threads").fetchone()[0]
            return int(j), int(c), int(t)

    def _row_to_job(self, row: sqlite3.Row) -> JobRecord:
        return JobRecord(
            job_id=row["job_id"],
            client_id=row["client_id"],
            title=row["title"],
            description=row["description"],
            required_skills=json.loads(row["required_skills_json"]),
            nice_to_have_skills=json.loads(row["nice_to_have_skills_json"]),
            budget=BudgetInfo(
                min=row["budget_min"],
                max=row["budget_max"],
                currency=row["currency"],
                budget_type=row["budget_type"],
            ),
            duration_days=row["duration_days"],
            experience_level=row["experience_level"],
            screening_questions=json.loads(row["screening_questions_json"]),
            connects_cost=row["connects_cost"],
            posted_at=row["posted_at"],
            domain_category=row["domain_category"],
            data_provenance=row["data_provenance"],
        )

    def list_jobs(
        self,
        q: str | None = None,
        skill: str | None = None,
        min_budget: float | None = None,
        max_budget: float | None = None,
        domain: str | None = None,
        limit: int = 25,
        offset: int = 0,
    ) -> JobListResponse:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM jobs ORDER BY posted_at DESC, job_id ASC").fetchall()

        all_jobs = [self._row_to_job(r) for r in rows]
        filtered: list[JobRecord] = []

        q_low = q.lower().strip() if q else None
        skill_low = skill.lower().strip() if skill else None
        domain_low = domain.lower().strip() if domain else None

        for job in all_jobs:
            if q_low and (q_low not in job.title.lower() and q_low not in job.description.lower()):
                continue
            if skill_low and not any(skill_low in s.lower() for s in job.required_skills + job.nice_to_have_skills):
                continue
            if min_budget is not None and job.budget.max < min_budget:
                continue
            if max_budget is not None and job.budget.min > max_budget:
                continue
            if domain_low and job.domain_category.lower() != domain_low:
                continue
            filtered.append(job)

        total = len(filtered)
        sliced = filtered[offset : offset + limit]
        return JobListResponse(total=total, limit=limit, offset=offset, jobs=sliced)

    def get_job(self, job_id: str) -> JobRecord:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job '{job_id}' not found")
        return self._row_to_job(row)

    def get_client(self, client_id: str) -> ClientRecord:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM clients WHERE client_id = ?", (client_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Client '{client_id}' not found")
        return ClientRecord(
            client_id=row["client_id"],
            name=row["name"],
            country=row["country"],
            payment_verified=bool(row["payment_verified"]),
            rating=row["rating"],
            reviews_count=row["reviews_count"],
            jobs_posted=row["jobs_posted"],
            hire_rate=row["hire_rate"],
            total_spent_usd=row["total_spent_usd"],
            avg_hourly_paid=row["avg_hourly_paid"],
            member_since=row["member_since"],
            past_reviews=json.loads(row["past_reviews_json"]),
            data_provenance=row["data_provenance"],
        )

    def _ensure_account(self, conn: sqlite3.Connection, freelancer_id: str) -> sqlite3.Row:
        row = conn.execute("SELECT * FROM accounts WHERE freelancer_id = ?", (freelancer_id,)).fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO accounts (freelancer_id, name, connects_balance, weekly_budget) VALUES (?, ?, ?, ?)",
                (freelancer_id, freelancer_id, 60, 60),
            )
            row = conn.execute("SELECT * FROM accounts WHERE freelancer_id = ?", (freelancer_id,)).fetchone()
        return row

    def get_connects_balance(self, freelancer_id: str = "persona_aiml_01") -> ConnectsBalanceResponse:
        with self._connect() as conn:
            acc = self._ensure_account(conn, freelancer_id)
            ledger_rows = conn.execute(
                """
                SELECT * FROM connects_ledger
                WHERE freelancer_id = ?
                ORDER BY rowid DESC LIMIT 15
                """,
                (freelancer_id,),
            ).fetchall()

        ledger = [
            ConnectsLedgerEntry(
                entry_id=r["entry_id"],
                freelancer_id=r["freelancer_id"],
                delta=r["delta"],
                balance_after=r["balance_after"],
                reason=r["reason"],
                job_id=r["job_id"],
                proposal_id=r["proposal_id"],
                created_at=r["created_at"],
            )
            for r in ledger_rows
        ]
        return ConnectsBalanceResponse(
            freelancer_id=acc["freelancer_id"],
            balance=int(acc["connects_balance"]),
            weekly_budget=int(acc["weekly_budget"]),
            recent_ledger=ledger,
        )

    def _check_idempotency(
        self, conn: sqlite3.Connection, idempotency_key: str, endpoint: str, payload: dict[str, Any]
    ) -> str | None:
        req_hash = _stable_hash(payload)
        existing = conn.execute(
            "SELECT * FROM idempotency_records WHERE idempotency_key = ?", (idempotency_key,)
        ).fetchone()
        if existing is None:
            return None
        if existing["endpoint"] != endpoint or existing["request_hash"] != req_hash:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Idempotency key '{idempotency_key}' was already used with a different request payload.",
            )
        return str(existing["response_json"])

    def _save_idempotency(
        self, conn: sqlite3.Connection, idempotency_key: str, endpoint: str, payload: dict[str, Any], response_obj: Any
    ) -> None:
        req_hash = _stable_hash(payload)
        conn.execute(
            """
            INSERT INTO idempotency_records (idempotency_key, endpoint, request_hash, response_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (idempotency_key, endpoint, req_hash, response_obj.model_dump_json(), _utc_now()),
        )

    def _evaluate_outbound_policy(self, text: str) -> list[str]:
        """Enforce marketplace anti-circumvention & safety policies on outbound text."""
        violations: list[str] = []
        lowered = text.lower()
        if EMAIL_PATTERN.search(text):
            violations.append("POL-001: Off-platform email address detected in outbound text.")
        if PHONE_PATTERN.search(text):
            violations.append("POL-001: Off-platform phone number detected in outbound text.")
        for term in OFF_PLATFORM_TERMS:
            if term in lowered:
                violations.append(f"POL-001/002/003: Prohibited marketplace phrase '{term}' detected.")
        return violations

    def _simulate_client_reaction(
        self, job: JobRecord, client: ClientRecord, req: ProposalSubmitRequest
    ) -> tuple[str, str | None]:
        """Deterministic simulator for client response/interview based on proposal quality."""
        text_low = req.cover_letter.lower()
        job_desc_low = f"{job.title} {job.description}".lower()

        # Scam / non-hiring clients never convert
        if not client.payment_verified and client.hire_rate < 0.15:
            return "submitted", None
        if any(k in job_desc_low for k in ("telegram", "whatsapp", "free sample project")):
            return "submitted", None

        score = 0.0
        # 1. Grounded evidence citations
        if len(req.cited_evidence_ids) >= 2:
            score += 0.35
        elif len(req.cited_evidence_ids) == 1:
            score += 0.20

        # 2. Concrete quantitative metrics in text
        if any(tok in text_low for tok in ("%", "ms", "recall", "latency", "uptime", "p95", "milestone")):
            score += 0.25

        # 3. Skill alignment
        matched_skills = sum(1 for sk in job.required_skills if sk.lower() in text_low)
        if job.required_skills:
            score += 0.25 * min(1.0, matched_skills / max(1, min(3, len(job.required_skills))))

        # 4. Penalize generic openings
        if any(bad in text_low[:120] for bad in ("i am very passionate", "dear sir/madam", "i read your job post")):
            score -= 0.25

        # 5. Budget alignment
        if req.proposed_rate <= job.budget.max * 1.15:
            score += 0.15
        else:
            score -= 0.20

        if score >= 0.72:
            reply = (
                f"Hi — thanks for sharing those concrete metrics on {job.required_skills[0] if job.required_skills else 'this stack'}. "
                f"Your background aligns well with our roadmap. Are you available for a 20-minute technical interview "
                f"tomorrow or Thursday to walk through Milestone 1?"
            )
            return "interview", reply
        if score >= 0.50:
            reply = (
                f"Thanks for applying! Your proposal stood out. Could you clarify your proposed timeline and "
                f"deliverables for the first week at ${req.proposed_rate:,.0f}?"
            )
            return "replied", reply
        if score >= 0.30:
            return "viewed", None
        return "submitted", None

    def create_proposal_draft(self, req: ProposalDraftRequest) -> ProposalResponse:
        payload = req.model_dump()
        job = self.get_job(req.job_id)

        with self._connect() as conn:
            cached = self._check_idempotency(conn, req.idempotency_key, "POST /proposals/draft", payload)
            if cached is not None:
                return ProposalResponse.model_validate_json(cached)

            acc = self._ensure_account(conn, req.freelancer_id)
            now = _utc_now()
            prop_id = f"prop_draft_{hashlib.sha256(req.idempotency_key.encode()).hexdigest()[:10]}"

            conn.execute(
                """
                INSERT INTO proposals (
                    proposal_id, job_id, freelancer_id, status, cover_letter,
                    screening_answers_json, proposed_rate, currency,
                    cited_evidence_ids_json, connects_charged, thread_id,
                    simulated_outcome, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    prop_id,
                    job.job_id,
                    req.freelancer_id,
                    "draft",
                    req.cover_letter,
                    json.dumps(req.screening_answers),
                    req.proposed_rate,
                    req.currency,
                    json.dumps(req.cited_evidence_ids),
                    0,
                    None,
                    "draft_saved",
                    now,
                    now,
                ),
            )

            resp = ProposalResponse(
                proposal_id=prop_id,
                job_id=job.job_id,
                freelancer_id=req.freelancer_id,
                status="draft",
                proposed_rate=req.proposed_rate,
                currency=req.currency,
                connects_charged=0,
                connects_remaining=int(acc["connects_balance"]),
                simulated_client_outcome="draft_saved",
                created_at=now,
            )
            self._save_idempotency(conn, req.idempotency_key, "POST /proposals/draft", payload, resp)
            return resp

    def submit_proposal(self, req: ProposalSubmitRequest) -> ProposalResponse:
        payload = req.model_dump()
        job = self.get_job(req.job_id)
        client = self.get_client(job.client_id)

        with self._connect() as conn:
            if not req.dry_run:
                cached = self._check_idempotency(conn, req.idempotency_key, "POST /proposals/submit", payload)
                if cached is not None:
                    return ProposalResponse.model_validate_json(cached)

            acc = self._ensure_account(conn, req.freelancer_id)
            current_bal = int(acc["connects_balance"])
            cost = int(job.connects_cost)

            combined_text = req.cover_letter + "\n" + "\n".join(req.screening_answers.values())
            policy_violations = self._evaluate_outbound_policy(combined_text)
            fee_usd = _compute_marketplace_fee(req.proposed_rate)
            net_usd = round(req.proposed_rate - fee_usd, 2)

            preview = DryRunPreview(
                job_id=job.job_id,
                job_title=job.title,
                freelancer_id=req.freelancer_id,
                proposed_rate=req.proposed_rate,
                currency=req.currency,
                estimated_fee_usd=fee_usd,
                estimated_net_usd=net_usd,
                connects_cost=cost,
                current_connects_balance=current_bal,
                remaining_connects_after=current_bal - cost,
                policy_passed=(len(policy_violations) == 0 and current_bal >= cost),
                policy_warnings=policy_violations
                + ([] if current_bal >= cost else [f"Insufficient Connects: need {cost}, have {current_bal}."]),
            )

            now = _utc_now()
            if req.dry_run:
                return ProposalResponse(
                    proposal_id=f"dryrun_{job.job_id}",
                    job_id=job.job_id,
                    freelancer_id=req.freelancer_id,
                    status="dry_run_ok",
                    proposed_rate=req.proposed_rate,
                    currency=req.currency,
                    connects_charged=0,
                    connects_remaining=current_bal,
                    simulated_client_outcome="dry_run_preview_only",
                    dry_run_preview=preview,
                    created_at=now,
                )

            if policy_violations:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail={"error": "marketplace_policy_violation", "violations": policy_violations},
                )

            if current_bal < cost:
                raise HTTPException(
                    status_code=status.HTTP_402_PAYMENT_REQUIRED,
                    detail=f"Insufficient Connects balance ({current_bal}) for job requiring {cost} Connects.",
                )

            new_bal = current_bal - cost
            prop_id = f"prop_{hashlib.sha256(req.idempotency_key.encode()).hexdigest()[:10]}"
            conn.execute(
                "UPDATE accounts SET connects_balance = ? WHERE freelancer_id = ?",
                (new_bal, req.freelancer_id),
            )
            conn.execute(
                """
                INSERT INTO connects_ledger (
                    entry_id, freelancer_id, delta, balance_after, reason, job_id, proposal_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    f"led_{prop_id}",
                    req.freelancer_id,
                    -cost,
                    new_bal,
                    f"Proposal submitted for {job.job_id}",
                    job.job_id,
                    prop_id,
                    now,
                ),
            )

            outcome_status, client_reply_text = self._simulate_client_reaction(job, client, req)
            thread_id: str | None = None

            if client_reply_text is not None:
                thread_id = f"thread_{prop_id}"
                conn.execute(
                    """
                    INSERT INTO threads (thread_id, job_id, client_id, freelancer_id, status, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (thread_id, job.job_id, client.client_id, req.freelancer_id, outcome_status, now),
                )
                conn.execute(
                    """
                    INSERT INTO messages (message_id, thread_id, sender, text, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (f"msg_{prop_id}_1", thread_id, "freelancer", req.cover_letter, now),
                )
                conn.execute(
                    """
                    INSERT INTO messages (message_id, thread_id, sender, text, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (f"msg_{prop_id}_2", thread_id, "client", client_reply_text, _utc_now()),
                )

            conn.execute(
                """
                INSERT INTO proposals (
                    proposal_id, job_id, freelancer_id, status, cover_letter,
                    screening_answers_json, proposed_rate, currency,
                    cited_evidence_ids_json, connects_charged, thread_id,
                    simulated_outcome, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    prop_id,
                    job.job_id,
                    req.freelancer_id,
                    outcome_status,
                    req.cover_letter,
                    json.dumps(req.screening_answers),
                    req.proposed_rate,
                    req.currency,
                    json.dumps(req.cited_evidence_ids),
                    cost,
                    thread_id,
                    outcome_status,
                    now,
                    now,
                ),
            )

            resp = ProposalResponse(
                proposal_id=prop_id,
                job_id=job.job_id,
                freelancer_id=req.freelancer_id,
                status=outcome_status,  # type: ignore[arg-type]
                proposed_rate=req.proposed_rate,
                currency=req.currency,
                connects_charged=cost,
                connects_remaining=new_bal,
                thread_id=thread_id,
                simulated_client_outcome=outcome_status,
                dry_run_preview=preview,
                created_at=now,
            )
            self._save_idempotency(conn, req.idempotency_key, "POST /proposals/submit", payload, resp)
            return resp

    def list_threads(self) -> list[ThreadResponse]:
        with self._connect() as conn:
            t_rows = conn.execute("SELECT * FROM threads ORDER BY updated_at DESC").fetchall()
            results: list[ThreadResponse] = []
            for tr in t_rows:
                m_rows = conn.execute(
                    "SELECT * FROM messages WHERE thread_id = ? ORDER BY rowid ASC", (tr["thread_id"],)
                ).fetchall()
                results.append(
                    ThreadResponse(
                        thread_id=tr["thread_id"],
                        job_id=tr["job_id"],
                        client_id=tr["client_id"],
                        freelancer_id=tr["freelancer_id"],
                        status=tr["status"],
                        messages=[
                            MessageItem(
                                message_id=m["message_id"],
                                thread_id=m["thread_id"],
                                sender=m["sender"],
                                text=m["text"],
                                timestamp=m["timestamp"],
                            )
                            for m in m_rows
                        ],
                    )
                )
            return results

    def get_thread_messages(self, thread_id: str) -> ThreadResponse:
        with self._connect() as conn:
            tr = conn.execute("SELECT * FROM threads WHERE thread_id = ?", (thread_id,)).fetchone()
            if tr is None:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Thread '{thread_id}' not found")
            m_rows = conn.execute(
                "SELECT * FROM messages WHERE thread_id = ? ORDER BY rowid ASC", (thread_id,)
            ).fetchall()

        return ThreadResponse(
            thread_id=tr["thread_id"],
            job_id=tr["job_id"],
            client_id=tr["client_id"],
            freelancer_id=tr["freelancer_id"],
            status=tr["status"],
            messages=[
                MessageItem(
                    message_id=m["message_id"],
                    thread_id=m["thread_id"],
                    sender=m["sender"],
                    text=m["text"],
                    timestamp=m["timestamp"],
                )
                for m in m_rows
            ],
        )

    def post_thread_message(self, thread_id: str, req: SendMessageRequest) -> SendMessageResponse:
        payload = {"thread_id": thread_id, **req.model_dump()}
        endpoint = f"POST /threads/{thread_id}/messages"

        with self._connect() as conn:
            cached = self._check_idempotency(conn, req.idempotency_key, endpoint, payload)
            if cached is not None:
                return SendMessageResponse.model_validate_json(cached)

            tr = conn.execute("SELECT * FROM threads WHERE thread_id = ?", (thread_id,)).fetchone()
            if tr is None:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Thread '{thread_id}' not found")

            violations = self._evaluate_outbound_policy(req.text)
            if violations:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail={"error": "marketplace_policy_violation", "violations": violations},
                )

            now = _utc_now()
            full_text = req.text
            if req.proposed_slots:
                full_text += "\nProposed Interview Slots:\n- " + "\n- ".join(req.proposed_slots)

            msg_id = f"msg_{hashlib.sha256(req.idempotency_key.encode()).hexdigest()[:10]}"
            conn.execute(
                "INSERT INTO messages (message_id, thread_id, sender, text, timestamp) VALUES (?, ?, ?, ?, ?)",
                (msg_id, thread_id, "freelancer", full_text, now),
            )
            sent_msg = MessageItem(
                message_id=msg_id,
                thread_id=thread_id,
                sender="freelancer",
                text=full_text,
                timestamp=now,
            )

            # Simulate client acknowledgment
            new_status = "interview_scheduled" if req.proposed_slots else "active_discussion"
            reply_id = f"msg_cli_{hashlib.sha256((req.idempotency_key + ':reply').encode()).hexdigest()[:10]}"
            reply_ts = _utc_now()
            if req.proposed_slots:
                reply_text = (
                    f"Confirmed — {req.proposed_slots[0]} works well for our engineering lead. "
                    f"We have sent the calendar invite inside the platform workspace."
                )
            else:
                reply_text = (
                    "Thank you for the clear technical breakdown. We are reviewing this with our team "
                    "and will follow up with next steps shortly."
                )

            conn.execute(
                "INSERT INTO messages (message_id, thread_id, sender, text, timestamp) VALUES (?, ?, ?, ?, ?)",
                (reply_id, thread_id, "client", reply_text, reply_ts),
            )
            conn.execute(
                "UPDATE threads SET status = ?, updated_at = ? WHERE thread_id = ?",
                (new_status, reply_ts, thread_id),
            )

            sim_reply = MessageItem(
                message_id=reply_id,
                thread_id=thread_id,
                sender="client",
                text=reply_text,
                timestamp=reply_ts,
            )

            resp = SendMessageResponse(
                thread_id=thread_id,
                sent_message=sent_msg,
                simulated_client_reply=sim_reply,
                thread_status=new_status,
            )
            self._save_idempotency(conn, req.idempotency_key, endpoint, payload, resp)
            return resp