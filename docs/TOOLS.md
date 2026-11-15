# GigBid MCP Tool Reference & Security Matrix (`docs/TOOLS.md`)

**Architecture:** GigBid standardizes all external and internal tool integrations through the **Model Context Protocol (MCP)** using the official MCP Python SDK (`mcp`) and `langchain-mcp-adapters`.

Every tool declares explicit MCP `ToolAnnotations` (`readOnlyHint`, `destructiveHint`, `idempotentHint`), runs behind Bearer-token authentication and per-user sliding-window rate limits, and writes sanitized telemetry to an append-only audit log (`logs/mcp_audit.jsonl`).

---

## 1. Complete MCP Tool Inventory

| Tool Name | Server | Read / Write | Annotations (`readOnly` / `destructive` / `idempotent`) | Auth | Trust Rationale & Guardrails |
| :--- | :--- | :---: | :---: | :--- | :--- |
| `search_evidence` | Portfolio MCP (`:8091`) | Read | `True` / `False` / `True` | Bearer Token (`MCP_SERVER_TOKEN`) | Strictly read-only lexical + skill + outcome retrieval over verified portfolio items (`data/portfolio_items.json`). Cannot mutate portfolio state. |
| `get_case_study` | Portfolio MCP (`:8091`) | Read | `True` / `False` / `True` | Bearer Token (`MCP_SERVER_TOKEN`) | Strictly read-only lookup of a single `evidence_id` for claim verification and inline citation tooltips. |
| `list_skills` | Portfolio MCP (`:8091`) | Read | `True` / `False` / `True` | Bearer Token (`MCP_SERVER_TOKEN`) | Strictly read-only skill-to-evidence aggregation across freelancer profiles and case studies. |
| `search_jobs` | Marketplace MCP (`:8092`) | Read | `True` / `False` / `True` | Bearer Token (`MCP_SERVER_TOKEN`) | Read-only job discovery routed to `GigSim` or `UpworkLive`. Returned job text is wrapped in `<UNTRUSTED_JOB_POST>` delimiters before LLM processing. |
| `get_job` | Marketplace MCP (`:8092`) | Read | `True` / `False` / `True` | Bearer Token (`MCP_SERVER_TOKEN`) | Read-only fetch of job details, screening questions, budget, and Connects cost. |
| `get_client_history` | Marketplace MCP (`:8092`) | Read | `True` / `False` / `True` | Bearer Token (`MCP_SERVER_TOKEN`) | Read-only fetch of client payment verification, hire rate, total spend, and past reviews for the `Risk Assessor` node. |
| `get_connects_balance` | Marketplace MCP (`:8092`) | Read | `True` / `False` / `True` | Bearer Token (`MCP_SERVER_TOKEN`) | Read-only check of remaining Connects balance and ledger history prior to budget gating (`POL-006`). |
| `list_messages` | Marketplace MCP (`:8092`) | Read | `True` / `False` / `True` | Bearer Token (`MCP_SERVER_TOKEN`) | Read-only fetch of active client conversation threads; incoming client text is isolated as untrusted data. |
| `create_proposal_draft` | Marketplace MCP (`:8092`) | Write | `False` / `False` / `True` | Bearer Token (`MCP_SERVER_TOKEN`) | Non-destructive draft creation (`connects_charged=0`). Requires `idempotency_key` and supports `dry_run=True`. |
| `submit_proposal` | Marketplace MCP (`:8092`) | Write | `False` / `True` / `True` | Bearer Token (`MCP_SERVER_TOKEN`) + Human `interrupt()` | **High-trust mutation:** Spends Connects and submits proposal in `gigsim` mode only after human approval. In `upwork_live` mode, **hard-restricted to draft-only output** with manual review URL (`POL-008`). Requires `idempotency_key` & `dry_run`. |
| `send_message` | Marketplace MCP (`:8092`) | Write | `False` / `True` / `True` | Bearer Token (`MCP_SERVER_TOKEN`) + Human `interrupt()` | **External communication:** Dispatches reply in `gigsim` mode only after human approval. In `upwork_live` mode, **hard-restricted to draft-only output** (`POL-008`). Requires `idempotency_key` & `dry_run`. |
| `get_free_calendar_slots` | Workspace MCP (`:8093`) | Read | `True` / `False` / `True` | Bearer Token (`MCP_SERVER_TOKEN`) | Read-only availability check excluding existing tentative holds in local SQLite (`gigsim/workspace.db`). |
| `create_tentative_interview_hold` | Workspace MCP (`:8093`) | Write | `False` / `False` / `True` | Bearer Token (`MCP_SERVER_TOKEN`) | Non-destructive local SQLite tentative hold (`status="tentative_hold"`). Requires `idempotency_key` and supports `dry_run=True`. |
| `create_email_draft` | Workspace MCP (`:8093`) | Write | `False` / `False` / `True` | Bearer Token (`MCP_SERVER_TOKEN`) | **Strictly draft-only (`auto_sent=False`):** Saves local email draft for human review; never sends external emails. Requires `idempotency_key` & `dry_run`. |
| `search_jobs` / `get_job_details` / `get_client_profile` | Official Upwork MCP (`mcp.upwork.com/mcp`) | Read | `True` / `False` / `True` | OAuth 2.1 PKCE (`UPWORK_ACCESS_TOKEN`) | Used exclusively for authenticated read queries when `MARKETPLACE_MODE=upwork_live`; write calls are intercepted locally by `UpworkLiveMarketplaceAdapter` to enforce draft-only compliance. |
| `get_file_contents` / `search_repositories` / `list_commits` | Official GitHub MCP (`@modelcontextprotocol/server-github`) | Read | `True` / `False` / `True` | GitHub PAT (`GITHUB_PERSONAL_ACCESS_TOKEN`) | Read-only repository ingestion into Portfolio RAG (`data/portfolio_items.json` / `pgvector`); never granted write/push scopes. |

---

## 2. MCP Resources & Prompts (Portfolio MCP `:8091`)

### Resources
- `portfolio://case/{id}` — Returns the canonical JSON case study or repository record for `evidence_id = {id}`.
- `portfolio://proposal/{id}` — Returns the canonical historical proposal record for `evidence_id = {id}`.

### Prompt Templates
- `answer_screening_question(question, job_context, freelancer_id)` — Automatically retrieves top matching portfolio evidence items and constructs a grounded prompt forbidding unverified claims.
- `draft_evidence_anchored_opener(job_title, key_requirement, freelancer_id)` — Injects top matching case-study metrics and style preferences to generate a non-generic 2-sentence opening hook.

---

## 3. Transport Security, Idempotency & Audit Logging

1. **Bearer Token Authentication (`BearerAuthAndRateLimitMiddleware`):**
   - All HTTP/SSE requests to `:8091`, `:8092`, and `:8093` (except unauthenticated `GET /health`) require `Authorization: Bearer <token>` validated via constant-time `secrets.compare_digest`.
2. **Per-User Sliding-Window Rate Limiter (`SlidingWindowRateLimiter`):**
   - Tracks request timestamps per `(user_id, token_prefix)` over a rolling 60-second window and returns `429 Too Many Requests` with `Retry-After` when `MCP_RATE_LIMIT_RPM` is exceeded.
3. **Write-Tool Idempotency & Dry-Run Contract (`MCPIdempotencyStore`):**
   - Every write tool (`create_proposal_draft`, `submit_proposal`, `send_message`, `create_tentative_interview_hold`, `create_email_draft`) enforces:
     - `dry_run: bool = False`: When `True`, returns fee/Connects/slot previews without mutating SQLite or consuming `idempotency_key`.
     - `idempotency_key: str`: Replaying a live call with the exact same key and arguments returns the cached response without double-charging Connects or duplicating holds/drafts; reusing the same key with altered arguments raises a conflict error.
4. **Append-Only JSONL Audit Log (`logs/mcp_audit.jsonl`):**
   - Every tool invocation logs `timestamp`, `trace_id`, `user_id`, `server_name`, `tool_name`, `status`, `duration_ms`, `sanitized_args` (with emails, phone numbers, and Bearer tokens automatically scrubbed), and `result_summary`.