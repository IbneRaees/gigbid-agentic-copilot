# AI Usage & Starter-Kit Adaptation Log (`docs/AI_USAGE.md`)

**Course:** AI-4016 — R&D Assignment II (GigBid)

**Policy Compliance:** Per assignment guidelines, every architectural decision, line of generated code, schema contract, and simulator rule has been reviewed, tested, and verified by the student author.

## 1. Adaptations from `GigBid_Student_Starter_Kit_v0.1.zip`

The TA starter kit provided a thin directory skeleton and baseline configuration conventions. Below is the exact mapping of what was adopted from the starter kit versus engineered/extended in Phase 1 and Phase 2A:

1. **Environment & Port Alignment (`.env.example`, `agent/config.yaml`, `infra/docker-compose.yml`):**

   * Adopted the starter kit's canonical environment variable names (`MARKETPLACE_MODE`, `GIGSIM_API_TOKEN`, `GIGSIM_BASE_URL=http://localhost:8081`, `MODEL_PROFILE`).

   * Extended `agent/config.yaml` with dual model profiles (`open` vs. `paid`), shared CPU-capable open-weight embeddings/reranker (`BAAI/bge-m3` and `BAAI/bge-reranker-v2-m3`), node-specific model routing, token/step/latency budgets, explicit memory promotion rules, and externalized marketplace policy parameters (prepared for the November 2 change request).

   * Standardized `infra/docker-compose.yml` as the single canonical Docker Compose topology including `postgres` (`pgvector`), `redis`, `gigsim` (`:8081`), the three MCP servers (`:8091`–`:8093`), `api` (`:8000`), `worker`, `web` (`:3000`), and an opt-in `ollama` profile (`--profile ollama`).

2. **Typed Hand-Off Contracts (`agent/handoffs.py`):**

   * Aligned field naming conventions (`required_skills`, `budget{min,max,currency}`, `duration_days`, `screening_questions`, `warning_signals`, `connects_cost`, `evidence_id`) with the starter kit while implementing complete Pydantic v2 validation across all 11 inter-agent schemas (`JobSpec`, `AnalystReport`, `RiskReport`, `FitReport`, `BidDecision`, `PriceQuote`, `ProposalDraft`, `Critique`, `ComplianceVerdict`, `ClientReplyIntent`, `ClientReplyDraft`).

3. **Standalone Marketplace Simulator (`gigsim/`):**

   * Built the complete FastAPI + SQLite simulator (`gigsim/app.py`, `gigsim/store.py`, `gigsim/models.py`) on port `8081` with Bearer token auth, per-token sliding-window rate limiting, `idempotency_key` deduplication, `dry_run` preview calculation, outbound policy enforcement, and deterministic client reply simulation.

4. **Reproducible Dataset Pipeline (`data/build_datasets.py`):**

   * Built an offline-first seed generator and optional Kaggle CSV ingestor producing `data/jobs.json`, `data/client_history.json`, `data/freelancers.json`, `data/portfolio_items.json`, `data/messages.json`, and `data/policy_pack/marketplace_policy.md` with strict `"data_provenance"` tagging.

## 2. AI Coding Assistant Usage Log

| Phase | Tool / Model Used | Files Created / Modified | Purpose & Assistance Scope | Verification & Testing Performed | 
| ----- | ----- | ----- | ----- | ----- | 
| Phase 1 | Claude / Gemini | `pyproject.toml`, `agent/state.py`, `Makefile` | Scaffolding directory layout, LangGraph `TypedDict` state reducers, and `uv` dependency pinning | Verified imports, `uv lock` resolution, and reducer merge semantics | 
| Phase 2A | Gemini | `.env.example`, `agent/config.yaml`, `infra/docker-compose.yml` | Aligning starter-kit env vars, externalizing policy rules, and wiring `gigsim` into canonical compose | Validated YAML syntax, Docker Compose config (`docker compose config`), and port bindings | 
| Phase 2A | Gemini | `agent/handoffs.py` | Implementing Pydantic v2 models for all Bid & Conversation graph hand-offs | Verified strict type validation, `min <= max` budget normalization, and JSON round-trip serialization | 
| Phase 2A | Gemini | `data/build_datasets.py`, `data/policy_pack/marketplace_policy.md` | Writing deterministic seed dataset generator, optional Kaggle CSV loader, and 9-rule policy pack | Executed script (`RNG_SEED=42`), verified SHA-256 manifest, and audited scam/injection edge cases | 
| Phase 2A | Gemini | `gigsim/__init__.py`, `gigsim/models.py`, `gigsim/store.py`, `gigsim/app.py`, `gigsim/README.md` | Building FastAPI + SQLite marketplace simulator with auth, rate limits, idempotency, and dry-run previews | Tested all REST endpoints via `curl` / `TestClient`, verified idempotency `409` conflicts and Connects deduction | 
| Phase 2A | Gemini | `docs/DATA_CARD.md`, `docs/AI_USAGE.md` | Structuring dataset provenance documentation and engineering adaptation log | Cross-checked all schema field names and counts against `data/build_datasets.py` | 

## 3. Concrete AI Errors Caught & Corrected During Review

Per assignment requirements, below are specific issues produced during AI-assisted drafting that were caught and fixed during manual code review:

1. **Dry-Run Side Effect on Idempotency Keys (`gigsim/store.py`):**

   * *Initial AI Bug:* The initial draft of `submit_proposal()` checked and saved `idempotency_key` unconditionally even when `dry_run=True`. If a client previewed a proposal with `dry_run=True` and then submitted with the same key and `dry_run=False`, it triggered a `409 Conflict` instead of executing the submission.

   * *Fix Applied:* Guarded `_check_idempotency` and `_save_idempotency` behind `if not req.dry_run:` so `dry_run=True` remains 100% side-effect-free on both Connects balances and idempotency records.

2. **Unconditional GPU / Ollama Startup Failure (`infra/docker-compose.yml`):**

   * *Initial AI Bug:* `ollama` was initially placed in the default service group without a Compose `profile`, which would cause slow multi-gigabyte image pulls or failures on CPU-only evaluation hosts using hosted open-weight providers (Groq/OpenRouter).

   * *Fix Applied:* Assigned `profiles: ["ollama", "local-gpu"]` to the `ollama` container and removed hard `depends_on` blockers from `api`/`worker` so `docker compose up` succeeds immediately in both local-GPU and hosted-open modes.

3. **YAML Environment Variable Interpolation Syntax (`infra/docker-compose.yml`):**

   * *Initial AI Bug:* Several environment variables in the initial compose draft had escaped dollar signs (`\${POSTGRES_USER}`), which Docker Compose treats as literal strings rather than interpolating from `.env`.

   * *Fix Applied:* Standardized all compose substitutions to valid `${VAR:-default}` syntax and verified with `docker compose -f infra/docker-compose.yml config`.