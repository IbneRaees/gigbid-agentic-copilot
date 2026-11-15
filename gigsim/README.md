GigSim — Standalone Marketplace Simulator (port 8081)

GigSim is a self-contained, deterministic freelance marketplace simulator built for GigBid (AI-4016). It mirrors real-world marketplace APIs (job discovery, client histories, Connects economy, proposal drafts/submissions, and multi-turn messaging) backed by local SQLite persistence (gigsim/gigsim.db).

Key Features

Zero-Config SQLite Persistence: Automatically initializes gigsim/gigsim.db and seeds jobs, clients, freelancer Connects accounts, and conversation threads from data/*.json on first boot.

Bearer Token Auth & Rate Limiting: All endpoints (except /health) require Authorization: Bearer <GIGSIM_API_TOKEN> and enforce per-token sliding-window rate limits (GIGSIM_RATE_LIMIT_RPM, default 120 RPM).

Idempotency-Key Deduplication: Every write endpoint (POST /proposals/draft, POST /proposals/submit, POST /threads/{thread_id}/messages) requires an idempotency_key. Retried requests return the exact cached response without double-charging Connects or duplicating messages; mismatched payloads return 409 Conflict.

Dry-Run Previews & Policy Enforcement: POST /proposals/submit supports dry_run: true to compute tiered service fees (POL-007), verify Connects availability (POL-006), and enforce anti-circumvention rules (POL-001–POL-003) before charging Connects.

Simulated Client Outcomes: Submitted proposals are scored deterministically on portfolio evidence citations, quantitative metrics, skill overlap, opening quality, and budget alignment, generating realistic interview, replied, viewed, or submitted outcomes and opening client message threads.

Fault Injection Header: Pass X-GigSim-Simulate-Error: 429 or 503 in integration tests to verify retry backoff and degraded-mode UI banners.

Running Locally

# 1. Generate seed datasets (if not already present)
uv run python data/build_datasets.py

# 2. Launch GigSim on port 8081
uv run uvicorn gigsim.app:app --host 0.0.0.0 --port 8081 --reload



Quick Verification (curl)

# Health check (unauthenticated)
curl -s http://localhost:8081/health | jq .

# Search jobs (authenticated)
curl -s -H "Authorization: Bearer gigsim-dev-token-2026" \
  "http://localhost:8081/jobs?skill=LangGraph&min_budget=2000" | jq '.total, .jobs[0].title'

# Check Connects balance
curl -s -H "Authorization: Bearer gigsim-dev-token-2026" \
  "http://localhost:8081/account/connects?freelancer_id=persona_aiml_01" | jq .

# Dry-run proposal preview
curl -s -X POST http://localhost:8081/proposals/submit \
  -H "Authorization: Bearer gigsim-dev-token-2026" \
  -H "Content-Type: application/json" \
  -d '{
    "idempotency_key": "test-dryrun-001",
    "job_id": "job_001",
    "freelancer_id": "persona_aiml_01",
    "cover_letter": "In our LangGraph legal contract review deployment (ev_aiml_001), we cut review time by 62% with PostgresSaver checkpointing and 1.8% unsupported claims.",
    "proposed_rate": 3800,
    "cited_evidence_ids": ["ev_aiml_001"],
    "dry_run": true
  }' | jq .

