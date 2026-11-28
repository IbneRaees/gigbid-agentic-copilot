# GigBid AI Agentic Freelancer Copilot

Welcome to the **GigBid** project – an end‑to‑end agentic AI system that scouts freelance jobs, composes winning proposals, and interacts with clients.

## Quick Start (Linux / Ubuntu)
1. **Clone the repository** and `cd` into the project root.
2. **Build & run the stack** with Docker Compose:
   ```bash
   docker compose -f infra/docker-compose.yml up --build -d
   ```
3. **Run the backend** (FastAPI) at `http://localhost:8000`.
4. **Open the web UI** (Next.js) at `http://localhost:3000`.
5. **Execute the demo**:
   ```bash
   make demo
   ```
   This will launch a sample bid run, pause for human approval, and then submit the proposal.

## Repository Layout
```
├─ agent/                # LangGraph agents, nodes, extensions
├─ apps/api/             # FastAPI backend
├─ apps/web/             # Next.js frontend (TypeScript, Tailwind)
├─ worker/               # Celery/Arq background worker
├─ infra/                # Docker‑Compose orchestration
├─ evals/                # Evaluation datasets & runners
├─ docs/                 # Design docs, runbook, reports
├─ tests/                # Unit & integration tests
└─ Makefile              # Common dev commands
```

## Key Features
- **Multi‑agent orchestration** with LangGraph StateGraphs.
- **Hybrid dense + BM25 RAG** for factual grounding.
- **Policy compliance** and claim verification.
- **Connects ROI Optimizer** extension (see `agent/extensions/connects_roi_optimizer.py`).
- **Human‑in‑the‑loop** approval workflow.
- **Full‑stack UI** to monitor runs, edit drafts, and view audit logs.
- **Automated CI/CD** via GitHub Actions.

## Documentation
- Architecture: `docs/ARCHITECTURE.md`
- Runbook & deployment: `docs/RUNBOOK.md`
- Final report & demo script: `docs/FINAL_REPORT.md`, `docs/DEMO_SCRIPT.md`
- Extension details: `docs/EXTENSION.md`

## License & Attribution
The project is released under the MIT License. It uses open‑source models (e.g., `BAAI/bge-small-en-v1.5`) and respects model provider terms.
