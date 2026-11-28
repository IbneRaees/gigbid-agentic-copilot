# Runbook

## Prerequisites (Linux / Ubuntu)
- Docker Engine (>= 24) and Docker Compose (v2).
- Python 3.11 (for local tooling, optional).
- Node.js 20+ (for local UI development, optional).

## Quick Local Development Setup
1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd gigbid
   ```
2. **Start the full stack**
   ```bash
   docker compose -f infra/docker-compose.yml up --build -d
   ```
   This brings up:
   - `gigsim` (mock marketplace) on **8081**
   - MCP services (`portfolio:8091`, `marketplace:8092`, `workspace:8093`)
   - PostgreSQL (`5432`) and Redis (`6379`) for persistence & broker.
   - FastAPI backend (`8000`).
   - Next.js web UI (`3000`).
3. **Run the demo** (or any custom run)
   ```bash
   make demo
   ```
   The command:
   - Launches a sample bid run via the FastAPI `/runs/bid` endpoint.
   - Pauses for human approval (see the "Approval Queue" UI).
   - Submits the proposal on approval.
4. **Run the full evaluation**
   ```bash
   python -m evals.runners.run_full_eval --dataset evals/datasets/dev_set.json
   ```
   Results are written to `results/full_eval_report.json`.
5. **Run judge calibration**
   ```bash
   python -m evals.runners.judge_calibration --dataset evals/datasets/dev_set.json
   ```
   Output placed at `results/judge_calibration.json`.

## Stopping the Stack
```bash
docker compose -f infra/docker-compose.yml down -v
```
This removes containers and associated volumes.

## Common Commands
- `make dev` – Start API & UI in watch mode (requires Python & Node locally).
- `make test` – Execute unit and integration tests (run on Linux).
- `make lint` – Run `ruff` for Python linting and `eslint` for TypeScript.
- `make docs` – Build Markdown docs into HTML (optional).

## Troubleshooting
- **Port conflicts** – Adjust ports in `infra/docker-compose.yml`.
- **Database migrations** – The project uses `alembic`; run `alembic upgrade head` inside the `api` container if schema changes.
- **Missing environment variables** – Copy `.env.example` to `.env` and fill in any required secrets (e.g., API keys for paid model profiles). The offline demo does not require external keys.
