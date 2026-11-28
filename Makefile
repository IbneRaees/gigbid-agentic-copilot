.PHONY: help install dev dev-api dev-web dev-mcp up down logs test test-unit test-integration test-contract lint eval rebuild-index demo clean

PY := uv run python
COMPOSE := docker compose -f infra/docker-compose.yml

help:
	@echo "GigBid (AI-4016) Make Targets:"
	@echo "  make install        - Sync Python & Node dependencies (uv + npm)"
	@echo "  make up             - Start full Docker Compose stack (Postgres, Redis, GigSim, MCPs, API, Web, Worker)"
	@echo "  make down           - Stop Docker Compose stack"
	@echo "  make dev            - Run local GigSim + MCPs + FastAPI + Next.js (offline_demo/open)"
	@echo "  make test           - Run all pytest suites (unit + contract + integration)"
	@echo "  make lint           - Run ruff check"
	@echo "  make eval           - Run RAG comparison, 25-case full eval, ablation, and judge calibration"
	@echo "  make rebuild-index  - Rebuild hybrid dense + BM25 RAG index"
	@echo "  make demo           - Run deterministic offline end-to-end demo script"

install:
	uv sync
	cd apps/web && npm install

up:
	$(COMPOSE) up -d --build

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f --tail=100

dev-gigsim:
	uv run uvicorn gigsim.app:app --host 0.0.0.0 --port 8081

dev-mcp:
	uv run python -m mcp_servers.portfolio.server & \
	uv run python -m mcp_servers.marketplace.server & \
	uv run python -m mcp_servers.workspace.server & \
	wait

dev-api:
	uv run uvicorn apps.api.main:app --host 0.0.0.0 --port 8000 --reload

dev-web:
	cd apps/web && npm run dev

dev:
	@echo "Starting GigSim (:8081), MCPs (:8091-8093), FastAPI (:8000), and Next.js (:3000)..."
	uv run uvicorn gigsim.app:app --port 8081 & \
	uv run python -m mcp_servers.portfolio.server & \
	uv run python -m mcp_servers.marketplace.server & \
	uv run python -m mcp_servers.workspace.server & \
	uv run uvicorn apps.api.main:app --port 8000 & \
	(cd apps/web && npm run dev) & \
	wait

lint:
	uv run ruff check agent apps rag mcp_servers worker evals tests

test-unit:
	GIGBID_OFFLINE_EMBEDDINGS=true uv run pytest tests/unit -v

test-contract:
	uv run pytest tests/contract -v

test-integration:
	GIGBID_OFFLINE_EMBEDDINGS=true MODEL_PROFILE=offline_demo uv run pytest tests/integration -v

test:
	GIGBID_OFFLINE_EMBEDDINGS=true MODEL_PROFILE=offline_demo uv run pytest tests/ -v

rebuild-index:
	$(PY) -m rag.index.hybrid_store --rebuild

eval:
	GIGBID_OFFLINE_EMBEDDINGS=true MODEL_PROFILE=offline_demo $(PY) -m evals.runners.run_rag_eval
	GIGBID_OFFLINE_EMBEDDINGS=true MODEL_PROFILE=offline_demo $(PY) -m evals.runners.run_full_eval
	GIGBID_OFFLINE_EMBEDDINGS=true MODEL_PROFILE=offline_demo $(PY) -m evals.runners.run_extension_ablation
	GIGBID_OFFLINE_EMBEDDINGS=true MODEL_PROFILE=offline_demo $(PY) -m evals.runners.judge_calibration

demo:
	GIGBID_OFFLINE_EMBEDDINGS=true MODEL_PROFILE=offline_demo $(PY) -c "\
	import asyncio; \
	from langgraph.types import Command; \
	from agent.graphs.bid_graph import build_bid_graph; \
	from agent.state import create_initial_bid_state; \
	g = build_bid_graph(); \
	s = create_initial_bid_state(job_id='job_001', freelancer_id='persona_aiml_01', model_profile='offline_demo'); \
	cfg = {'configurable': {'thread_id': s['thread_id']}}; \
	r1 = asyncio.run(g.ainvoke(s, config=cfg)); \
	print('Interrupted at HITL:', '__interrupt__' in r1, 'Grounding:', r1['verification_report']['grounding_score']); \
	r2 = asyncio.run(g.ainvoke(Command(resume={'approved': True}), config=cfg)); \
	print('Submitted status:', r2['submission_result']['status'], 'Connects charged:', r2['submission_result']['connects_charged'])"

clean:
	rm -rf .pytest_cache .ruff_cache data/processed/*.db data/processed/*.json logs/*.jsonl
