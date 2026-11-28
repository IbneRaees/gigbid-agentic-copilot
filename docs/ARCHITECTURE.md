# Architecture Overview

```mermaid
graph LR
    subgraph Client
        C1[Web UI (Next.js)]
        C2[API Calls (Axios)]
    end
    subgraph Backend
        B1[FastAPI (apps/api)]
        B2[LangGraph StateGraph]
        B3[Agent Nodes]
        B4[Memory Store (SQLite/Postgres)]
        B5[MCP Clients]
        B6[Hybrid RAG Store]
        B7[Claim Verifier]
        B8[Connects ROI Optimizer]
    end
    subgraph Worker
        W1[Celery/Arq Tasks]
        W2[Background Jobs]
    end
    subgraph Infra
        I1[Docker Compose]
        I2[Postgres]
        I3[Redis]
        I4[GigSim Marketplace (Mock)]
    end

    C1 -->|REST| B1
    C2 -->|WebSocket SSE| B1
    B1 --> B2
    B2 --> B3
    B3 --> B5
    B3 --> B6
    B3 --> B7
    B3 --> B8
    B5 --> I4
    B6 --> I4
    B6 --> I2
    B4 --> I2
    B1 --> W1
    W1 --> W2
    W2 --> I3
    I1 --> B1
    I1 --> C1
    I1 --> W1
```

### Components
- **Web UI (Next.js)**: Job feed, live trace, approval queue, calendar liaison, dashboards.
- **FastAPI Backend**: Health, job browsing, run management, SSE streaming, approval endpoints, memory & audit APIs.
- **LangGraph Agent**: Orchestrates scouting, analysis, risk assessment, fit matching, pricing, writing, critique/compliance loops, client liaison.
- **Hybrid RAG Store**: Dense embeddings (BAAI/bge‑small‑en‑v1.5) + BM25 for evidence retrieval.
- **Claim Verifier**: Validates generated claims against retrieved evidence.
- **Connects ROI Optimizer**: Extension estimating ROI for up‑sell "connects" offers.
- **MCP Clients**: Interact with portfolio, marketplace, workspace services via Antigravity MCP layer.
- **Background Worker**: Executes long‑running tasks, retries, cron‑style jobs.
- **Infrastructure**: Docker Compose orchestrates all services; PostgreSQL stores persistent state; Redis broker for Celery/Arq; GigSim simulates marketplace.
