#!/usr/bin/env bash
# =============================================================================
# GigBid Phase 1 scaffold  (Ubuntu 22.04 LTS, Python 3.11+, uv)
#
# Usage:   bash setup.sh [target_dir]        (default target: ./gigbid)
#
# - Creates the EXACT directory layout from the AI-4016 assignment brief
#   (plus the extra paths graders/CI depend on: infra/, .github/, docs/).
# - Creates every empty file listed, never overwriting an existing one.
# - Installs uv if missing and creates a Python 3.11 virtualenv (.venv).
# Idempotent: safe to re-run.
# =============================================================================
set -Eeuo pipefail

log()  { printf '\033[1;34m[setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[fail]\033[0m %s\n' "$*" >&2; exit 1; }
trap 'die "setup failed at line $LINENO"' ERR

# -- Preflight ----------------------------------------------------------------
if [[ -r /etc/os-release ]]; then
  # shellcheck disable=SC1091
  . /etc/os-release
  if [[ "${ID:-}" != "ubuntu" || "${VERSION_ID:-}" != "22.04" ]]; then
    warn "Tested on Ubuntu 22.04; detected ${PRETTY_NAME:-unknown}. Continuing."
  fi
fi

for bin in curl git; do
  command -v "$bin" >/dev/null 2>&1 || die "'$bin' is required. Run: sudo apt-get update && sudo apt-get install -y $bin"
done

ROOT="${1:-gigbid}"
PY_VERSION="3.11"

# -- Directory tree -----------------------------------------------------------
DIRS=(
  apps/web/src/app
  apps/web/src/components
  apps/web/src/lib/api
  apps/web/tests/e2e
  apps/api/routers
  apps/api/auth
  apps/api/schemas
  agent/graphs
  agent/nodes
  agent/prompts
  agent/memory
  rag/ingest
  rag/index
  rag/variants
  mcp_servers/portfolio/tests
  mcp_servers/marketplace/adapters
  mcp_servers/marketplace/tests
  mcp_servers/workspace/tests
  worker
  gigsim                      # PROVIDED by instructor: do not modify
  data/personas               # PROVIDED
  data/policy_pack            # PROVIDED
  data/processed              # generated, git-ignored
  evals/datasets
  evals/runners
  evals/results
  tests/unit
  tests/integration
  tests/contract
  docker                      # compose file requested for Phase 1
  infra/docker                # one Dockerfile per service (assignment layout)
  .github/workflows
  docs/adr
  docs/images
)

FILES=(
  README.md
  Makefile
  pyproject.toml
  .env.example
  .gitignore
  apps/api/main.py
  apps/api/streaming.py
  agent/state.py
  agent/handoffs.py
  agent/graphs/bid_graph.py
  agent/graphs/conversation_graph.py
  agent/memory/store.py
  agent/memory/promotion_rules.py
  agent/mcp_client.py
  agent/marketplace_port.py
  agent/config.yaml
  rag/retriever.py
  rag/verify.py
  mcp_servers/marketplace/server.py
  mcp_servers/portfolio/server.py
  mcp_servers/portfolio/tools.py
  mcp_servers/portfolio/resources.py
  mcp_servers/portfolio/prompts.py
  mcp_servers/marketplace/adapters/gigsim.py
  mcp_servers/marketplace/adapters/upwork_live.py
  mcp_servers/workspace/server.py
  worker/__init__.py
  evals/runners/rag_eval.py
  evals/runners/decision_eval.py
  evals/runners/gigsim_outcomes.py
  evals/runners/judge_calibration.py
  evals/runners/ablation.py
  docker/docker-compose.yml
  infra/docker-compose.yml
  .github/workflows/ci.yml
  .github/workflows/eval.yml
  docs/ARCHITECTURE.md
  docs/DATA_CARD.md
  docs/TOOLS.md
  docs/RUNBOOK.md
  docs/EXTENSION.md
  docs/AI_USAGE.md
)

# Python packages need __init__.py so imports (agent.state, rag.verify, ...) work.
PY_PACKAGES=(
  apps apps/api apps/api/routers apps/api/auth apps/api/schemas
  agent agent/graphs agent/nodes agent/memory
  rag rag/ingest rag/index rag/variants
  mcp_servers mcp_servers/portfolio mcp_servers/marketplace
  mcp_servers/marketplace/adapters mcp_servers/workspace
  evals evals/runners
  tests tests/unit tests/integration tests/contract
)

# Directories that must exist in git even while empty.
KEEP_DIRS=(
  apps/web/src/app apps/web/src/components apps/web/src/lib/api apps/web/tests/e2e
  agent/prompts
  mcp_servers/portfolio/tests mcp_servers/marketplace/tests mcp_servers/workspace/tests
  gigsim data/personas data/policy_pack data/processed
  evals/datasets evals/results infra/docker docs/adr docs/images
)

log "Creating project root: $ROOT"
mkdir -p "$ROOT"
cd "$ROOT"

log "Creating ${#DIRS[@]} directories"
for d in "${DIRS[@]}"; do
  mkdir -p "${d%%[[:space:]]*}"
done

log "Creating empty files (existing files are never overwritten)"
for f in "${FILES[@]}"; do
  mkdir -p "$(dirname "$f")"
  [[ -e "$f" ]] || : > "$f"
done

for p in "${PY_PACKAGES[@]}"; do
  [[ -e "$p/__init__.py" ]] || : > "$p/__init__.py"
done

for k in "${KEEP_DIRS[@]}"; do
  [[ -e "$k/.gitkeep" ]] || : > "$k/.gitkeep"
done

# infra/docker-compose.yml is the path the graders expect; it delegates to the
# Phase 1 compose file so there is a single source of truth.
if [[ ! -s infra/docker-compose.yml ]]; then
  cat > infra/docker-compose.yml <<'YAML'
# Canonical path required by the assignment. Delegates to docker/docker-compose.yml.
# Requires Docker Compose v2.20+ (the `include` element).
include:
  - path: ../docker/docker-compose.yml
YAML
fi

if [[ ! -s .gitignore ]]; then
  cat > .gitignore <<'EOF'
# secrets: never commit
.env
*.pem
*.key
# python
.venv/
__pycache__/
*.py[cod]
.pytest_cache/
.ruff_cache/
.mypy_cache/
*.egg-info/
# generated data and hidden eval set
data/processed/*
!data/processed/.gitkeep
evals/datasets/hidden*
# node / web
node_modules/
.next/
apps/web/playwright-report/
apps/web/test-results/
# editors / os
.vscode/
.idea/
.DS_Store
EOF
fi

# -- uv + virtualenv ----------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  log "Installing uv (https://astral.sh/uv)"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
command -v uv >/dev/null 2>&1 || die "uv not found on PATH after install. Add \$HOME/.local/bin to PATH and re-run."
log "Using $(uv --version)"

log "Creating virtualenv (.venv) with Python >= ${PY_VERSION}"
uv venv --python "${PY_VERSION}" .venv

if [[ -s pyproject.toml ]]; then
  log "Resolving and installing dependencies from pyproject.toml (includes dev group)"
  uv sync --python "${PY_VERSION}" --all-groups || warn "uv sync failed; fix pyproject.toml and run 'uv sync --all-groups'"
else
  warn "pyproject.toml is empty; copy the provided one in, then run 'uv sync --all-groups'"
fi

if [[ ! -e .env && -s .env.example ]]; then
  cp .env.example .env
  log "Created .env from .env.example (edit it; it is git-ignored)"
fi

if [[ ! -d .git ]]; then
  git init -q && log "Initialised git repository"
fi

cat <<EOF

--------------------------------------------------------------------------
GigBid skeleton ready in: $(pwd)

Next steps
  1. Copy the provided files over the empty ones:
       pyproject.toml  .env.example  agent/config.yaml  agent/state.py
       docker/docker-compose.yml
  2. Drop the instructor starter kit into gigsim/ and data/ (do NOT modify gigsim/).
  3. source .venv/bin/activate && uv sync --all-groups
  4. uv run playwright install --with-deps chromium
  5. docker compose -f docker/docker-compose.yml config   # validate
--------------------------------------------------------------------------
EOF
