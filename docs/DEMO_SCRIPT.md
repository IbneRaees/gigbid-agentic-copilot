# Demo Script (3‑minute video)

**Scene 1 – Introduction (0:00‑0:20)**
- Show project title slide: *GigBid AI Agentic Freelancer Copilot*.
- Brief voice‑over: "We built an end‑to‑end AI system that scouts freelance jobs, writes winning proposals, and negotiates with clients."

**Scene 2 – Architecture Overview (0:20‑0:45)**
- Display the architecture diagram from `docs/ARCHITECTURE.md`.
- Highlight the FastAPI backend, Next.js UI, LangGraph agent, and the Connects ROI Optimizer extension.

**Scene 3 – Starting the Stack (0:45‑1:10)**
- Terminal capture (Linux) running `make dev`.
- Show Docker Compose bringing up services (API on 8000, UI on 3000, GigSim on 8081, MCP services, Postgres, Redis).

**Scene 4 – Browsing Jobs (1:10‑1:30)**
- UI screen: job feed fetching jobs from the GigSim mock marketplace.
- Click on a job – display details.
- Press **"Run Bid"** button.

**Scene 5 – Live Agent Trace (1:30‑1:50)**
- SSE timeline panel updates as the LangGraph state machine progresses through scout, analyst, pricing, writer, critique, etc.
- Highlight a citation badge linking back to a retrieved evidence ID.

**Scene 6 – Human‑in‑the‑Loop Approval (1:50‑2:10)**
- Approval queue pops up with the draft proposal.
- Show the **"Approve"** button being clicked.
- UI shows the proposal being submitted to the mock marketplace.

**Scene 7 – Connects ROI Decision (2:10‑2:30)**
- Open the **"Profitability"** tab.
- Display the ROI Optimizer output: win probability, expected value, and decision (accept).
- Explain how the optimizer informs the pricing strategy.

**Scene 8 – Evaluation Results (2:30‑2:50)**
- Show a snippet of `results/full_eval_report.json` with overall accuracy.
- Show `results/judge_calibration.json` with kappa score.

**Scene 9 – Closing (2:50‑3:00)**
- Summarize key achievements: multi‑agent orchestration, safety loops, ROI optimization, full‑stack UI.
- Call‑to‑action: try the repo, run `make demo`, extend with your own extensions.

*All commands are shown running on Linux/Ubuntu as the repository is designed to be fully reproducible offline.*
