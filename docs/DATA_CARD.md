# GigBid Data Card (`docs/DATA_CARD.md`)

**Project:** GigBid — Multi-Agent Freelancer Copilot (AI-4016)  
**Pipeline Script:** `data/build_datasets.py` (`RNG_SEED = 42`)  
**Manifest Output:** `data/processed/dataset_manifest.json`  

---

## 1. Overview & Purpose

GigBid requires realistic, reproducible marketplace data to evaluate retrieval-augmented generation (RAG), multi-agent routing, risk/scam detection, prompt-injection resilience, pricing ROI, and multi-turn client negotiation without risking real marketplace accounts or violating platform terms of service.

To satisfy both realism and deterministic offline reproducibility, `data/build_datasets.py` supports a **hybrid ingestion + deterministic synthesis pipeline**:
1. **Optional Real Marketplace Corpus (`"data_provenance": "kaggle_real"`):** Ingests real historical Upwork job postings from Kaggle CSV files when placed in `data/raw/`.
2. **Deterministic Synthetic Corpus (`"data_provenance": "synthetic"`):** Generates 48 structured job postings, matching client histories, 2 distinct freelancer personas, 24 verifiable portfolio evidence items, and 8 multi-turn client message threads out-of-the-box with zero external downloads required.

Every single JSON record in `data/*.json` includes an explicit `"data_provenance"` field (`"kaggle_real"` or `"synthetic"`) so evaluations and reports never conflate real historical postings with synthetic test fixtures.

---

## 2. Dataset Sources & Provenance Separation

### 2.1 Optional Kaggle Upwork Datasets (`data/raw/*.csv`)
When CSV exports from the instructor-approved Kaggle Upwork collections are placed in `data/raw/`, `load_kaggle_jobs()` normalizes them into GigBid's canonical job schema:
- `asaniczka/upwork-job-postings-dataset-2024-50k-records` (50k+ historical job titles, descriptions, skills, budgets, and countries)
- `asaniczka/all-jobs-on-upwork-200k-plus` (200k+ historical listings used for category & pricing distribution checks)
- `ahmedmyalo/upwork-freelance-jobs-60k` (60k job postings with hourly/fixed budget splits)
- `asaniczka/freelance-contracts-dataset-1-3-million-entries` (market rate benchmarking statistics)

**Normalization & Provenance Rules:**
- Records imported from CSVs receive deterministic SHA-256 IDs prefixed with `job_kg_` and `cli_kg_`, retain their `"source_file"` filename, and are tagged `"data_provenance": "kaggle_real"`.
- Raw Kaggle CSV files under `data/raw/` and generated manifests under `data/processed/` are git-ignored so large external files are never committed to the repository.

### 2.2 Deterministic Synthetic Seed Corpora (`RNG_SEED = 42`)
To guarantee deterministic regression testing across CI and grading environments, `data/build_datasets.py` generates the following synthetic files:

| File Path | Record Count | Provenance Tag | Purpose |
| :--- | :---: | :---: | :--- |
| `data/jobs.json` | 48 (+ optional Kaggle) | `synthetic` / `kaggle_real` | Job stream covering high-fit matches, vague briefs, lowballs, scams, and prompt injections |
| `data/client_history.json` | 10 (+ optional Kaggle) | `synthetic` / `kaggle_real` | Client trust metrics (`payment_verified`, `hire_rate`, `total_spent_usd`, `past_reviews`) |
| `data/freelancers.json` | 2 | `synthetic` | Two distinct freelancer personas (`persona_aiml_01`, `persona_fullstack_02`) |
| `data/portfolio_items.json` | 24 | `synthetic` | Ground-truth evidence corpus (`case_study`, `github_repo`, `past_proposal`, `cv_section`) |
| `data/messages.json` | 8 threads (16 msgs) | `synthetic` | Multi-turn client conversations testing intent classification and calendar slot drafting |
| `data/policy_pack/marketplace_policy.md` | 9 rules (`POL-001`–`009`) | `synthetic` (authored) | Ground-truth compliance rules enforced by `Compliance Guard` and `GigSim` |

---

## 3. Stratification of Synthetic Jobs (`data/jobs.json`)

The 48 deterministic synthetic jobs are stratified across distinct evaluation regimes:
- **High-Fit AI/ML & Agentic Jobs (`job_001`, `job_002`, `job_004`, `job_010`, plus batch variants):** Multi-requirement engineering briefs testing requirement decomposition (`V1`), hybrid sparse+dense retrieval (`V2`), and cross-encoder reranking (`V3`).
- **High-Fit Full-Stack Web Jobs (`job_003`, `job_011`, plus batch variants):** Next.js, TypeScript, FastAPI, SSE, OAuth 2.1, and Playwright specs testing persona routing and portfolio grounding.
- **Vague / Underspecified Client Briefs (`job_009`):** Short, non-technical descriptions designed to test HyDE (`V7`) query expansion.
- **Off-Platform Scam & Anti-Circumvention Traps (`job_005`):** Unverified client (`cli_scam_01`) requesting Telegram/WhatsApp contact and direct USDT/PayPal payment (`POL-001`, `POL-002`).
- **Unpaid Free-Work Speculative Traps (`job_006`):** Low-trust client (`cli_scam_02`) demanding an unpaid working prototype prior to contract award (`POL-003`).
- **Indirect Prompt Injection Traps (`job_007`):** Otherwise legitimate FastAPI/Postgres job post containing an embedded `[SYSTEM OVERRIDE — IGNORE ALL PREVIOUS INSTRUCTIONS...]` payload attempting to zero out `risk_score`, inflate the bid to `$99,999`, and exfiltrate an email address (`POL-009`).
- **Economic Mismatch & Zero-Fit Skips (`job_008`, `job_012`, lowball batch items):** Jobs whose budget-to-scope ratio violates the freelancer's floor/Connects ROI (`POL-006`) or whose required stack (e.g., Unity 3D / HLSL shaders) has zero portfolio overlap.

---

## 4. Schema Reference

### 4.1 `data/jobs.json`
- `job_id` (`str`): Primary key (e.g., `"job_001"`).
- `client_id` (`str`): Foreign key into `client_history.json`.
- `title` (`str`), `description` (`str`): Raw untrusted marketplace text.
- `required_skills` (`list[str]`), `nice_to_have_skills` (`list[str]`): Skill tags.
- `budget` (`object`): `{ "min": float, "max": float, "currency": "USD", "budget_type": "fixed" | "hourly" }`.
- `duration_days` (`int`), `experience_level` (`str`), `screening_questions` (`list[str]`).
- `connects_cost` (`int`): Connects required to submit a proposal (`4`–`16`).
- `domain_category` (`str`), `eval_tag` (`str`), `gold_decision` (`"bid" | "skip"`), `gold_evidence_ids` (`list[str]`).
- `data_provenance` (`"kaggle_real" | "synthetic"`).

### 4.2 `data/portfolio_items.json`
- `evidence_id` (`str`): Unique verifiable identifier (e.g., `"ev_aiml_001"`, `"ev_fs_001"`).
- `freelancer_id` (`str`): Owner persona (`"persona_aiml_01"` or `"persona_fullstack_02"`).
- `type` (`str`): `"case_study" | "github_repo" | "past_proposal" | "cv_section"`.
- `title` (`str`), `client_industry` (`str`), `date` (`YYYY-MM-DD`), `outcome` (`"won" | "interview" | "replied" | "ignored"`).
- `skills` (`list[str]`), `summary` (`str`), `full_text` (`str`), `metrics` (`dict`), `repo_url` (`str | null`).
- `data_provenance` (`"synthetic"`).

---

## 5. Licensing, Privacy & Limitations

1. **Licensing:**
   - All synthetic seed files (`jobs.json`, `client_history.json`, `freelancers.json`, `portfolio_items.json`, `messages.json`, `marketplace_policy.md`) are original works created for AI-4016 and licensed under MIT along with the repository.
   - Optional Kaggle datasets remain subject to their respective Kaggle dataset licenses and are processed strictly locally without redistribution in git.
2. **Privacy & PII Safety:**
   - All freelancer names, client organizations, GitHub URLs, and contact strings in scam fixtures (e.g., `+1-555-019-8372`, `admin@evil-exfil.com`) are 100% synthetic/reserved RFC-compliant test values. No real freelancer PII or live Upwork client data is stored in `data/`.
3. **Limitations & Hidden Set Isolation:**
   - GigSim's client reply simulator (`gigsim/store.py`) uses deterministic heuristic scoring over evidence citations, quantitative metrics, skill coverage, opening quality, and budget alignment; while repeatable for ablation studies, it is a proxy rather than a live human buyer.
   - Per course rules, the instructor's hidden evaluation set is never committed to this repository or used to tune retrieval/decision thresholds.