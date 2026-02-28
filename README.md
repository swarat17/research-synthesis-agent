# Research Synthesis & Hypothesis Agent

> A **7-node multi-agent pipeline** that takes a scientific topic, autonomously fetches papers from ArXiv, synthesizes findings, detects contradictions, and generates novel research hypotheses — with per-node cost tracking logged to Supabase.

[![CI](https://github.com/swarat17/research-synthesis-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/research-synthesis-agent/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What It Does

Submit a research topic and the pipeline:

1. **Extracts keywords** via GPT-4o-mini to enrich the search query
2. **Fetches papers** from ArXiv (4–20, configurable)
3. **Deduplicates** by normalised title, ranked by citation count
4. **Embeds** papers into Pinecone (text-embedding-3-small) for vector retrieval
5. **Synthesizes** a 400–600 word narrative anchored to your query (Claude Sonnet 4.6)
6. **Detects contradictions** across papers with severity ratings (GPT-4o-mini)
7. **Generates 3 novel hypotheses** with confidence scores and suggested methods (Claude Sonnet 4.6)
8. **Logs cost + latency** per node to Supabase

---

## Architecture

```mermaid
graph LR
    A([User Query]) --> B[Router {GPT-4o-mini}]
    B --> C[ArXiv Fetcher]
    C --> D[Deduplicator]
    D --> E[Synthesizer (Claude Sonnet 4.6)]
    E --> F[Contradiction Detector (GPT-4o-mini)]
    F --> G[Hypothesis Generator (Claude Sonnet 4.6)]
    G --> H[Cost Auditor (Supabase)]
    H --> I([Response])
```

### LLM Routing

| Node | Model | Reason |
|---|---|---|
| Router | GPT-4o-mini | Lightweight keyword extraction — ~$0.00002/call |
| Contradiction Detector | GPT-4o-mini | Structured JSON output, cost-efficient |
| Synthesizer | Claude Sonnet 4.6 | Best long-form scientific coherence |
| Hypothesis Generator | Claude Sonnet 4.6 | Creative reasoning at temp=0.7 |
| Embeddings | text-embedding-3-small | 1536-dim, fast, cheap |

---

## Key Technical Features

- **`CostTracker` singleton** — instruments every LLM call with per-node USD cost and latency; raises `CostLimitExceededError` if a configurable cap is exceeded
- **Query anchoring** — `original_query` is preserved through the pipeline so the synthesis stays focused on what you asked, not on the enriched search string
- **Semantic deduplication** — title normalisation (lowercase, hyphen→space, strip punctuation) + citation-count-aware dedup; papers ranked by citations then year
- **Pinecone vector store** — papers embedded per `query_id` namespace for isolated per-query retrieval
- **Serverless deployment** — FastAPI + Mangum adapter packages the pipeline as an AWS Lambda container image behind HTTP API Gateway
- **Streamlit dashboard** — Research Query UI and live Cost Dashboard backed by Supabase

---

## Performance

| Metric | Value |
|---|---|
| Avg end-to-end latency | ~40s |
| Avg cost per query (10 papers) | ~$0.035 |
| Papers fetched per query | 4–20 (configurable) |
| Synthesis length | 400–600 words |
| Hypotheses generated | 3 |

### Per-Node Cost Breakdown (10 papers)

| Node | Model | Avg input tokens | Avg output tokens | Avg cost |
|---|---|---|---|---|
| Router | GPT-4o-mini | ~80 | ~25 | ~$0.00002 |
| Embeddings | text-embedding-3-small | ~1,200 | — | ~$0.00002 |
| Synthesizer | Claude Sonnet 4.6 | ~1,200 | ~800 | ~$0.015 |
| Contradiction Detector | GPT-4o-mini | ~350 | ~10 | ~$0.00006 |
| Hypothesis Generator | Claude Sonnet 4.6 | ~960 | ~1,300 | ~$0.022 |
| **Total** | | | | **~$0.035** |

> Claude Sonnet accounts for ~99% of cost. GPT-4o-mini and embeddings are negligible.

---

## Local Setup

### Prerequisites

You will need API keys for:
- [OpenAI](https://platform.openai.com) — Router, Contradiction Detector, Embeddings
- [Anthropic](https://console.anthropic.com) — Synthesizer, Hypothesis Generator
- [Pinecone](https://app.pinecone.io) — Vector store (free tier, index: 1536 dims, cosine)
- [Supabase](https://supabase.com) — Cost log storage (free tier)

### Steps

```bash
# 1. Clone
git clone https://github.com/your-username/research-synthesis-agent
cd research-synthesis-agent

# 2. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Fill in your API keys in .env

# 5. Create Supabase table (run once in the Supabase SQL Editor)
# Copy and run the contents of infra/supabase_schema.sql

# 6. Start the API server
uvicorn src.api.main:app --reload

# 7. Start the Streamlit dashboard (separate terminal)
streamlit run frontend/app.py
```

**Or with Docker:**
```bash
docker compose -f docker/docker-compose.yml up
```

The API will be available at `http://localhost:8000` and the dashboard at `http://localhost:8501`.

---

## API Usage

```bash
# Run a research query
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"query": "offline reinforcement learning on medical datasets", "max_papers": 10}'

# Health check
curl http://localhost:8000/health

# Query stats from Supabase
curl http://localhost:8000/stats
```

### Response shape

```json
{
  "query_id": "a1b2c3d4",
  "papers": [...],
  "synthesis": "...",
  "contradictions": [
    {
      "claim_a": "...", "claim_b": "...",
      "paper_a_title": "...", "paper_b_title": "...",
      "severity": "high | medium | low",
      "topic": "..."
    }
  ],
  "hypotheses": [
    {
      "hypothesis": "...",
      "rationale": "...",
      "confidence": 0.82,
      "novelty": "high | medium | low",
      "suggested_method": "...",
      "supporting_papers": ["..."]
    }
  ],
  "cost_report": {
    "total_cost_usd": 0.034,
    "total_latency_ms": 41200,
    "breakdown": [...]
  },
  "errors": []
}
```

---

## Running Tests

```bash
# Unit tests — no API keys needed, runs in ~4s
pytest tests/unit/ -v

# Integration tests — requires real keys in .env, costs ~$0.035
pytest tests/integration/ -m integration -v
```

---

## Deploy to AWS Lambda

```bash
# Requires AWS CLI, SAM CLI, and Docker
chmod +x infra/deploy.sh
./infra/deploy.sh
```

---

## Project Structure

```
src/
├── agents/       router, fetchers, deduplicator, synthesizer,
│                 contradiction, hypothesis, cost_auditor
├── graph/        state.py (ResearchState TypedDict), pipeline.py
├── api/          main.py (FastAPI + Mangum), models.py (Pydantic)
├── storage/      pinecone_store.py, supabase_store.py
└── utils/        cost_tracker.py, logger.py
frontend/         app.py (Streamlit), helpers.py
tests/            unit/, integration/, e2e/
docker/           Dockerfile, docker-compose.yml
infra/            template.yaml (SAM), deploy.sh, supabase_schema.sql
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | GPT-4o-mini + embeddings |
| `ANTHROPIC_API_KEY` | Yes | Claude Sonnet 4.6 |
| `PINECONE_API_KEY` | Yes | Vector store |
| `PINECONE_INDEX` | Yes | Index name (1536 dims, cosine) |
| `SUPABASE_URL` | Yes | Project URL |
| `SUPABASE_KEY` | Yes | service_role key |
| `AWS_ACCESS_KEY_ID` | Deploy only | Lambda deployment |
| `AWS_SECRET_ACCESS_KEY` | Deploy only | Lambda deployment |
| `MAX_COST_PER_QUERY` | No | Hard cost cap in USD (default: 0.50) |
| `COST_WARNING_THRESHOLD` | No | Soft warning threshold (default: 0.25) |

---

## License

MIT
