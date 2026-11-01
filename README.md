# TraceLoop

TraceLoop is a local-first agentic Retrieval-Augmented Generation (RAG) service equipped with an automated trace-to-eval regression loop. It connects production observability directly to continuous integration testing, preventing retrieval decay and tool selection errors from reaching production.

## 1. System Architecture

In production large language model applications, three silent bottlenecks cause failure:
1. Retrieval decay occurs quietly when documents change, embeddings drift, or chunking parameters shift.
2. Production traces and evaluation frameworks live in separate silos, preventing failed traces from becoming regression gates.
3. Multi-agent tool choice is unmeasured, so wrong tool selection is misdiagnosed as bad generation.

TraceLoop solves these bottlenecks by maintaining a continuous automated feedback loop:
Production Agent -> Langfuse Trace Store -> Nightly Trace-to-Dataset Sync -> RAGAS Evaluation -> Fault Classifier -> MLflow Baseline Store -> GitHub Actions Regression Gate.

```
                     +---------------------------------------+
                     |         User / Web Client UI          |
                     +---------------------------------------+
                                         |
                                (POST /ask request)
                                         v
                     +---------------------------------------+
                     |          FastAPI App Server           |
                     |             (app/main.py)             |
                     +---------------------------------------+
                                         |
                                         v
                     +---------------------------------------+
                     |         LangChain Tool Agent          |
                     |            (app/agent.py)             |
                     +---------------------------------------+
                               /                              \
       (Domain Questions)    /                                  \   (Conversational)
                            v                                    v
               +---------------------------+       +---------------------------+
               |  Knowledge Retriever Tool |       |   Fallback Direct Tool    |
               |       (app/tools.py)      |       |   (Outputs "no retrieval")|
               +---------------------------+       +---------------------------+
                             |
                   (Vector Search Query)
                             v
               +---------------------------+
               |   Local Qdrant Instance   |
               |     (Vector Collection)   |
               +---------------------------+
                             |
                             +-----------------------+
                                                     |
                                (All nested spans: Agent, Tool, Generation)
                                                     v
                                       +---------------------------+
                                       |   Self-Hosted Langfuse    |
                                       |       Trace Store         |
                                       +---------------------------+
                                                     |
                                       (Sync failed/flagged traces)
                                                     v
                                       +---------------------------+
                                       |  Trace Promotion Script   |
                                       | (tracing/sync_dataset.py) |
                                       +---------------------------+
                                                     |
                                                     v
                                       +---------------------------+
                                       | RAGAS Evaluator + MLflow  |
                                       | (Context Precision/Recall |
                                       |  Faithfulness, Relevancy) |
                                       +---------------------------+
                                                     |
                                                     v
                                       +---------------------------+
                                       | Diagnostic Fault Engine   |
                                       | (Retriever / Tool / Gen)  |
                                       +---------------------------+
                                                     |
                                                     v
                                       +---------------------------+
                                       | GitHub Actions CI Gate    |
                                       | (Fails if delta > 0.05)   |
                                       +---------------------------+
```

## 2. Repository Layout

```
traceloop/
  .github/
    workflows/
      ci.yml
  app/
    __init__.py
    agent.py
    config.py
    main.py
    models.py
    qdrant_client.py
    tools.py
    static/
      app.js
      index.html
      style.css
  data/
    processed/
      chunks_metadata.jsonl
      promoted_traces.jsonl
    raw/
      sample_documents.jsonl
  docker/
    Dockerfile
    docker-compose.yml
    mlflow.Dockerfile
  evals/
    __init__.py
    baseline.json
    baseline_manager.py
    fault_classifier.py
    golden_dataset.json
    mlflow_logger.py
    ragas_evaluator.py
    run_evals.py
  ingestion/
    __init__.py
    chunker.py
    config.py
    embedder.py
    indexer.py
    loader.py
    run_ingest.py
  tests/
    __init__.py
    conftest.py
    test_agent.py
    test_api.py
    test_baseline_manager.py
    test_chunker.py
    test_config.py
    test_embedder.py
    test_eval_gate.py
    test_fault_classifier.py
    test_golden_dataset.py
    test_indexer.py
    test_loader.py
    test_qdrant_client.py
    test_ragas_evaluator.py
    test_sync.py
    test_tools.py
  tracing/
    __init__.py
    client.py
    sync_dataset.py
  .env.example
  .gitignore
  pyproject.toml
  README.md
  requirements.txt
  WALKTHROUGH.md
```

## 3. Quickstart: Docker Compose

Launch the entire stack (Qdrant, PostgreSQL, Langfuse, MLflow, and the TraceLoop application) with a single command:

```bash
docker compose -f docker/docker-compose.yml up -d --build
```

Service endpoints:
* Diagnostic Web UI and API: http://localhost:8000
* Langfuse Observability: http://localhost:3000
* MLflow Experiment Tracking: http://localhost:5000
* Qdrant Vector Engine: http://localhost:6333/dashboard

To shut down services and preserve persistent storage volumes:

```bash
docker compose -f docker/docker-compose.yml down
```

## 4. Local Development Setup

To run TraceLoop locally without Docker:

```bash
# 1. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy local environment defaults
cp .env.example .env
```

## 5. Ingestion Pipeline and Idempotency Verification

The ingestion pipeline loads technical documents from `data/raw/sample_documents.jsonl`, splits them into overlapping 500-character chunks, computes 384-dimensional unit-normalized embeddings via BGE, and indexes them into Qdrant.

Execute the ingestion pipeline:

```bash
python -m ingestion.run_ingest
```

### Idempotency Verification

TraceLoop generates point identifiers using UUIDv5 combined with a SHA-256 hash of `doc_id` and normalized chunk content. Running ingestion multiple times preserves identical collection point counts without generating duplicate records:

```bash
# First ingestion run
python -m ingestion.run_ingest

# Second ingestion run (asserts identical point count)
python -m ingestion.run_ingest
```

## 6. API Endpoint Specifications

Start the local development server:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 6.1 Query Execution: POST /ask

Executes user queries through the tool agent. Domain technical questions route to `knowledge_retriever_tool`, while conversational questions route to `fallback_direct_tool`.

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "How does agent memory consolidation bridge working memory and long-term stores?"}'
```

Response format:

```json
{
  "answer": "Regarding How does agent memory consolidation bridge working memory and long-term stores?, based on the indexed technical documentation [Chunk: doc_001_chunk_0000]: ...",
  "tool_used": "knowledge_retriever_tool",
  "retrieved_chunk_ids": [
    "doc_001_chunk_0000",
    "doc_001_chunk_0003"
  ],
  "latency_ms": 142.50,
  "trace_id": "673f5546-2b4a-4e89-a29d-9cf361ec1e2d",
  "trace_url": "http://localhost:3000/project/traceloop/traces/673f5546-2b4a-4e89-a29d-9cf361ec1e2d"
}
```

Conversational query example:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "Hello! What can you help me with?"}'
```

### 6.2 Health Check: GET /health

Returns service health and vector storage connectivity status:

```bash
curl -X GET http://localhost:8000/health
```

### 6.3 Telemetry Metrics: GET /metrics

Returns request volume counters and tool routing distributions:

```bash
curl -X GET http://localhost:8000/metrics
```

## 7. Evaluation Runner and MLflow Experiment Viewing

The evaluation harness evaluates agent performance across thirty-five golden benchmark cases or promoted production traces. It computes four core RAGAS metrics:
* Context Recall: Ratio of ground truth statements supported by retrieved chunks.
* Context Precision: Signal-to-noise ratio and rank order of relevant passages.
* Faithfulness: Ratio of claims in generated answers grounded in retrieved context.
* Answer Relevancy: Semantic alignment between user query and generated response.

Run evaluation in deterministic mock judge mode:

```bash
python -m evals.run_evals --dataset evals/golden_dataset.json --mock-judge
```

To update the baseline with verified metrics after an intentional architectural upgrade:

```bash
python -m evals.run_evals --dataset evals/golden_dataset.json --mock-judge --update-baseline
```

To view logged experiment runs and artifacts in the MLflow dashboard:
1. Open http://localhost:5000 in a web browser.
2. Select experiment `traceloop-evals`.
3. Compare run metrics, parameters, and artifact summaries.

## 8. Production Trace Synchronization

TraceLoop continuously mines failed or low-scoring production sessions from Langfuse and promotes them to regression test datasets.

Run trace synchronization:

```bash
python -m tracing.sync_dataset --threshold 0.5
```

The script extracts input queries, retrieved chunks, generated answers, and tool decisions, deduplicating records via SHA-256 query hashing before appending them to `data/processed/promoted_traces.jsonl`.

## 9. CI/CD Regression Gate and Verification Matrix

TraceLoop enforces an automated regression gate in GitHub Actions (`.github/workflows/ci.yml`). Pull requests run the evaluation harness against `evals/baseline.json`. If `delta_faithfulness > 0.05` or `delta_context_precision > 0.05`, the workflow exits with code 1 and blocks the pull request.

Run the local test matrix:

```bash
pytest -v
```
