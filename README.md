<div align="center">
  <img src="assets/vivy.png" alt="Vivy" width="200" />
  <br><br>

  [![Crates.io](https://img.shields.io/crates/v/vivy-core?label=vivy-core)](https://crates.io/crates/vivy-core)
  [![PyPI](https://img.shields.io/badge/pypi-vivy--vdb-blue)](https://pypi.org/project/vivy-vdb/)
  [![License](https://img.shields.io/badge/license-BSL--1.1-green)](LICENSE)
  [![Rust](https://img.shields.io/badge/rust-1.81%2B-orange)](https://www.rust-lang.org)
  [![HNSW](https://img.shields.io/badge/index-HNSW-8A2BE2)](#)
  [![WAL](https://img.shields.io/badge/crash--safe-WAL-blue)](#)
  [![FTS5](https://img.shields.io/badge/hybrid-FTS5%20%2B%20Vector%20RRF-blue)](#)
  [![PyO3](https://img.shields.io/badge/bindings-PyO3-yellow)](#)
  [![Release Candidate](https://img.shields.io/badge/status-v0.1.0--RC1%20Ready-brightgreen)](#)
</div>

A local, durable long-term memory (LTM) runtime and single-machine vector engine for AI agents. No cloud databases, no network latency, no external daemons. Rust core with PyO3 Python bindings.

---

## Workspace Packages

| Crate / Package | Path | Responsibility |
| :--- | :--- | :--- |
| **`vivy-memory`** | [`ltm/`](ltm) | Durable LTM runtime: multi-tenant isolation, SQLite WAL canonical store, operation journal, hybrid recall (FTS5 + HNSW vector RRF), explainable scoring & MMR diversity. |
| **`vivy-core`** | [`vec/`](vec) | In-process vector engine: HNSW graph search, Roaring bitmap filters, 64-bit ID safety, atomic segment manifests & WAL. |
| **`vivy-py`** | [`py/`](py) | PyO3 Python bindings exposing `vivy.MemoryStore` and low-level `vivy.Index` with GIL release. |
| **`bench`** | [`sim/`](sim) | Synthetic data benchmark suite for QPS, latency, and ground-truth recall verification. |

---

## Benchmark Performance Matrix

Measured on single-machine benchmark suite (`cargo run --release --package bench`):

| Metric / Scenario | Measured Value | Description |
| :--- | :--- | :--- |
| **Cold Start Latency** | `~51 ms` | Time to initialize fresh database, SQLite WAL, & vector engine |
| **Warm Start Latency** | `~51 ms` | Time to recover startup & replay operation journal (500 records) |
| **Write Throughput** | `5,216 writes/sec` | Dual-write durability (SQLite WAL commit + HNSW graph update) |
| **Hybrid Recall Throughput** | `305.1 QPS` | Dense Vector KNN + SQLite FTS5 + RRF Fusion + MMR Rerank |
| **Hybrid Recall Latency** | `3.27 ms` | End-to-end mean search & reranking latency |
| **Vector Engine (64-dim)** | `6,012 QPS` | Pure HNSW vector search throughput (`0.16 ms` mean latency) |
| **Vector Engine (512-dim)** | `1,433 QPS` | Pure HNSW vector search throughput (`0.69 ms` mean latency) |
| **Tombstone Vacuum Speed** | `74 ms` | Resumable physical tombstone scrubbing (100 rows batch) |

---

## Key Capabilities

* **Durable Agent Memory (`vivy-memory`)**: SQLite WAL serves as canonical truth for memory records, revisions, and operation journal. Vector index acts as a derived, auto-rebuildable accelerator.
* **Strict Multi-Tenant Isolation**: Enforces tenant, namespace, agent, and user boundaries at API entry and SQL level. Zero cross-tenant data leakage.
* **Hybrid Candidate Recall**: Combines SQLite FTS5 lexical keyword matching with dense HNSW vector search using Reciprocal Rank Fusion (RRF).
* **Transparent Reranking & MMR**: 4-component weighted scoring (Similarity, Importance, Recency, Reinforcement) plus optional Maximal Marginal Relevance (MMR) deduplication.
* **Security & Operations Primitives**: Encrypted storage interfaces (`KeyProvider`), telemetry redaction (`TelemetryRecord`), non-blocking health checks (`StoreHealth`), and resumable vacuuming (`vacuum_tombstones`).
* **High Performance Vector Search (`vivy-core`)**: HNSW vector graph with non-blocking inserts and Roaring bitmap metadata filtering.

---

## Architecture

```text
+-----------------------------------------------------------------+
| Caller / Python Agent / LLM Application                         |
+-----------------------------------------------------------------+
                                |
                                v
+-----------------------------------------------------------------+
| vivy-memory (LTM Runtime Layer)                                 |
| - MemoryScope (tenant_id, namespace, agent_id, user_id)         |
| - Operation Journal (idempotency, 2-phase state machine)        |
| - Hybrid Reciprocal Rank Fusion (SQLite FTS5 + HNSW Vector RRF) |
| - Explainable Reranker & MMR Diversity Pass                     |
+-----------------------------------------------------------------+
           |                                           |
           v                                           v
+-----------------------------------+   +-------------------------+
| SQLite (Canonical Store / WAL)    |   | vivy-core (Vector ANN)  |
| - memories (Content, Provenance)  |   | - Rebuildable HNSW      |
| - memories_fts (FTS5 Lexical)     |   | - Fast Cosine Retrieval |
| - operations (Journal state)      |   | - Derived Accelerators  |
+-----------------------------------+   +-------------------------+
```

---

## Getting Started (Python)

### Agent Long-Term Memory (`MemoryStore`)

```python
import vivy

# Open or create a local durable memory store
store = vivy.MemoryStore.open(
    path="./agent_memory",
    dimensions=1536,
    embedding_model="text-embedding-3-small"
)

# Remember an observation with an operation ID for idempotency
mem_id = store.remember(
    tenant_id="acme",
    namespace="support",
    content="User x prefers concise technical answers with benchmarks.",
    embedding=[0.1] * 1536,
    kind="preference",
    importance=0.9,
    operation_id="op-pref-101"
)

# Hybrid Recall (FTS5 text search + Vector embedding RRF fusion)
results = store.recall(
    tenant_id="acme",
    namespace="support",
    query_embedding=[0.1] * 1536,
    query_text="x benchmarks",
    limit=5,
    include_explanations=True,
    mmr_lambda=0.5  # Apply MMR diversity
)

for memory_id, content, score in results:
    print(f"[{score:.3f}] {content}")

# Soft-delete / tombstone a memory
store.forget(tenant_id="acme", namespace="support", id=mem_id)

# Physical maintenance & health check
health = store.health()
print(f"Store active: {health['total_active_records']}, tombstones: {health['total_tombstoned_records']}")
purged = store.vacuum_tombstones(batch_size=100)
```

### Low-Level Vector Search (`Index`)

```python
import vivy

idx = vivy.Index(dims=768, metric="cosine")
idx.insert([0.1] * 768, metadata={"color": "red"})
idx.insert([0.9] * 768, metadata={"color": "blue"})

results = idx.search([0.5] * 768, k=5, filter={"color": "red"})
```

---

## Build & Verification

```sh
# Run workspace test suite (48 integration/unit tests)
cargo test --workspace

# Run zero-warning clippy check
cargo clippy --workspace --all-targets -- -D warnings

# Build release binaries
cargo build --release

# Run performance benchmark suite
cargo run --release --package bench
```

---

## Licensing & Commercial Terms

**Vivy** components (`vivy-core`, `vivy-memory`, `vivy-py`, `bench`) are published under **The Business Source License 1.1 (BSL-1.1)** (see [`LICENSE`](LICENSE)).

### Permitted Uses (BSL 1.1 Additional Use Grant)
- **Non-Production & Evaluation**: Free use for development, testing, research, and evaluation.
- **Production Workloads**: Free use in production for non-commercial applications, single-node deployments, and internal AI agent workloads, provided Vivy is not offered as a managed SaaS or cloud API vector service to third parties.

### Commercial Pro / Enterprise Licensing
For managed cloud service providers or enterprise deployments requiring custom SLAs:
- Cloud cluster synchronization & distributed multi-region replication.
- Hardware Security Module (HSM) & AWS KMS / GCP KMS `KeyProvider` integration.
- Role-Based Access Control (RBAC) policy enforcement engine.

---
