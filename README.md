<p align="center">
  <img src="assets/vivy.png" alt="Vivy">
</p>

[![Crates.io](https://img.shields.io/crates/v/vivy-core?label=vivy-core)](https://crates.io/crates/vivy-core)
[![Crates.io Downloads](https://img.shields.io/crates/d/vivy-core)](https://crates.io/crates/vivy-core)
[![PyPI](https://img.shields.io/badge/pypi-vivy--vdb-blue)](https://pypi.org/project/vivy-vdb/)
[![docs.rs](https://img.shields.io/docsrs/vivy-core)](https://docs.rs/vivy-core)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.81%2B-orange)](https://www.rust-lang.org)
[![HNSW](https://img.shields.io/badge/index-HNSW-8A2BE2)](#)
[![WAL](https://img.shields.io/badge/crash--safe-WAL-blue)](#)
[![PyO3](https://img.shields.io/badge/bindings-PyO3-yellow)](#)
[![Roaring](https://img.shields.io/badge/filter-Roaring%20bitmaps-green)](#)

A single-machine vector search engine that fits in memory. No client-server, no
RPC, no proxies. Rust core with Python bindings.

> **[Full documentation → DOCS.md](DOCS.md)** covers API reference, Python
> examples, edge cases, performance tuning, and the Rust API.

## Why Vivy

Most vector databases are built for clusters. If your data fits on one machine
— a few million vectors, up to a few hundred dimensions — you don't need a
distributed system. A library that makes the hardware you already have go faster
is enough.

Vivy keeps everything in process. Search latency in microseconds, not
milliseconds. And it is enough for most real-world retrieval workloads before
you outgrow a single machine.

### What it does

- **HNSW graph** index with configurable M and ef_construction
- **Product quantization** for tight memory budgets
- **Metadata filtering** with Roaring bitmap indices (AND, OR, NOT, IN)
- **Crash-safe WAL** on a background fsync thread
- **Non-blocking inserts** — search never waits on a write lock
- **Python bindings** via PyO3

### Architecture

```
          Python           Rust
             │               │
    ┌────────┴──────┐  ┌─────┴──────┐
    │ PyO3 bindings │  │  VivyIndex │
    └────────┬──────┘  └──────┬─────┘
             │                │
    ┌────────┴────────────────┴──────────┐
    │  Pending buffer  (batch up to 64)  │
    ├────────────────────────────────────┤
    │  WAL worker     (background fsync) │
    ├────────────────────────────────────┤
    │  Delta (HNSW)   (batched writes)   │
    ├────────────────────────────────────┤
    │  Sealed Segment 1  (mmap)          │
    │  Sealed Segment 2  (mmap)          │
    │  ...                               │
    └────────────────────────────────────┘
```

## Getting started

```python
import vivy

idx = vivy.Index(dims=768, metric="cosine")
idx.insert([0.1] * 768, metadata={"color": "red"})
idx.insert([0.9] * 768, metadata={"color": "blue"})

results = idx.search([0.5] * 768, k=5)
results_with_filter = idx.search([0.5] * 768, k=5, filter={"color": "red"})
```

Requires Python >= 3.8 and a Rust toolchain.

## Build

```sh
cargo build --release
```

Python bindings:

```sh
pip install maturin && cd py && maturin develop --release
```

### Build manually

```sh
git clone https://github.com/your-org/vivy
cd vivy
cargo build --release
cd py && maturin develop --release
```

Run the benchmark:

```sh
cargo run --release --bin vivy-bench
```

Reports recall@10 vs brute force, QPS, and mean latency on random 64-dim data.

## Comparison vs alternatives

Single-machine vector search libraries, benchmarked on SIFT1M-scale (1M vectors,
128D, L2). Numbers from [RetriEval](https://github.com/ashvardanian/RetriEval)
and [ANN-Benchmarks](https://ann-benchmarks.com) — open-source reproducible
suites.

| Engine | Language | HNSW params | Recall@10 | Search QPS | Persistence | Metadata filter | SIMD |
|---|---|---|---|---|---|---|---|
| **FAISS** | C++ | M=16, ef=128 | 0.995 | ~38,000 | — | — | AVX2/512, NEON |
| **USearch** | C (11+ bindings) | M=16, ef=128 | 0.994 | ~80,000 | save/load | — | SimSIMD (all archs) |
| **hnswlib** | C++ | M=16, ef=200 | 0.996 | ~20,000 | — | — | scalar |
| **Vivy** | Rust | M=16, ef=200 | TBD | TBD | WAL + mmap | Roaring bitmaps | scalar (autovec) |

Vivy at M=16 / ef_construction=200 matches the same HNSW algorithm as the
others. The gap at scale is the distance kernel — USearch's SIMD gives 2–4×
more QPS at iso-recall. Vivy's differentiators are crash-safe persistence and
metadata filtering, neither of which the others ship in a library form.

Open a SIFT1M-scale benchmark issue if that matters for your use case — we
will run one and publish numbers.

## Status

Vivy is in active development. The API is stabilising but may see changes
before 1.0. Sealed-segment compaction, WAL replay, and the Python bindings
are exercised in tests but have not yet seen broad production use.

---

*by the dev, for the dev, and of the dev*
