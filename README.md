<img src="assets/vivy.png" alt="Vivy" width="200">

[![CI](https://github.com/anomalyco/vivy/actions/workflows/ci.yml/badge.svg)](https://github.com/anomalyco/vivy/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/vivy-core?label=vivy-core)](https://crates.io/crates/vivy-core)
[![PyPI](https://img.shields.io/badge/pypi-vivy--vdb-blue)](https://pypi.org/project/vivy-vdb/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.81%2B-orange)](https://www.rust-lang.org)

Single-machine vector search that fits in memory.

HNSW graph index with optional product quantization, metadata filtering,
and crash-safe WAL. Spun from a Python REPL or embedded in Rust.

> **[Full documentation → DOCS.md](DOCS.md)** — detailed API reference, examples,
> edge cases, performance tuning, and Rust API.

## Quick start

```python
import vivy

idx = vivy.Index(dims=768, metric="cosine")
idx.insert([0.1] * 768, metadata={"color": "red"})
idx.insert([0.9] * 768, metadata={"color": "blue"})

results = idx.search([0.5] * 768, k=5)
results_with_filter = idx.search([0.5] * 768, k=5, filter={"color": "red"})
```

Requires Python ≥3.8. Build with `maturin` (see below).

## Why Vivy

Most vector databases are built for clusters. If your data fits on one
machine — a few million vectors, a few hundred dimensions — you don't need
a distributed system, you need a library that makes the hardware you already
have go faster.

Vivy keeps everything in process. No client-server, no RPC, no proxies.
Search latency is measured in microseconds, not milliseconds.

## Architecture

```
          Python           Rust 
             │               │
    ┌────────┴──────┐  ┌─────┴──────┐
    │ PyO3 bindings │  │  VivyIndex │
    └────────┬──────┘  └──────┬─────┘
             │                │
    ┌────────┴────────────────┴─────┐
    │         Delta (HNSW)          │  ← mutable, write-locked
    ├───────────────────────────────┤
    │  Sealed Segment 1  (mmap)     │  ← immutable
    │  Sealed Segment 2  (mmap)     │
    │  ...                          │
    └───────────────────────────────┘
```

- **Delta segment**: HNSW graph in memory. New inserts go here. Small (~10K
  vectors before compaction), so writes stay fast and contention stays low.
- **Sealed segments**: Immutable files on disk, memory-mapped. No locks.
  Scanned linearly (small enough that it doesn't matter — see *When Vivy is
  fast enough* below).
- **Compactor**: Background thread that freezes the delta when it overflows,
  writes it to a sealed segment, and atomically swaps it into the search path
  via `arc_swap`. Polls every 5 seconds.
- **Filter index**: Roaring bitmap per (field, value) pair. Filters are
  pushed into the HNSW search when selective; otherwise applied as a
  post-filter.

## Indexes

| Index | When to use |
|-------|-------------|
| `HNSW` (default) | General purpose. M=16, ef_construction=200 — >95% recall@10 on SIFT1M. |
| `Flat` | Brute force. Used internally as a correctness oracle and for ground truth in benchmarks. |
| `PQ` | Tiny memory budget. 64-byte codes instead of 3072 for 768-dim f32. ADC for distance computation. |

PQ encoding runs during compaction. The HNSW graph itself stays on disk
(mmap); only the quantized codes live in memory at query time.

## When Vivy is fast enough

- **≤10M vectors** (768-dim f32, ~30 GB RAM) with HNSW + full-precision vectors
- **≤100M vectors** with PQ (M=64, ~6.4 GB for codes, graph on disk via mmap)
- **≤10K vectors per sealed segment** — linear scan at p99 is a few hundred µs;
  cross-segment merge adds a constant factor

Past this point you probably want a distributed system.

## Build

```sh
cargo build --release
```

Python:

```sh
pip install maturin && cd py && maturin develop --release
```

## Benchmarks

```sh
cargo run --release --bin vivy-bench
```

Reports recall@10 vs brute force, QPS, and mean latency on random 64-dim
data. Run it unoptimised first — the gap between debug and release builds is
substantial.

