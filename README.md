<p align="center">
  <img src="assets/vivy.png" alt="Vivy">
</p>

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
    ┌────────┴────────────────┴──────────┐
    │  Pending buffer  (batch up to 64)  │  ← Mutex, never blocks writers
    ├────────────────────────────────────┤
    │  WAL worker     (background fsync) │  ← bounded channel, non-blocking
    ├────────────────────────────────────┤
    │  Delta (HNSW)  (batched writes)    │  ← RwLock, 1 lock per 64 inserts
    ├────────────────────────────────────┤
    │  Sealed Segment 1  (mmap)          │  ← immutable
    │  Sealed Segment 2  (mmap)          │
    │  ...                               │
    └────────────────────────────────────┘
```

- **Pending buffer**: New inserts land here under a lightweight `Mutex`, then
  flush to the delta in batches of 64. A write lock is acquired once per batch
  instead of once per insert — inserts become essentially lock-free from the
  search path's perspective.
- **WAL worker**: Append+fsync runs in a dedicated thread via a bounded
  `crossbeam_channel`. The insert path sends entries with `try_send` and never
  stalls on disk I/O.
- **Delta segment**: HNSW graph in memory. Receives batched writes from the
  pending buffer. Small (~10K vectors before compaction).
- **Sealed segments**: Immutable files on disk, memory-mapped. No locks.
  Scanned linearly.
- **Search**: Snapshots delta entries under a brief read lock, releases it,
  then searches the snapshot + sealed segments. No lock held during distance
  computation.
- **Compactor**: Background thread that flushes the pending buffer, freezes
  the delta on overflow, writes a sealed segment, and atomically swaps it
  into the search path via `arc_swap`. Event-driven — instant wake-up when
  the delta exceeds threshold, with a 30-second safety timeout.
- **Filter index**: Roaring bitmap per (field, value) pair. Filters are
  applied as a post-filter on the delta snapshot and sealed segment scan.

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
