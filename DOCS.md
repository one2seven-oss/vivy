# Vivy Documentation

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Creating an Index](#creating-an-index)
- [Inserting Vectors](#inserting-vectors)
- [Searching](#searching)
- [Distance Metrics](#distance-metrics)
- [Metadata Filtering](#metadata-filtering)
- [Edge Cases & Error Handling](#edge-cases--error-handling)
- [Performance Considerations](#performance-considerations)
- [Rust API](#rust-api)
- [FAQ](#faq)

---

## Installation

```sh
pip install maturin
cd py
maturin develop --release
```

Requires Python >= 3.8 and a Rust toolchain.

---

## Quick Start

```python
import vivy

idx = vivy.Index(4, "l2")

idx.insert([1.0, 0.0, 0.0, 0.0])
idx.insert([0.0, 1.0, 0.0, 0.0])
idx.insert([0.0, 0.0, 1.0, 0.0])

results = idx.search([1.0, 0.0, 0.0, 0.0], k=2)
print(results)

ids_with_distances = idx.search([0.5, 0.5, 0.0, 0.0], k=5)
for vector_id, distance in ids_with_distances:
    print(f"ID {vector_id}: distance {distance:.4f}")
```

---

## Creating an Index

```python
import vivy

idx = vivy.Index(dims=768, metric="cosine")
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `dims` | `int` | Dimensionality of vectors (must match all inserts/searches) |
| `metric` | `str` | Distance metric: `"l2"`, `"cosine"`, or `"dot"` |

### Metrics explained

| Metric | Formula | Best for |
|--------|---------|----------|
| `"l2"` | Σ(aᵢ − bᵢ)² | Raw Euclidean distance, word2vec, general purpose |
| `"cosine"` | 1 − cos(θ) | Text embeddings (OpenAI, BERT, SBERT) where direction matters |
| `"dot"` | −(a·b) | Unit-normalised vectors, maximum inner product search |

```python
l2_idx = vivy.Index(128, "l2")
cos_idx = vivy.Index(128, "cosine")
dot_idx = vivy.Index(128, "dot")
```

---

## Inserting Vectors

### Basic insert

```python
idx = vivy.Index(3, "l2")
vector_id = idx.insert([1.0, 2.0, 3.0])
print(vector_id)
```

IDs are auto-assigned starting from 1 and incrementing.

### Insert with metadata

```python
vector_id = idx.insert(
    [0.5, 0.2, 0.9],
    metadata={"color": "red", "category": "a"}
)
```

The `metadata` parameter is a `dict[str, str]`. Metadata is indexed in Roaring
bitmaps for fast filtering at search time.

---

## Searching

### Unfiltered search

```python
results = idx.search(query_vector, k=10)
```

Returns a list of `(id, distance)` tuples sorted by increasing distance
(closest first).

### Filtered search

```python
results = idx.search(
    [0.5, 0.3, 0.1],
    k=10,
    filter={"color": "red"}
)
```

Multiple filter keys are AND-combined:

```python
results = idx.search(
    [0.5, 0.3, 0.1],
    k=10,
    filter={"color": "red", "category": "a"}
)
```

### Example: Search with post-processing

```python
idx = vivy.Index(2, "l2")
for i in range(100):
    idx.insert([float(i), float(100 - i)], metadata={"id_str": str(i)})

results = idx.search([50.0, 50.0], k=10)

top_ids = [rid for rid, _ in results]
print("Top 10 IDs:", top_ids)

filtered = idx.search([50.0, 50.0], k=5, filter={"id_str": "25"})
print("Filtered:", filtered)
```

---

## Distance Metrics

### L2 (Squared Euclidean)

```python
idx = vivy.Index(3, "l2")
idx.insert([1.0, 0.0, 0.0])
idx.insert([10.0, 0.0, 0.0])

# Query identical to first vector → distance ≈ 0
results = idx.search([1.0, 0.0, 0.0], k=1)
assert abs(results[0][1]) < 1e-6
```

### Cosine

```python
idx = vivy.Index(2, "cosine")
idx.insert([1.0, 0.0])

# Orthogonal vectors → distance = 1.0
results = idx.search([0.0, 1.0], k=1)
assert abs(results[0][1] - 1.0) < 1e-5

# Identical vectors → distance = 0.0
results = idx.search([1.0, 0.0], k=1)
assert abs(results[0][1]) < 1e-6
```

### Dot

```python
idx = vivy.Index(3, "dot")
idx.insert([1.0, 0.0, 0.0])
idx.insert([2.0, 0.0, 0.0])

# Larger dot product → smaller distance (negated)
results = idx.search([1.0, 0.0, 0.0], k=2)
assert results[0][0] == 2  # vector with larger magnitude wins
```

### Choosing a metric

L2 and Cosine can produce different rankings. Cosine only cares about
direction; L2 considers both direction and magnitude:

```python
l2 = vivy.Index(2, "l2")
cos = vivy.Index(2, "cosine")

l2.insert([10.0, 0.0])   # id=1
l2.insert([1.0, 1.0])    # id=2
cos.insert([10.0, 0.0])  # id=1
cos.insert([1.0, 1.0])   # id=2

query = [2.0, 0.0]

# L2 prefers the vector closest in magnitude (id=2)
print("L2 top:", l2.search(query, k=2))

# Cosine prefers the vector with same direction (id=1)
print("Cosine top:", cos.search(query, k=2))
```

---

## Metadata Filtering

### Single field

```python
idx = vivy.Index(2, "l2")
for i in range(100):
    color = "red" if i % 2 == 0 else "blue"
    idx.insert([float(i), 0.0], metadata={"color": color})

# Only return vectors with color="red"
results = idx.search([50.0, 0.0], k=5, filter={"color": "red"})
```

### Multiple fields (AND)

```python
idx = vivy.Index(2, "l2")
for i in range(100):
    color = "red" if i % 2 == 0 else "blue"
    size = "large" if i >= 50 else "small"
    idx.insert([float(i), 0.0], metadata={"color": color, "size": size})

results = idx.search(
    [50.0, 0.0], k=5,
    filter={"color": "red", "size": "large"},
)
```

### Filter that matches nothing

```python
results = idx.search([0.0, 0.0], k=5, filter={"nonexistent": "value"})
assert results == []
```

---

## Edge Cases & Error Handling

### Empty index

```python
idx = vivy.Index(2, "l2")
assert len(idx) == 0
assert idx.search([0.0, 0.0], k=5) == []
```

### Dimension mismatch

```python
idx = vivy.Index(2, "l2")
idx.insert([1.0, 2.0])  # OK — 2-dim

try:
    idx.insert([1.0, 2.0, 3.0])  # 3-dim → ValueError
except ValueError:
    pass

try:
    idx.search([1.0, 2.0, 3.0], k=5)  # 3-dim query → ValueError
except ValueError:
    pass
```

### Zero k

```python
results = idx.search([1.0, 2.0], k=0)
assert results == []
```

### Invalid metric

```python
try:
    vivy.Index(3, "not_a_metric")
except ValueError:
    pass
```

### len() semantics

```python
idx = vivy.Index(2, "l2")
idx.insert([1.0, 2.0])
assert len(idx) == 1
idx.insert([3.0, 4.0])
assert len(idx) == 2
```

---

## Performance Considerations

### When Vivy is fast enough

| Scenario | Max vectors | Memory |
|----------|-------------|--------|
| HNSW + full-precision f32 | ~10M (768-dim) | ~30 GB RAM |
| HNSW + PQ (M=64) | ~100M (768-dim) | ~6.4 GB codes + graph on disk |

### Tips

- **Batch inserts** when possible — use `insert_batch` (Rust) or bulk-insert in
  a loop (Python). Batched inserts acquire the delta write lock once for the
  entire batch instead of once per vector.
- **Use cosine** for text embeddings (OpenAI, SBERT, etc.).
- **Use L2** when absolute distance matters (recommendation, geospatial).
- **Use metadata filters** sparingly on large datasets — they add bitmap
  intersection overhead.
- **Compaction** (delta → sealed segment) happens automatically when the delta
  exceeds 10K vectors. Searches fan out across the delta and all sealed
  segments, so many small segments are less efficient than few large ones.

### What Vivy is not

- Not a distributed database — single machine only.
- Not designed for real-time streaming updates at high velocity.
- Not a full-featured vector database — no built-in client-server protocol,
  no replication, no sharding.

---

## Rust API

Vivy is written in Rust and exposes its full API through the `vivy-core` crate.
The Python bindings cover the most common operations. For advanced use cases
(flat index, product quantization, WAL replay, segment I/O, custom filter
expressions), use Rust directly.

```toml
[dependencies]
vivy-core = { path = "lib" }
```

```rust
use vivy_core::concurrent::VivyIndex;
use vivy_core::distance::Metric;
use vivy_core::filter::FilterExpr;

let idx = VivyIndex::new(Metric::L2, Option::<&str>::None, Option::<&str>::None).unwrap();
idx.insert(vec![1.0, 0.0, 0.0]).unwrap();
let results = idx.search(&[1.0, 0.0, 0.0], 5);
```

Rust API features not available from Python:

| Feature | Rust Module |
|---------|-------------|
| Tokio async wrapper (`features = ["async"]`) | `vivy_core::async_index::AsyncVivyIndex` |
| Flat (brute-force) index | `vivy_core::flat::FlatIndex` |
| Product Quantization | `vivy_core::pq::ProductQuantizer` |
| Filter expression AST (Or, Not, In) | `vivy_core::filter::FilterExpr` |
| Write-Ahead Log | `vivy_core::storage::wal::WalWriter` |
| Sealed segment I/O | `vivy_core::storage::segments` |
| Batch insert | `VivyIndex::insert_batch(vectors)` |
| Non-blocking WAL (background fsync) | built into `VivyIndex` via `wal_path` |
| Pending-buffer batching (64 inserts per write lock) | built into `VivyIndex` |
| Snapshot-based search (no lock held during distance computation) | built into `VivyIndex` |
| Persistence + compaction | `VivyIndex::new(metric, wal_path, data_dir)` |

---

## FAQ

**Q: Why is cosine distance sometimes > 1?**  
A: Cosine distance = 1 − cos(θ). For opposite directions cos(θ) = −1, so the
distance is 2. Range is [0, 2].

**Q: Are IDs guaranteed to be sequential?**  
A: Yes, auto-assigned IDs start at 1 and increment atomically. No gaps unless
you use the Rust API for compaction or manual ID assignment.

**Q: Can I insert the same vector twice?**  
A: Yes, duplicates are allowed. They get different IDs.

**Q: How do I know if the index is empty?**  
A: Use `len(idx)` or check if `idx.search(...)` returns an empty list.

**Q: What happens when I search with a filter that doesn't match anything?**  
A: Empty results. No error.

**Q: Can I remove vectors?**  
A: Not from the Python API. The Rust WAL supports Delete entries, but the
Python bindings currently only expose insert/search.

**Q: Is Vivy thread-safe from Python?**  
A: Yes. The GIL is released during insert/search, so multiple Python threads
can operate on the same index concurrently. Inserts do not block searches:
WAL fsync runs on a background thread, writes to the delta are batched
(one write lock per 64 inserts), and search snapshots the delta under a
brief read lock with no lock held during distance computation.
