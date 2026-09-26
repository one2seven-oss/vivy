import pytest
import numpy as np
import vivy
import tempfile
import time

def test_index_insert_and_search_numpy():
    idx = vivy.Index(3, "l2")
    vec1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    vec2 = np.array([4.0, 5.0, 6.0], dtype=np.float32)

    id1 = idx.insert(vec1)
    id2 = idx.insert(vec2)

    query = np.array([1.1, 2.0, 3.0], dtype=np.float32)
    results = idx.search(query, k=2)

    assert len(results) == 2
    assert results[0][0] == id1
    assert results[1][0] == id2

def test_identical_recall_list_vs_numpy():
    idx_list = vivy.Index(128, "cosine")
    idx_np = vivy.Index(128, "cosine")

    np.random.seed(42)
    data = np.random.randn(50, 128).astype(np.float32)

    for row in data:
        idx_list.insert(row.tolist())
        idx_np.insert(row)

    query_data = np.random.randn(128).astype(np.float32)

    results_list = idx_list.search(query_data.tolist(), k=5)
    results_np = idx_np.search(query_data, k=5)

    assert results_list == results_np

def test_numpy_error_handling():
    idx = vivy.Index(3, "l2")

    # float64 array should fail with PyTypeError
    vec_f64 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    with pytest.raises(TypeError):
        idx.insert(vec_f64)

    with pytest.raises(TypeError):
        idx.search(vec_f64, k=1)

    # 2D array passed to 1D insert should fail
    vec_2d = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    with pytest.raises(TypeError):
        idx.insert(vec_2d)

    with pytest.raises(TypeError):
        idx.search(vec_2d, k=1)

    # Int array should fail
    vec_int = np.array([1, 2, 3], dtype=np.int32)
    with pytest.raises(TypeError):
        idx.insert(vec_int)

    # Non-contiguous slice should fail with ValueError
    full_arr = np.array([1.0, 0.0, 2.0, 0.0, 3.0, 0.0], dtype=np.float32)
    strided_arr = full_arr[::2]
    assert not strided_arr.flags['C_CONTIGUOUS']
    with pytest.raises(ValueError):
        idx.insert(strided_arr)

def test_index_insert_batch_numpy_2d():
    idx = vivy.Index(4, "l2")
    data = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ], dtype=np.float32)

    ids = idx.insert_batch(data)
    assert len(ids) == 3

    query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    results = idx.search(query, k=1)
    assert results[0][0] == ids[0]

def test_memory_store_numpy():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = vivy.MemoryStore.open(tmpdir, dimensions=4, embedding_model="test-model")
        
        emb1 = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        emb2 = np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float32)

        id1 = store.remember(
            tenant_id="t1",
            namespace="n1",
            content="first numpy memory",
            embedding=emb1,
            kind="fact"
        )
        assert id1 is not None

        # Batch remember with numpy embeddings
        records = [
            {
                "tenant_id": "t1",
                "namespace": "n1",
                "content": "second numpy memory",
                "embedding": emb2,
                "kind": "preference"
            }
        ]
        ids = store.remember_batch(records)
        assert len(ids) == 1

        # Recall with numpy query
        query_np = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        recalled = store.recall(
            tenant_id="t1",
            namespace="n1",
            query_embedding=query_np,
            limit=2
        )
        assert len(recalled) == 2
        assert recalled[0][1] == "first numpy memory"

def test_microbenchmark_search_performance():
    dims = 1536
    idx = vivy.Index(dims, "cosine")

    # Insert 100 vectors
    np.random.seed(42)
    vectors = np.random.randn(100, dims).astype(np.float32)
    for v in vectors:
        idx.insert(v)

    query_list = [0.1] * dims
    query_np = np.full(dims, 0.1, dtype=np.float32)

    iterations = 5000

    start_list = time.perf_counter()
    for _ in range(iterations):
        idx.search(query_list, k=5)
    duration_list = time.perf_counter() - start_list

    start_np = time.perf_counter()
    for _ in range(iterations):
        idx.search(query_np, k=5)
    duration_np = time.perf_counter() - start_np

    print(f"\n[Microbenchmark] {iterations} queries (dim={dims}):")
    print(f"  Python list duration: {duration_list:.4f}s ({duration_list/iterations*1e6:.2f} µs/op)")
    print(f"  NumPy zero-copy duration: {duration_np:.4f}s ({duration_np/iterations*1e6:.2f} µs/op)")
    print(f"  Speedup: {duration_list / duration_np:.2f}x")

    # NumPy search should be faster than Python list search
    assert duration_np < duration_list
