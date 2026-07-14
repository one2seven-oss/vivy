import vivy


def test_high_dimensional_vectors():
    dims = 128
    idx = vivy.Index(dims, "l2")
    for i in range(100):
        v = [float(i + j) * 0.01 for j in range(dims)]
        idx.insert(v)

    query = [0.5 for _ in range(dims)]
    results = idx.search(query, k=5)
    assert len(results) == 5


def test_single_dimension():
    idx = vivy.Index(1, "l2")
    idx.insert([0.0])
    idx.insert([10.0])
    idx.insert([100.0])

    results = idx.search([5.0], k=2)
    assert len(results) == 2
    assert results[0][0] == 1


def test_large_dimensionality_basic():
    dims = 256
    idx = vivy.Index(dims, "cosine")
    v = [1.0 / (i + 1) for i in range(dims)]
    idx.insert(v)
    results = idx.search(v, k=1)
    assert results[0][0] == 1
    assert abs(results[0][1]) < 1e-5
