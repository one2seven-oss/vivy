import vivy


def test_query_dimension_mismatch():
    idx = vivy.Index(2, "l2")
    idx.insert([1.0, 2.0])
    try:
        idx.search([1.0, 2.0, 3.0], k=5)
    except ValueError:
        pass


def test_insert_wrong_dimension():
    idx = vivy.Index(2, "l2")
    try:
        idx.insert([1.0, 2.0, 3.0])
    except ValueError:
        pass


def test_zero_k_returns_empty():
    idx = vivy.Index(2, "l2")
    idx.insert([1.0, 2.0])
    results = idx.search([1.0, 2.0], k=0)
    assert len(results) == 0
