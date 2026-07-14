import vivy


def test_bulk_insert_and_search():
    idx = vivy.Index(2, "l2")
    for i in range(5000):
        idx.insert([float(i), float(5000 - i)])

    assert len(idx) == 5000
    results = idx.search([2500.0, 2500.0], k=10)
    assert len(results) == 10


def test_consecutive_searches():
    idx = vivy.Index(2, "l2")
    for i in range(1000):
        idx.insert([float(i), 0.0])

    for _ in range(100):
        results = idx.search([500.0, 0.0], k=5)
        assert len(results) == 5


def test_insert_returns_increasing_ids():
    idx = vivy.Index(2, "l2")
    ids = []
    for i in range(100):
        _id = idx.insert([float(i), 0.0])
        ids.append(_id)
    assert ids == list(range(1, 101))
