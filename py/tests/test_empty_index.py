import vivy


def test_search_empty_returns_empty():
    idx = vivy.Index(2, "l2")
    results = idx.search([1.0, 0.0], k=5)
    assert results == []


def test_len_of_empty_index():
    idx = vivy.Index(2, "l2")
    assert len(idx) == 0


def test_len_after_insert():
    idx = vivy.Index(2, "l2")
    idx.insert([1.0, 2.0])
    assert len(idx) == 1
    idx.insert([3.0, 4.0])
    assert len(idx) == 2
