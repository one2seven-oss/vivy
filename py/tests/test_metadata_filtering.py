import vivy


def test_filter_by_single_field():
    idx = vivy.Index(2, "l2")
    for i in range(100):
        color = "red" if i % 2 == 0 else "blue"
        idx.insert([float(i), 0.0], metadata={"color": color})

    results = idx.search([50.0, 0.0], k=10, filter={"color": "red"})
    assert len(results) == 10
    for _id, _dist in results:
        assert _id % 2 == 1


def test_filter_by_multiple_fields_and():
    idx = vivy.Index(2, "l2")
    for i in range(100):
        color = "red" if i % 2 == 0 else "blue"
        size = "large" if i >= 50 else "small"
        idx.insert([float(i), 0.0], metadata={"color": color, "size": size})

    results = idx.search(
        [50.0, 0.0], k=5,
        filter={"color": "red", "size": "large"},
    )
    assert len(results) >= 1
    for _id, _dist in results:
        assert _id >= 51


def test_filter_no_matches():
    idx = vivy.Index(2, "l2")
    for i in range(50):
        idx.insert([float(i), 0.0], metadata={"group": "a"})

    results = idx.search([25.0, 0.0], k=5, filter={"group": "nonexistent"})
    assert len(results) == 0


def test_filter_all_match():
    idx = vivy.Index(2, "l2")
    for i in range(50):
        idx.insert([float(i), 0.0], metadata={"tag": "all"})

    results = idx.search([25.0, 0.0], k=5, filter={"tag": "all"})
    assert len(results) == 5


def test_filter_without_metadata_returns_nothing():
    idx = vivy.Index(2, "l2")
    for i in range(50):
        idx.insert([float(i), 0.0])

    results = idx.search([25.0, 0.0], k=5, filter={"color": "red"})
    assert len(results) == 0
