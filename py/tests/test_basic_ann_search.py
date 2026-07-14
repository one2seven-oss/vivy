import vivy


def test_insert_and_search():
    idx = vivy.Index(3, "l2")
    idx.insert([1.0, 0.0, 0.0])
    idx.insert([0.0, 1.0, 0.0])
    idx.insert([0.0, 0.0, 1.0])

    results = idx.search([1.0, 0.0, 0.0], k=2)
    assert len(results) == 2
    assert results[0][0] == 1


def test_nearest_neighbor_is_closest():
    idx = vivy.Index(2, "l2")
    for i in range(100):
        idx.insert([float(i), 0.0])

    results = idx.search([50.0, 0.0], k=1)
    assert len(results) == 1
    assert results[0][1] < 1.0


def test_results_sorted_by_distance():
    idx = vivy.Index(2, "l2")
    for i in range(50):
        idx.insert([float(i), float(50 - i)])

    results = idx.search([25.0, 25.0], k=10)
    assert len(results) == 10
    for i in range(len(results) - 1):
        assert results[i][1] <= results[i + 1][1]


def test_returns_correct_number_of_results():
    idx = vivy.Index(2, "l2")
    for i in range(20):
        idx.insert([float(i), 0.0])

    assert len(idx.search([0.0, 0.0], k=5)) == 5
    assert len(idx.search([0.0, 0.0], k=100)) == 20
