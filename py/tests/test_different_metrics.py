import vivy
import math


def test_l2_identical_is_zero():
    idx = vivy.Index(3, "l2")
    idx.insert([1.0, 2.0, 3.0])
    results = idx.search([1.0, 2.0, 3.0], k=1)
    assert abs(results[0][1]) < 1e-6


def test_cosine_identical_is_zero():
    idx = vivy.Index(3, "cosine")
    idx.insert([1.0, 2.0, 3.0])
    results = idx.search([1.0, 2.0, 3.0], k=1)
    assert abs(results[0][1]) < 1e-6


def test_cosine_orthogonal_is_one():
    idx = vivy.Index(2, "cosine")
    idx.insert([1.0, 0.0])
    results = idx.search([0.0, 1.0], k=1)
    assert abs(results[0][1] - 1.0) < 1e-5


def test_cosine_vs_l2_ranking_differs():
    l2 = vivy.Index(2, "l2")
    cos = vivy.Index(2, "cosine")

    vecs = [[10.0, 0.0], [1.0, 1.0]]
    for v in vecs:
        l2.insert(v)
        cos.insert(v)

    query = [2.0, 0.0]
    l2_top = l2.search(query, k=2)
    cos_top = cos.search(query, k=2)

    assert l2_top[0][0] != cos_top[0][0]


def test_dot_product():
    idx = vivy.Index(3, "dot")
    idx.insert([1.0, 0.0, 0.0])
    idx.insert([2.0, 0.0, 0.0])
    results = idx.search([1.0, 0.0, 0.0], k=2)
    assert len(results) == 2
    assert results[0][0] == 2


def test_invalid_metric_raises():
    try:
        vivy.Index(3, "invalid_metric")
        assert False, "should have raised"
    except ValueError:
        pass
