import pytest

from daindex.core import db_ineq


def test_db_ineq_positive_difference():
    c1_di = {"k-step": 0.5}
    c2_di = {"k-step": 0.6}
    ineq = db_ineq(c1_di, c2_di)
    assert ineq == 0.1


def test_db_ineq_negative_difference():
    c1_di = {"k-step": 0.6}
    c2_di = {"k-step": 0.5}
    ineq = db_ineq(c1_di, c2_di)
    assert ineq == -0.1


def test_db_ineq_zero_difference():
    c1_di = {"k-step": 0.5}
    c2_di = {"k-step": 0.5}
    ineq = db_ineq(c1_di, c2_di)
    assert ineq == 0.0


def test_db_ineq_missing_key():
    c1_di = {"k-step": 0.5}
    c2_di = {}
    with pytest.raises(KeyError):
        db_ineq(c1_di, c2_di)


def test_db_ineq_non_numeric_value():
    c1_di = {"k-step": 0.5}
    c2_di = {"k-step": "a"}
    with pytest.raises(TypeError):
        db_ineq(c1_di, c2_di)
