import math

import pytest

from AgentBasedModel.utils.pools import ConstantProductPool, HFPool


@pytest.mark.parametrize("pool_cls,pkwargs", [
    (ConstantProductPool, {}),
    (HFPool, {"A": 100.0, "gamma": 1e-3}),
])
def test_invariant_preserved_apply_out_in(pool_cls, pkwargs):
    pool = pool_cls(100.0, 100.0, **pkwargs)
    k_before = pool.x * pool.y
    dy = pool.apply_out(10.0)
    assert dy > 0
    assert pool.x > 0 and pool.y > 0
    k_after = pool.x * pool.y
    # Инвариант должен оставаться на том же порядке величины
    assert math.isfinite(k_after)
    assert k_after > 0
    assert abs(k_after - k_before) / k_before < 0.1

    dx = pool.apply_in(dy / 2)
    assert dx >= 0
    assert pool.x > 0 and pool.y > 0


def test_constant_product_symmetry_small_trade():
    pool = ConstantProductPool(100.0, 100.0)
    dx = 1.0
    dy = pool.quote_out(dx)
    dx_back = pool.quote_in(dy)
    assert dy > 0
    assert dx_back > 0
    # Малые торги должны быть почти обратимы
    assert abs(dx_back - dx) / dx < 0.05


def test_hfpool_monotone_pricing_and_no_negative():
    pool = HFPool(100.0, 100.0, A=100.0, gamma=1e-3)
    prices = []
    for dx in (1.0, 5.0, 10.0, 20.0):
        dy = pool.quote_out(dx)
        price = dy / dx if dx > 0 else float("inf")
        prices.append(price)
        assert dy >= 0
        assert dy < pool.y  # не должен отдавать больше резерва
    # Цена должна снижаться с ростом dx (просокальзывание монотонно)
    assert all(prices[i] >= prices[i + 1] for i in range(len(prices) - 1))


def test_curve_points_monotone_and_positive():
    pool = HFPool(100.0, 100.0, A=100.0, gamma=1e-3)
    xs, ys = pool.curve_points(n=50)
    assert len(xs) == len(ys) == 50
    # x растёт, y должен убывать и быть положительным
    assert all(xs[i] < xs[i + 1] for i in range(len(xs) - 1))
    assert all(ys[i] > 0 for i in range(len(ys)))
    assert all(ys[i] >= ys[i + 1] for i in range(len(ys) - 1))
