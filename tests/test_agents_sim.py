import random

import pytest

from AgentBasedModel.agents.agents import ExchangeAgent, AutoMarketMaker, HFMarketMaker, MarketMaker
from AgentBasedModel.utils.orders import Order


def make_empty_exchange(price=100):
    ex = ExchangeAgent(price=price, std=0, volume=0)
    ex.order_book = {"bid": ex.order_book["bid"].__class__("bid"), "ask": ex.order_book["ask"].__class__("ask")}
    return ex


def make_simple_book(exchange, bid_price=95, ask_price=105, qty=100):
    bid = Order(bid_price, qty, "bid", None)
    ask = Order(ask_price, qty, "ask", None)
    exchange.order_book["bid"].push(bid)
    exchange.order_book["ask"].append(ask)


def test_market_maker_on_empty_book_no_orders():
    ex = make_empty_exchange()
    mm = MarketMaker(ex, cash=1000, assets=10)
    mm.call()
    assert mm.orders == []
    # нет цены -> spread None
    assert ex.spread() is None


@pytest.mark.parametrize("agent_cls,kwargs", [
    (AutoMarketMaker, {"initial_ratio": 0.5}),
    (HFMarketMaker, {"A": 50.0, "gamma": 1e-3}),
])
def test_amms_place_limits_and_trade_on_simple_book(agent_cls, kwargs):
    random.seed(42)
    if agent_cls is HFMarketMaker:
        ex = make_empty_exchange(price=1)
        make_simple_book(ex, bid_price=0.95, ask_price=1.05, qty=200)
        cash, assets = 5000, 5000
    else:
        ex = make_empty_exchange()
        make_simple_book(ex, bid_price=95, ask_price=105, qty=200)
        cash, assets = 5000, 200
    agent = agent_cls(ex, cash=cash, assets=assets, **kwargs)
    if hasattr(agent, "cash_to_buy"):
        agent.cash_to_buy = 0  # отключаем начальную балансировку, чтобы дойти до сетки
    # отключаем рыночные сделки, чтобы книга не опустела
    agent.MAX_TRADE_FRACTION = 0.0
    agent.call()
    # Агент должен выставить хотя бы одну лимитку
    assert len(agent.orders) > 0
    # Все цены лимиток неотрицательные
    assert all(o.price >= 0 for o in agent.orders)
    # Кол-во ордеров ограничено классом
    max_orders = agent.MAX_LIMIT_BUY + agent.MAX_LIMIT_SELL
    assert len(agent.orders) <= max_orders


@pytest.mark.parametrize("agent_cls,kwargs", [
    (AutoMarketMaker, {"initial_ratio": 0.5}),
    (HFMarketMaker, {"A": 50.0, "gamma": 1e-3}),
])
def test_amms_price_grid_monotone(agent_cls, kwargs):
    random.seed(123)
    ex = make_empty_exchange()
    # Добавим книгу, чтобы call() дошёл до построения сетки
    make_simple_book(ex, bid_price=95, ask_price=105, qty=50)
    agent = agent_cls(ex, cash=2000, assets=100, **kwargs)
    agent.call()
    bids = [o.price for o in agent.orders if o.order_type == "bid"]
    asks = [o.price for o in agent.orders if o.order_type == "ask"]
    if bids:
        # цены bid должны быть упорядочены невозрастанию (лучшие выше)
        assert all(bids[i] >= bids[i + 1] for i in range(len(bids) - 1))
    if asks:
        # цены ask неубывающие
        assert all(asks[i] <= asks[i + 1] for i in range(len(asks) - 1))
