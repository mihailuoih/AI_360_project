from AgentBasedModel.simulator import Simulator
from AgentBasedModel.agents import Trader, Universalist, Fundamentalist, MarketMaker, AutoMarketMaker, Chartist, Random
from AgentBasedModel.utils.orders import Order
from itertools import chain
import random
import math


class Event:
    def __init__(self, it: int):
        self.it = it  # Activation it
        self.simulator = None

    def __repr__(self):
        return f'empty (it={self.it})'

    def call(self, it: int):
        if self.simulator is None:
            raise Exception('No simulator link found')
        if it != self.it:
            return True
        return False

    def link(self, simulator: Simulator):
        self.simulator = simulator
        return self


class FundamentalPriceShock(Event):
    def __init__(self, it: int, price_change: float):
        super().__init__(it)
        self.dp = price_change

    def __repr__(self):
        return f'fundamental price shock (it={self.it}, dp={self.dp})'

    def call(self, it: int):
        if super().call(it):
            return
        divs = self.simulator.exchange.dividend_book  # link to dividend book
        r = self.simulator.exchange.risk_free  # risk-free rate

        self.simulator.exchange.dividend_book = [div + self.dp * r for div in divs]


class FundamentalPriceShockRelative(Event):
    def __init__(self, it: int, fraction: float):
        super().__init__(it)
        self.fraction = fraction

    def __repr__(self):
        pct = round(self.fraction * 100, 2)
        return f'fundamental price shock relative (it={self.it}, pct={pct}%)'

    def call(self, it: int):
        if super().call(it):
            return
        multiplier = max(0.0, 1.0 + self.fraction)
        divs = self.simulator.exchange.dividend_book
        self.simulator.exchange.dividend_book = [max(div * multiplier, 0.0) for div in divs]


class MarketPriceShock(Event):
    def __init__(self, it: int, price_change: float):
        super().__init__(it)
        self.dp = round(price_change)

    def __repr__(self):
        return f'market price shock (it={self.it}, dp={self.dp})'

    def call(self, it: int):
        if super().call(it):
            return

        book = self.simulator.exchange.order_book
        for order in chain(*book.values()):
            order.price += round(self.dp, 1)


class LiquidityShock(Event):
    def __init__(self, it: int, volume_change: float):
        super().__init__(it)
        self.dv = round(volume_change)

    def __repr__(self):
        return f'liquidity shock (it={self.it}, dv={self.dv})'

    def call(self, it: int):
        if super().call(it):
            return
        exchange = self.simulator.exchange
        pseudo_trader = Trader(exchange, 1e6, int(1e4))
        if self.dv < 0:  # buy
            order = Order(exchange.order_book['ask'].last.price, abs(self.dv), 'bid', pseudo_trader)
        else:  # sell
            order = Order(exchange.order_book['bid'].last.price, abs(self.dv), 'ask', pseudo_trader)
        exchange.market_order(order)


class AgentExitShock(Event):
    def __init__(self, it: int, fraction: float):
        super().__init__(it)
        self.fraction = max(0.0, min(1.0, fraction))

    def __repr__(self):
        pct = round(self.fraction * 100, 1)
        return f'agent exit shock (it={self.it}, fraction={pct}%)'

    def call(self, it: int):
        if super().call(it):
            return

        if not self.simulator.traders:
            return

        victims = set()
        eligible = [
            trader for trader in self.simulator.traders
            if not isinstance(trader, (MarketMaker, AutoMarketMaker))
        ]
        if not eligible or self.fraction <= 0:
            return

        # Группируем по типу, чтобы удалять долю равномерно.
        groups = {}
        for trader in eligible:
            groups.setdefault(type(trader), []).append(trader)

        for traders in groups.values():
            if not traders:
                continue
            n_remove = max(0, math.ceil(len(traders) * self.fraction))
            random.shuffle(traders)
            for trader in traders[:n_remove]:
                victims.add(trader)

        if not victims:
            return

        for trader in victims:
            for order in list(trader.orders):
                trader._cancel_order(order)

        self.simulator.traders = [trader for trader in self.simulator.traders if trader not in victims]


class InformationShock(Event):
    def __init__(self, it, access: int):
        super().__init__(it)
        self.access = access

    def __repr__(self):
        return f'information shock (it={self.it}, access={self.access})'

    def call(self, it: int):
        if super().call(it):
            return
        for trader in self.simulator.traders:
            if type(trader) in (Universalist, Fundamentalist):
                trader.access = self.access


class MarketMakerIn(Event):
    def __init__(self, it, cash: float = 10**3, assets: int = 0, softlimit: int = 100):
        super().__init__(it)
        self.cash = cash
        self.assets = assets
        self.softlimit = softlimit

    def __repr__(self):
        return f'mm in (it={self.it}, softlimit={self.softlimit})'

    def call(self, it: int):
        if super().call(it):
            return

        maker = MarketMaker(self.simulator.exchange, self.cash, self.assets, self.softlimit)
        self.simulator.traders.append(maker)


class MarketMakerOut(Event):
    def __init__(self, it):
        super().__init__(it)

    def __repr__(self):
        return f'mm out (it={self.it})'

    def call(self, it: int):
        if super().call(it):
            return

        self.simulator.traders = [tr for tr in self.simulator.traders if type(tr) != MarketMaker]


class TransactionCost(Event):
    def __init__(self, it, cost):
        super().__init__(it)
        self.cost = cost

    def __repr__(self):
        return f'transaction cost (it={self.it}, cost={self.cost}%)'

    def call(self, it: int):
        if super().call(it):
            return

        self.simulator.exchange.transaction_cost = self.cost
