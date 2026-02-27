import random

from math import floor, ceil

from AgentBasedModel.utils import Order, OrderList
from AgentBasedModel.utils.math import exp, mean
import math
import copy
from AgentBasedModel.utils.pools import BasePool, ConstantProductPool, HFPool


class ExchangeAgent:
    """
    ExchangeAgent implements automatic orders handling within the order book. It supports limit orders,
    market orders, cancel orders, returns current spread prices and volumes.
    """
    id = 0

    def __init__(self, price: float or int = 100, std: float or int = 25, volume: int = 1000, rf: float = 5e-4,
                 transaction_cost: float = 0):
        """
        Initialization parameters
        :param price: stock initial price
        :param std: standard deviation of order prices in book
        :param volume: number of orders in book
        :param rf: risk-free rate (interest rate for cash holdings of agents)
        :param transaction_cost: cost that is paid on each successful deal
        """
        self.name = f'ExchangeAgent{self.id}'
        ExchangeAgent.id += 1

        self.order_book = {'bid': OrderList('bid'), 'ask': OrderList('ask')}
        self.dividend_book = list()  # list of future dividends
        self.risk_free = rf
        self.transaction_cost = transaction_cost
        self._fill_book(price, std, volume, rf * price)

    def generate_dividend(self):
        """
        Generate time series on future dividends.
        """
        # Generate future dividend
        d = self.dividend_book[-1] * self._next_dividend()
        self.dividend_book.append(max(d, 0))  # dividend > 0
        self.dividend_book.pop(0)

    def _fill_book(self, price, std, volume, div: float = 0.05):
        """
        Fill order book with random orders. Fill dividend book with n future dividends.
        """
        # Order book
        prices1 = [round(random.normalvariate(price - std, std), 1) for _ in range(volume // 2)]
        prices2 = [round(random.normalvariate(price + std, std), 1) for _ in range(volume // 2)]
        quantities = [random.randint(1, 5) for _ in range(volume)]

        for (p, q) in zip(sorted(prices1 + prices2), quantities):
            if p > price:
                order = Order(round(p, 1), q, 'ask', None)
                self.order_book['ask'].append(order)
            else:
                order = Order(p, q, 'bid', None)
                self.order_book['bid'].push(order)

        # Dividend book
        for i in range(100):
            self.dividend_book.append(max(div, 0))  # dividend > 0
            div *= self._next_dividend()

    def spread(self) -> dict or None:
        """
        :return: {'bid': float, 'ask': float}
        """
        if self.order_book['bid'] and self.order_book['ask']:
            return {'bid': self.order_book['bid'].first.price, 'ask': self.order_book['ask'].first.price}
        return None

    def spread_volume(self) -> dict or None:
        """
        :return: {'bid': float, 'ask': float}
        """
        if self.order_book['bid'] and self.order_book['ask']:
            return {'bid': self.order_book['bid'].first.qty, 'ask': self.order_book['ask'].first.qty}
        return None

    def price(self) -> float or None:
        spread = self.spread()
        if spread:
            return round((spread['bid'] + spread['ask']) / 2, 1) # (LD) IMPORTANT: how is the price calculated
        raise Exception(f'Price cannot be determined, since no orders either bid or ask')

    def dividend(self, access: int = None) -> list or float:
        """
        Returns current dividend payment value. If called by a trader, returns n future dividends
        given information access.
        """
        if access is None:
            return self.dividend_book[0]
        return self.dividend_book[:access]

    @classmethod
    def _next_dividend(cls, std=5e-3):
        return exp(random.normalvariate(0, std))

    def limit_order(self, order: Order):
        """
        Executes limit order, fulfilling orders if on other side of spread

        :return: void
        """
        bid, ask = self.spread().values()
        t_cost = self.transaction_cost
        if not bid or not ask:
            return

        if order.order_type == 'bid':
            if order.price >= ask:
                order = self.order_book['ask'].fulfill(order, t_cost)
            if order.qty > 0:
                self.order_book['bid'].insert(order)
            return

        elif order.order_type == 'ask':
            if order.price <= bid:
                order = self.order_book['bid'].fulfill(order, t_cost)
            if order.qty > 0:
                self.order_book['ask'].insert(order)

    def market_order(self, order: Order) -> Order:
        """
        Executes market order, fulfilling orders on the other side of spread

        :return: Order
        """
        t_cost = self.transaction_cost
        if order.order_type == 'bid':
            order = self.order_book['ask'].fulfill(order, t_cost)
        elif order.order_type == 'ask':
            order = self.order_book['bid'].fulfill(order, t_cost)
        return order

    def cancel_order(self, order: Order):
        """
        Cancel order from order book

        :return: void
        """
        if order.order_type == 'bid':
            self.order_book['bid'].remove(order)
        elif order.order_type == 'ask':
            self.order_book['ask'].remove(order)

    def get_best_bid(self):
        if self.order_book['bid']:
            return {'price': self.order_book['bid'].first.price, 'qty': self.order_book['bid'].first.qty}
        return None

    def get_best_ask(self):
        if self.order_book['ask']:
            return {'price': self.order_book['ask'].first.price, 'qty': self.order_book['ask'].first.qty}
        return None


class Trader:
    id = 0

    def __init__(self, market: ExchangeAgent, cash: float or int, assets: int = 0):
        """
        Trader that is activated on call to perform action.

        :param market: link to exchange agent
        :param cash: trader's cash available
        :param assets: trader's number of shares hold
        """
        self.type = 'Unknown'
        self.name = f'Trader{self.id}'
        self.id = Trader.id
        Trader.id += 1

        self.market = market
        self.orders = list()

        self.cash = cash
        self.assets = assets

    def __str__(self) -> str:
        return f'{self.name} ({self.type})'

    def equity(self):
        price = self.market.price() if self.market.spread() is not None else 0
        return self.cash + self.assets * price

    def _buy_limit(self, quantity, price):
        order = Order(round(price, 1), round(quantity), 'bid', self)
        self.orders.append(order)
        self.market.limit_order(order)

    def _sell_limit(self, quantity, price):
        order = Order(round(price, 1), round(quantity), 'ask', self)
        self.orders.append(order)
        self.market.limit_order(order)

    def _buy_market(self, quantity) -> int:
        """
        :return: quantity unfulfilled
        """
        if not self.market.order_book['ask']:
            return quantity
        order = Order(self.market.order_book['ask'].last.price, round(quantity), 'bid', self)
        return self.market.market_order(order).qty

    def _sell_market(self, quantity) -> int:
        """
        :return: quantity unfulfilled
        """
        if not self.market.order_book['bid']:
            return quantity
        order = Order(self.market.order_book['bid'].last.price, round(quantity), 'ask', self)
        return self.market.market_order(order).qty

    def _cancel_order(self, order: Order):
        self.market.cancel_order(order)
        self.orders.remove(order)


class Random(Trader):
    """
    Random creates noisy orders to recreate trading in real environment.
    """
    def __init__(self, market: ExchangeAgent, cash: float or int, assets: int = 0):
        super().__init__(market, cash, assets)
        self.type = 'Random'

    @staticmethod
    def draw_delta(std: float or int = 2.5):
        lamb = 1 / std
        return random.expovariate(lamb)

    @staticmethod
    def draw_price(order_type, spread: dict, std: float or int = 2.5) -> float:
        """
        Draw price for limit order of Noise Agent. The price is calculated as:
        1) 35% - within the spread - uniform distribution
        2) 65% - out of the spread - delta from best price is exponential distribution r.v.
        """
        random_state = random.random()  # Determines IN spread OR OUT of spread

        # Within the spread
        if random_state < .35:
            return random.uniform(spread['bid'], spread['ask'])

        # Out of spread
        else:
            delta = Random.draw_delta(std)
            if order_type == 'bid':
                return spread['bid'] - delta
            if order_type == 'ask':
                return spread['ask'] + delta

    @staticmethod
    def draw_quantity(a=1, b=5) -> float:
        """
        Draw random quantity to buy from uniform distribution.

        :param a: minimal quantity
        :param b: maximal quantity
        :return: quantity for order
        """
        return random.randint(a, b)

    def call(self):
        spread = self.market.spread()
        if spread is None:
            return

        random_state = random.random()

        if random_state > .5:
            order_type = 'bid'
        else:
            order_type = 'ask'

        random_state = random.random()
        # Market order
        if random_state > .85:
            quantity = self.draw_quantity()
            if order_type == 'bid':
                self._buy_market(quantity)
            elif order_type == 'ask':
                self._sell_market(quantity)

        # Limit order
        elif random_state > .5:
            price = self.draw_price(order_type, spread)
            quantity = self.draw_quantity()
            if order_type == 'bid':
                self._buy_limit(quantity, price)
            elif order_type == 'ask':
                self._sell_limit(quantity, price)

        # Cancellation order
        elif random_state < .35:
            if self.orders:
                order_n = random.randint(0, len(self.orders) - 1)
                self._cancel_order(self.orders[order_n])


class Fundamentalist(Trader):
    """
    Fundamentalist traders strictly believe in the information they receive. If they find an ask
    order with a price lower or a bid order with a price higher than their estimated present
    value, i.e. E(V|Ij,k), they accept the limit order, otherwise they put a new limit order
    between the former best bid and best ask prices.
    """
    def __init__(self, market: ExchangeAgent, cash: float or int, assets: int = 0, access: int = 1):
        """
        :param market: exchange agent link
        :param cash: number of cash
        :param assets: number of assets
        :param access: number of future dividends informed
        """
        super().__init__(market, cash, assets)
        self.type = 'Fundamentalist'
        self.access = access

    @staticmethod
    def evaluate(dividends: list, risk_free: float):
        """
        Evaluate stock using constant dividend model.
        """
        divs = dividends  # expected value of future dividends
        r = risk_free  # risk-free rate

        perp = divs[-1] / r / (1 + r)**(len(divs) - 1)  # perpetual payments
        known = sum([divs[i] / (1 + r)**(i + 1) for i in range(len(divs) - 1)]) if len(divs) > 1 else 0
        return known + perp

    @staticmethod
    def draw_quantity(pf, p, gamma: float = 5e-3):
        q = round(abs(pf - p) / p / gamma)
        return min(q, 5)

    def call(self):
        pf = round(self.evaluate(self.market.dividend(self.access), self.market.risk_free), 1)  # fundamental price
        spread = self.market.spread()
        t_cost = self.market.transaction_cost

        if spread is None:
            return
        p = self.market.price()

        random_state = random.random()
        qty = Fundamentalist.draw_quantity(pf, p)  # quantity to buy
        if not qty:
            return

        # Limit or Market order
        if random_state > .45:
            random_state = random.random()

            ask_t = round(spread['ask'] * (1 + t_cost), 1)
            bid_t = round(spread['bid'] * (1 - t_cost), 1)

            if pf >= ask_t:
                if random_state > .5:
                    self._buy_market(qty)
                else:
                    self._sell_limit(qty, (pf + Random.draw_delta()) * (1 + t_cost))

            elif pf <= bid_t:
                if random_state > .5:
                    self._sell_market(qty)
                else:
                    self._buy_limit(qty, (pf - Random.draw_delta()) * (1 - t_cost))

            elif ask_t > pf > bid_t:
                if random_state > .5:
                    self._buy_limit(qty, (pf - Random.draw_delta()) * (1 - t_cost))
                else:
                    self._sell_limit(qty, (pf + Random.draw_delta()) * (1 + t_cost))

        # Cancel order
        else:
            if self.orders:
                self._cancel_order(self.orders[0])


class Chartist(Trader):
    """
    Chartist traders are searching for trends in the price movements. Each trader has sentiment - opinion
    about future price movement (either increasing, or decreasing). Based on sentiment trader either
    buys stock or sells. Sentiment revaluation happens at the end of each iteration based on opinion
    propagation among other chartists, current price changes.
    """
    def __init__(self, market: ExchangeAgent, cash: float or int, assets: int = 0):
        """
        :param market: exchange agent link
        :param cash: number of cash
        :param assets: number of assets
        """
        super().__init__(market, cash, assets)
        self.type = 'Chartist'
        self.sentiment = 'Optimistic' if random.random() > .5 else 'Pessimistic'

    def call(self):
        """
        If 'steps' consecutive steps of upward (downward) price movements -> buy (sell) market order. If there are no
        such trend, act as random trader placing only limit orders.
        """
        random_state = random.random()
        t_cost = self.market.transaction_cost
        spread = self.market.spread()
        if spread is None:
            return

        if self.sentiment == 'Optimistic':
            # Market order
            if random_state > .85:
                self._buy_market(Random.draw_quantity())
            # Limit order
            elif random_state > .5:
                self._buy_limit(Random.draw_quantity(), Random.draw_price('bid', spread) * (1 - t_cost))
            # Cancel order
            elif random_state < .35:
                if self.orders:
                    self._cancel_order(self.orders[-1])
        elif self.sentiment == 'Pessimistic':
            # Market order
            if random_state > .85:
                self._sell_market(Random.draw_quantity())
            # Limit order
            elif random_state > .5:
                self._sell_limit(Random.draw_quantity(), Random.draw_price('ask', spread) * (1 + t_cost))
            # Cancel order
            elif random_state < .35:
                if self.orders:
                    self._cancel_order(self.orders[-1])

    def change_sentiment(self, info, a1=1, a2=1, v1=.1):
        """
        Change sentiment

        :param info: SimulatorInfo
        :param a1: importance of chartists opinion
        :param a2: importance of current price changes
        :param v1: frequency of revaluation of opinion for sentiment
        """
        n_traders = len(info.traders)  # number of all traders
        n_chartists = sum([tr_type == 'Chartist' for tr_type in info.types[-1].values()])
        n_optimistic = sum([tr_type == 'Optimistic' for tr_type in info.sentiments[-1].values()])
        n_pessimists = sum([tr_type == 'Pessimistic' for tr_type in info.sentiments[-1].values()])
        if n_chartists <= 0:
            return
        if self.market.spread() is None:
            return

        dp = info.prices[-1] - info.prices[-2] if len(info.prices) > 1 else 0  # price derivative
        p = self.market.price()  # market price
        x = (n_optimistic - n_pessimists) / n_chartists

        U = a1 * x + a2 / v1 * dp / p
        if self.sentiment == 'Optimistic':
            prob = v1 * n_chartists / n_traders * exp(U)
            if prob > random.random():
                self.sentiment = 'Pessimistic'

        elif self.sentiment == 'Pessimistic':
            prob = v1 * n_chartists / n_traders * exp(-U)
            if prob > random.random():
                self.sentiment = 'Optimistic'

        # print('sentiment', prob)


class Universalist(Fundamentalist, Chartist):
    """
    Universalist mixes Fundamentalist, Chartist trading strategies, and allows to change from
    one strategy to another.
    """
    def __init__(self, market: ExchangeAgent, cash: float or int, assets: int = 0, access: int = 1):
        """
        :param market: exchange agent link
        :param cash: number of cash
        :param assets: number of assets
        :param access: number of future dividends informed
        """
        super().__init__(market, cash, assets)
        self.type = 'Chartist' if random.random() > .5 else 'Fundamentalist'  # randomly decide type
        self.sentiment = 'Optimistic' if random.random() > .5 else 'Pessimistic'  # sentiment about trend (Chartist)
        self.access = access  # next n dividend payments known (Fundamentalist)

    def call(self):
        """
        Call one of parents' methods depending on what type it is currently set.
        """
        if self.type == 'Chartist':
            Chartist.call(self)
        elif self.type == 'Fundamentalist':
            Fundamentalist.call(self)

    def change_strategy(self, info, a1=1, a2=1, a3=1, v1=.1, v2=.1, s=.1):
        """
        Change strategy or sentiment

        :param info: SimulatorInfo
        :param a1: importance of chartists opinion
        :param a2: importance of current price changes
        :param a3: importance of fundamentalist profit
        :param v1: frequency of revaluation of opinion for sentiment
        :param v2: frequency of revaluation of opinion for strategy
        :param s: importance of fundamental value opportunities
        """
        # Gather variables
        n_traders = len(info.traders)  # number of all traders
        n_fundamentalists = sum([tr.type == 'Fundamentalist' for tr in info.traders.values()])
        n_optimistic = sum([tr.sentiment == 'Optimistic' for tr in info.traders.values() if tr.type == 'Chartist'])
        n_pessimists = sum([tr.sentiment == 'Pessimistic' for tr in info.traders.values() if tr.type == 'Chartist'])
        if n_traders <= 0:
            return
        if self.market.spread() is None:
            return

        dp = info.prices[-1] - info.prices[-2] if len(info.prices) > 1 else 0  # price derivative
        p = self.market.price()  # market price
        pf = self.evaluate(self.market.dividend(self.access), self.market.risk_free)  # fundamental price
        r = pf * self.market.risk_free  # expected dividend return
        R = mean(info.returns[-1].values())  # average return in economy

        # Change sentiment
        if self.type == 'Chartist':
            Chartist.change_sentiment(self, info, a1, a2, v1)

        # Change strategy
        U1 = max(-100, min(100, a3 * ((r + 1 / v2 * dp) / p - R - s * abs((pf - p) / p))))
        U2 = max(-100, min(100, a3 * (R - (r + 1 / v2 * dp) / p - s * abs((pf - p) / p))))

        if self.type == 'Chartist':
            if self.sentiment == 'Optimistic':
                prob = v2 * n_optimistic / (n_traders * exp(U1))
                if prob > random.random():
                    self.type = 'Fundamentalist'
            elif self.sentiment == 'Pessimistic':
                prob = v2 * n_pessimists / (n_traders * exp(U2))
                if prob > random.random():
                    self.type = 'Fundamentalist'

        elif self.type == 'Fundamentalist':
            prob = v2 * n_fundamentalists / (n_traders * exp(-U1))
            if prob > random.random() and self.sentiment == 'Pessimistic':
                self.type = 'Chartist'
                self.sentiment = 'Optimistic'

            prob = v2 * n_fundamentalists / (n_traders * exp(-U2))
            if prob > random.random() and self.sentiment == 'Optimistic':
                self.type = 'Chartist'
                self.sentiment = 'Pessimistic'


class MarketMaker(Trader):
    """
    MarketMaker creates limit orders on both sides of the spread trying to gain on
    spread between bid and ask prices, and maintain its assets to cash ratio in balance.
    """

    def __init__(self, market: ExchangeAgent, cash: float, assets: int = 0, softlimit: int = 100):
        super().__init__(market, cash, assets)
        self.type = 'Market Maker'
        self.softlimit = softlimit
        self.ul = softlimit
        self.ll = -softlimit
        self.panic = False

    def call(self):
        # Clear previous orders
        for order in self.orders.copy():
            self._cancel_order(order)

        spread = self.market.spread()
        if spread is None:
            return

        # Calculate bid and ask volume
        bid_volume = max(0., self.ul - 1 - self.assets)
        ask_volume = max(0., self.assets - self.ll - 1)

        # If in panic state we only either sell or buy commodities
        if bid_volume <= 0 or ask_volume <= 0:
            self.panic = True
            mid_inventory = (self.ul + self.ll) / 2
            if ask_volume <= 0:
                need = mid_inventory - self.assets
                if need > 0:
                    self._buy_market(need)
            if bid_volume <= 0:
                excess = self.assets - mid_inventory
                if excess > 0:
                    self._sell_market(excess)
            return
        else:
            self.panic = False
            base_offset = -((spread['ask'] - spread['bid']) * (self.assets / self.softlimit))  # Price offset
            self._buy_limit(bid_volume, spread['bid'] - base_offset - .1)  # BID
            self._sell_limit(ask_volume, spread['ask'] + base_offset + .1)  # ASK


class BaseAMMAgent(Trader):
    """
    Общая логика для AMM-агентов: хранит ссылку на пул-инвариант и утилиты для расчета сеток цен.
    """

    def __init__(self, market: ExchangeAgent, cash: float, assets: int, pool_cls, pool_kwargs=None):
        super().__init__(market, cash, assets)
        self.pool_cls = pool_cls
        self.pool_kwargs = pool_kwargs or {}
        self.pool = self.pool_cls(self.assets, self.cash, **self.pool_kwargs)

    def _sync_pool(self):
        """Синхронизировать математический пул с фактическими запасами агента."""
        self.pool.x = float(self.assets)
        self.pool.y = float(self.cash)

    def _clone_pool(self):
        """Сделать копию пула для симуляции ценовой сетки."""
        cloned_kwargs = copy.deepcopy(self.pool_kwargs)
        clone = self.pool_cls(self.pool.x, self.pool.y, **cloned_kwargs)
        return clone

    @staticmethod
    def _dy_for_dx(pool: BasePool, dx_target: float, tol: float = 1e-6) -> float:
        """
        Сколько cash (dy) нужно занести в пул, чтобы забрать dx_target базового актива.
        Ищется двоичным поиском по quote_in.
        """
        if dx_target <= 0:
            return 0.0

        # начальное верхнее ограничение — удваиваем, пока не хватит
        hi = max(pool.y * 0.1, 1.0)
        while pool.quote_in(hi) < dx_target:
            hi *= 2
            if hi > pool.y * 1e6 + 1:
                return 0.0
        lo = 0.0
        for _ in range(60):
            mid = (lo + hi) / 2
            dx = pool.quote_in(mid)
            if abs(dx - dx_target) <= tol:
                return mid
            if dx < dx_target:
                lo = mid
            else:
                hi = mid
        return hi

    def _price_grid_sell(self, max_orders: int, unit: float = 1.0) -> list[float]:
        """
        Сетка цен на продажу базового актива размером unit, посчитанная вдоль инварианта.
        """
        tmp = self._clone_pool()
        prices = []
        for _ in range(max_orders):
            dy = tmp.apply_out(unit)
            if dy <= 0:
                break
            prices.append(round(dy / unit, 2))
        return prices

    def _price_grid_buy(self, max_orders: int, unit: float = 1.0) -> list[float]:
        """
        Сетка цен на покупку базового актива (тратим cash), размер unit.
        """
        tmp = self._clone_pool()
        prices = []
        for _ in range(max_orders):
            dy = self._dy_for_dx(tmp, unit)
            if dy <= 0:
                break
            received = tmp.apply_in(dy)
            if received <= 0:
                break
            prices.append(round(dy / received, 2))
        return prices


class AutoMarketMaker(BaseAMMAgent):
    """
    AutoMarketMaker creates limit orders on both sides of the spread trying to gain on
    spread between bid and ask prices, and maintain its assets to cash ratio in balance.
    """

    MAX_LIMIT_BUY = 100
    MAX_LIMIT_SELL = 200
    MAX_TRADE_FRACTION = 0.1 # доля запасов, доступная для рыночных действий за шаг
    LIMIT_FRACTION = 0.2 # доля запасов, распределяемая по сетке лимитных ордеров

    def __init__(self, market: ExchangeAgent, cash: float, assets: int = 0, initial_ratio: float = 0.5):
        super().__init__(market, cash, assets, pool_cls=ConstantProductPool)
        self.type = 'Automated Market Maker'
        self.initial_ratio = initial_ratio
        self.cash_to_buy = self.cash * self.initial_ratio
        if not (0 < initial_ratio < 1):
            raise ValueError(f'Wrong initial ratio type! Ratio: {self.initial_ratio}')

    def _inner_price(self) -> float:
        """
        Internal valuation proxy. If reserves are empty, fall back to current spread midpoint
        to avoid division-by-zero explosions.
        """
        if self.assets > 0:
            return self.cash / self.assets
        spread = self.market.spread()
        if spread:
            return (spread['ask'] + spread['bid']) / 2
        return max(self.cash, 1.0)

    def filling_amount_bid(self, price, qty):
        if self.assets <= 0 or self.cash <= 0:
            return 0

        inner_price = self._inner_price()
        if price <= inner_price:
            return 0

        k = self.assets - self.cash / price
        return max(0, floor(min(k, qty)))

    def filling_amount_ask(self, price, qty):
        if self.cash <= 0:
            return 0

        inner_price = self._inner_price()
        if price >= inner_price:
            return 0

        k = self.cash / price - self.assets
        return max(0, floor(min(k, qty)))

    def call(self):
        # Clear previous orders
        for order in self.orders.copy():
            self._cancel_order(order)

        # Начальная балансировка до целевого отношения
        if self.cash_to_buy > 0:
            spend_cap = min(self.cash_to_buy, self.cash * self.MAX_TRADE_FRACTION)
            best_ask = self.market.get_best_ask()
            if best_ask and best_ask['price'] > 0 and spend_cap > 0:
                affordable_qty = min(best_ask['qty'], spend_cap / best_ask['price'])
                if affordable_qty > 0:
                    cash_before = self.cash
                    unfilled = self._buy_market(affordable_qty)
                    spent = cash_before - self.cash
                    self.cash_to_buy = max(0.0, self.cash_to_buy - spent)
            if self.cash_to_buy > 0:
                return

        # Одноразовая рыночная подстройка к книге заявок
        bid = self.market.get_best_bid()
        ask = self.market.get_best_ask()

        max_sell_volume = max(0.0, min(self.assets, self.assets * self.MAX_TRADE_FRACTION))
        max_buy_volume = 0.0
        if ask and ask['price'] > 0:
            max_buy_volume = max(0.0, min(self.cash / ask['price'], (self.cash / ask['price']) * self.MAX_TRADE_FRACTION))

        if bid:
            to_sell = min(self.filling_amount_bid(**bid), max_sell_volume)
            if to_sell > 0:
                self._sell_market(to_sell)

        if ask:
            to_buy = min(self.filling_amount_ask(**ask), max_buy_volume)
            if to_buy > 0:
                self._buy_market(to_buy)

        # Размещение ограниченной сетки лимитных ордеров
        self._sync_pool()
        allowed_sell_orders = min(self.MAX_LIMIT_SELL, max(1, int(self.assets * self.LIMIT_FRACTION)))
        try:
            ref_price = max(self.market.price(), 1e-6)
        except Exception:
            ref_price = max(self.pool.spot_price(), 1e-6)
        allowed_buy_orders = min(self.MAX_LIMIT_BUY, max(1, int((self.cash / ref_price) * self.LIMIT_FRACTION)))

        sell_prices = self._price_grid_sell(allowed_sell_orders, unit=1.0)
        buy_prices = self._price_grid_buy(allowed_buy_orders, unit=1.0)

        for p in sell_prices:
            self._sell_limit(1, p)
        for p in buy_prices:
            self._buy_limit(1, p)


class HFMarketMaker(BaseAMMAgent):
    """
    Hybrid Function Market Maker (HFMM) по Curve CryptoInvariant (без комиссии).
    Использует HFPool для вычислений обменной кривой.
    """

    MAX_LIMIT_BUY = 50
    MAX_LIMIT_SELL = 50
    MAX_TRADE_FRACTION = 0.1
    LIMIT_FRACTION = 0.2

    def __init__(self, market: ExchangeAgent, cash: float, assets: int = 0, A: float = 100.0, gamma: float = 1e-3):
        super().__init__(market, cash, assets, pool_cls=HFPool, pool_kwargs={"A": A, "gamma": gamma})
        self.type = 'HFMM'

    def call(self):
        # Remove previous orders
        for order in self.orders.copy():
            self._cancel_order(order)

        self._sync_pool()
        if self.assets <= 0 or self.cash <= 0:
            return

        bid = self.market.get_best_bid()
        ask = self.market.get_best_ask()

        # Рыночная подстройка с ограничением на долю запасов
        max_sell = min(self.assets * self.MAX_TRADE_FRACTION, self.assets)
        max_buy_cash = self.cash * self.MAX_TRADE_FRACTION

        if bid and max_sell > 0:
            qty = min(max_sell, bid['qty'])
            dy = self.pool.quote_out(qty)
            price_pool = dy / qty if qty > 0 else float('inf')
            if dy > 0 and bid['price'] >= price_pool:
                self._sell_market(qty)

        if ask and max_buy_cash > 0:
            qty_affordable = min(max_buy_cash / ask['price'], ask['qty'])
            dy_needed = self._dy_for_dx(self.pool, qty_affordable)
            price_pool = dy_needed / qty_affordable if qty_affordable > 0 else float('inf')
            if dy_needed > 0 and ask['price'] <= price_pool:
                self._buy_market(qty_affordable)

        # Сетка лимитных ордеров
        self._sync_pool()
        sell_orders = min(self.MAX_LIMIT_SELL, max(1, int(self.assets * self.LIMIT_FRACTION)))
        try:
            ref_price = max(self.market.price(), 1e-6)
        except Exception:
            ref_price = max(self.pool.spot_price(), 1e-6)
        buy_orders = min(self.MAX_LIMIT_BUY, max(1, int((self.cash / ref_price) * self.LIMIT_FRACTION)))

        sell_prices = self._price_grid_sell(sell_orders, unit=1.0)
        buy_prices = self._price_grid_buy(buy_orders, unit=1.0)

        for p in sell_prices:
            self._sell_limit(1, p)
        for p in buy_prices:
            self._buy_limit(1, p)
