from AgentBasedModel.agents import ExchangeAgent, Universalist, Chartist, Fundamentalist
from AgentBasedModel.utils.math import mean, std, difference, rolling
import random
from tqdm import tqdm


class Simulator:
    """
    Simulator is responsible for launching agents' actions and executing scenarios
    """
    def __init__(self, exchange: ExchangeAgent = None, traders: list = None, events: list = None):
        self.exchange = exchange
        self.events = [event.link(self) for event in events] if events else None  # link all events to simulator
        self.traders = traders
        self.info = SimulatorInfo(self.exchange, self.traders)  # links to existing objects

    def _payments(self):
        for trader in self.traders:
            # Dividend payments
            trader.cash += trader.assets * self.exchange.dividend()  # allow negative dividends
            # Interest payment
            trader.cash += trader.cash * self.exchange.risk_free  # allow risk-free loan

    def simulate(self, n_iter: int, silent=False) -> object:
        for it in tqdm(range(n_iter), desc='Simulation', disable=silent):
            # Call scenario
            if self.events:
                for event in self.events:
                    event.call(it)

            # Capture current info
            self.info.capture()

            # Change behaviour
            for trader in self.traders:
                if type(trader) == Universalist:
                    trader.change_strategy(self.info)
                elif type(trader) == Chartist:
                    trader.change_sentiment(self.info)

            # Call Traders
            random.shuffle(self.traders)
            for trader in self.traders:
                trader.call()

            # Payments and dividends
            self._payments()  # pay dividends
            self.exchange.generate_dividend()  # generate next dividends

        return self


class SimulatorInfo:
    """
    SimulatorInfo is responsible for capturing data during simulating
    """

    def __init__(self, exchange: ExchangeAgent = None, traders: list = None):
        self.exchange = exchange
        self.traders = {t.id: t for t in traders}

        # Market Statistics
        self.prices = list()  # price at the end of iteration
        self.spreads = list()  # bid-ask spreads
        self.dividends = list()  # dividend paid at each iteration
        self.orders = list()  # order book statistics

        # Agent statistics
        self.equities = list()  # agent: equity
        self.cash = list()  # agent: cash
        self.assets = list()  # agent: number of assets
        self.types = list()  # agent: current type
        self.sentiments = list()  # agent: current sentiment
        self.returns = [{tr_id: 0 for tr_id in self.traders.keys()}]  # agent: iteration return

        """
        # Market Statistics
        self.prices = list()  # price at the end of iteration
        self.spreads = list()  # bid-ask spreads
        self.spread_sizes = list()  # bid-ask spread sizes
        self.dividends = list()
        self.orders_quantities = list()  # list -> (bid, ask)
        self.orders_volumes = list()  # list -> (bid, ask) -> (sum, mean, q1, q3, std)
        self.orders_prices = list()  # list -> (bid, ask) -> (mean, q1, q3, std)

        # Agent Statistics
        self.equity = list()  # sum of equity of agents
        self.cash = list()  # sum of cash of agents
        self.assets_qty = list()  # sum of number of assets of agents
        self.assets_value = list()  # sum of value of assets of agents
        """

    def capture(self):
        """
        Method called at the end of each iteration to capture basic info on simulation.

        **Attributes:**

        *Market Statistics*

        - :class:`list[float]` **prices** --> stock prices on each iteration
        - :class:`list[dict]` **spreads** --> order book spreads on each iteration
        - :class:`list[float]` **dividends** --> dividend paid on each iteration
        - :class:`list[dict[dict]]` **orders** --> order book price, volume, quantity stats on each iteration

        *Traders Statistics*

        - :class:`list[dict]` **equities** --> each agent's equity on each iteration
        - :class:`list[dict]` **cash** --> each agent's cash on each iteration
        - :class:`list[dict]` **assets** --> each agent's number of stocks on each iteration
        - :class:`list[dict]` **types** --> each agent's type on each iteration
        """
        # Market Statistics
        self.prices.append(self.exchange.price() if self.exchange.spread() else 0)
        self.spreads.append((self.exchange.spread()))
        self.dividends.append(self.exchange.dividend())
        self.orders.append({
            'quantity': {'bid': len(self.exchange.order_book['bid']), 'ask': len(self.exchange.order_book['ask'])},
            # 'price mean': {
            #     'bid': mean([order.price for order in self.exchange.order_book['bid']]),
            #     'ask': mean([order.price for order in self.exchange.order_book['ask']])},
            # 'price std': {
            #     'bid': std([order.price for order in self.exchange.order_book['bid']]),
            #     'ask': std([order.price for order in self.exchange.order_book['ask']])},
            # 'volume sum': {
            #     'bid': sum([order.qty for order in self.exchange.order_book['bid']]),
            #     'ask': sum([order.qty for order in self.exchange.order_book['ask']])},
            # 'volume mean': {
            #     'bid': mean([order.qty for order in self.exchange.order_book['bid']]),
            #     'ask': mean([order.qty for order in self.exchange.order_book['ask']])},
            # 'volume std': {
            #     'bid': std([order.qty for order in self.exchange.order_book['bid']]),
            #     'ask': std([order.qty for order in self.exchange.order_book['ask']])}
        })

        # Trader Statistics
        self.equities.append({t_id: t.equity() for t_id, t in self.traders.items()})
        self.cash.append({t_id: t.cash for t_id, t in self.traders.items()})
        self.assets.append({t_id: t.assets for t_id, t in self.traders.items()})
        self.types.append({t_id: t.type for t_id, t in self.traders.items()})
        self.sentiments.append({t_id: t.sentiment for t_id, t in self.traders.items() if t.type == 'Chartist'})
        self.returns.append({tr_id: (self.equities[-1][tr_id] - self.equities[-2][tr_id]) / self.equities[-2][tr_id]
                             for tr_id in self.traders.keys()}) if len(self.equities) > 1 else None

    def fundamental_value(self, access: int = 1) -> list:
        divs = self.dividends.copy()
        n = len(divs)  # number of iterations
        divs.extend(self.exchange.dividend(access)[1:access])  # add not recorded future divs
        r = self.exchange.risk_free

        return [Fundamentalist.evaluate(divs[i:i+access], r) for i in range(n)]

    def stock_returns(self, roll: int = None) -> list or float:
        p = self.prices
        div = self.dividends
        r = [(p[i+1] - p[i]) / p[i] + div[i] / p[i] for i in range(len(p) - 1)]
        return rolling(r, roll) if roll else mean(r)

    def abnormal_returns(self, roll: int = None) -> list:
        rf = self.exchange.risk_free
        r = [r - rf for r in self.stock_returns()]
        return rolling(r, roll) if roll else r

    def return_volatility(self, window: int = None) -> list or float:
        returns = self.stock_returns(1)
        if not returns:
            return [] if window else 0.0
        if window is None:
            return std(returns)
        if window <= 0 or len(returns) < window:
            return []
        return [std(returns[i:i+window]) for i in range(len(returns) - window + 1)]

    def price_volatility(self, window: int = None) -> list or float:
        if not self.prices:
            return [] if window else 0.0
        if window is None:
            return std(self.prices)
        if window <= 0 or len(self.prices) < window:
            return []
        return [std(self.prices[i:i+window]) for i in range(len(self.prices) - window + 1)]

    def liquidity(self, roll: int = None) -> list or float:
        n = len(self.prices)
        spreads = [el['ask'] - el['bid'] for el in self.spreads]
        prices = self.prices
        liq = [spreads[i] / prices[i] for i in range(n)]
        return rolling(liq, roll) if roll else mean(liq)
