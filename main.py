from AgentBasedModel import *
from AgentBasedModel.agents.agents import AutoMarketMaker
from AgentBasedModel.utils.math import *

import itertools
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

RANGE = list(range(10, 11)) #6 # КОЛИЧЕСТВО ДРУГИХ АГЕНТОВ
WINDOW = 5
SIZE = 10
FUNCS = [
    ('price', lambda info, w: info.prices)
]

traders = list()
before = list()
after = list()

for n_rand, n_fund, n_chart, n_univ, n_mm in tqdm(list(itertools.product(RANGE, repeat=5))):
    for is_amm in range(2):
        exchange = ExchangeAgent(volume=1000)
        simulator = Simulator(**{
            'exchange': exchange,
            'traders': [
                *[Random(exchange, 10 ** 3) for _ in range(n_rand)],
                *[Fundamentalist(exchange, 10 ** 3) for _ in range(n_fund)],
                *[Chartist(exchange, 10 ** 3) for _ in range(n_chart)],
                *[Universalist(exchange, 10 ** 3) for _ in range(n_univ)],
                *[MarketMaker(exchange, 10 ** 3) for _ in range(n_mm)],
                *[AutoMarketMaker(exchange, 10 ** 3) for _ in range(is_amm)]
            ],
            'events': [MarketPriceShock(200, -10)]
        })
        info = simulator.info
        simulator.simulate(500, silent=True)

        tmp = aggToShock(simulator, 1, FUNCS)['market price shock (it=200, dp=-10)']['price']

        traders.append({'Random': n_rand, 'Fundamentalist': n_fund, 'Chartist': n_chart, 'Universalist': n_univ,
                        'MarketMaker': n_mm, "AutoMarketMaker": is_amm})
        before.append(tmp['right before'])
        after.append(tmp['after'])
        plot_price(info)
