from AgentBasedModel.simulator import SimulatorInfo
import matplotlib.pyplot as plt
import pandas as pd


def print_book(info: SimulatorInfo, n=5):
    val = pd.concat([
        pd.DataFrame({
            'Sell': [v.price for v in info.exchange.order_book['ask']],
            'Quantity': [v.qty for v in info.exchange.order_book['ask']]
            }).groupby('Sell').sum().reset_index().head(n),
        pd.DataFrame({
            'Buy': [v.price for v in info.exchange.order_book['bid']],
            'Quantity': [v.qty for v in info.exchange.order_book['bid']]
        }).groupby('Buy').sum().reset_index().sort_values('Buy', ascending=False).head(n)
    ])
    print(val[['Buy', 'Sell', 'Quantity']].fillna('').to_string(index=False))


def plot_book(info: SimulatorInfo, bins=50, figsize=(6, 6)):
    bid = list()
    for order in info.exchange.order_book['bid']:
        for p in range(order.qty):
            bid.append(order.price)

    ask = list()
    for order in info.exchange.order_book['ask']:
        for p in range(order.qty):
            ask.append(order.price)

    plt.figure(figsize=figsize)
    plt.title('Order book')
    plt.hist(bid, label='bid', color='green', bins=bins)
    plt.hist(ask, label='ask', color='red', bins=bins)
    plt.show()
