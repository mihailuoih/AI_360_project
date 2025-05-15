from AgentBasedModel.simulator import SimulatorInfo
import AgentBasedModel.utils.math as math
import matplotlib.pyplot as plt


def plot_equity(info: SimulatorInfo, rolling: int = 1, figsize=(6, 6)):
    plt.figure(figsize=figsize)
    plt.title('Equity') if rolling == 1 else plt.title(f'Equity (MA {rolling})')
    plt.xlabel('Iterations')
    plt.ylabel('Mean Equity')

    labels = ['Random', 'Fundamentalist', 'Chartist']
    data = math.aggregate(info.types, info.equities, labels)
    for k, v in data.items():
        if sum([_ is not None for _ in v]):
            plt.plot(range(rolling, len(v)), math.rolling(v, rolling), label=k)

    plt.legend()
    plt.show()


def plot_cash(info: SimulatorInfo, rolling: int = 1, figsize=(6, 6)):
    plt.figure(figsize=figsize)
    plt.title('Cash') if rolling == 1 else plt.title(f'Cash (MA {rolling})')
    plt.xlabel('Iterations')
    plt.ylabel('Mean Cash')

    labels = ['Random', 'Fundamentalist', 'Chartist']
    data = math.aggregate(info.types, info.cash, labels)
    for k, v in data.items():
        if sum([_ is not None for _ in v]):
            plt.plot(range(rolling, len(v)), math.rolling(v, rolling), label=k)

    plt.legend()
    plt.show()


def plot_assets(info: SimulatorInfo, rolling: int = 1, figsize=(6, 6)):
    plt.figure(figsize=figsize)
    plt.title('Assets') if rolling == 1 else plt.title(f'Assets (MA {rolling})')
    plt.xlabel('Iterations')
    plt.ylabel('Mean Assets')

    labels = ['Random', 'Fundamentalist', 'Chartist']
    data = math.aggregate(info.types, info.assets, labels)
    for k, v in data.items():
        if sum([_ is not None for _ in v]):
            plt.plot(range(rolling, len(v)), math.rolling(v, rolling), label=k)

    plt.legend()
    plt.show()


def plot_strategies(info: SimulatorInfo, rolling: int = 1, figsize=(6, 6)):
    plt.figure(figsize=figsize)
    plt.title('Strategy') if rolling == 1 else plt.title(f'Strategy (MA {rolling})')
    plt.xlabel('Iterations')
    plt.ylabel('Number of traders')

    for tr_type in ['Fundamentalist', 'Chartist']:
        v = [sum([t == tr_type for t in v.values()]) for v in info.types]
        plt.plot(range(rolling, len(v)), math.rolling(v, rolling), label=tr_type)

    plt.legend()
    plt.show()


def plot_strategies2(info: SimulatorInfo, rolling: int = 1, figsize=(6, 6)):
    plt.figure(figsize=figsize)
    plt.title('Strategy') if rolling == 1 else plt.title(f'Strategy (MA {rolling})')
    plt.xlabel('Iterations')
    plt.ylabel('Share of Chartists among Traders')

    v = [sum([t == 'Chartist' for t in v.values()]) / len(v) for v in info.types]
    plt.plot(range(rolling, len(v)), math.rolling(v, rolling), color='black')
    plt.show()


def plot_sentiments(info: SimulatorInfo, rolling: int = 1, figsize=(6, 6)):
    plt.figure(figsize=figsize)
    plt.title('Sentiment') if rolling == 1 else plt.title(f'Sentiment (MA {rolling})')
    plt.xlabel('Iterations')
    plt.ylabel('Number of traders')

    for tr_type in ['Optimistic', 'Pessimistic']:
        v = [sum([t == tr_type for t in v.values()]) for v in info.sentiments]
        plt.plot(range(rolling, len(v)), math.rolling(v, rolling), label=tr_type)

    plt.legend()
    plt.show()


def plot_sentiments2(info: SimulatorInfo, rolling: int = 1, figsize=(6, 6)):
    plt.figure(figsize=figsize)
    plt.title('Sentiment') if rolling == 1 else plt.title(f'Sentiment (MA {rolling})')
    plt.xlabel('Iterations')
    plt.ylabel('Share of Pessimists among Chartists')

    v = [sum([t == 'Pessimistic' for t in v.values()]) / len(v) for v in info.sentiments]
    plt.plot(range(rolling, len(v)), math.rolling(v, rolling), color='black')

    plt.show()


def plot_returns(info: SimulatorInfo, rolling: int = 1, figsize=(6, 6)):
    plt.figure(figsize=figsize)
    plt.title('Realized Returns') if rolling == 1 else plt.title(f'Realized Returns (MA {rolling})')
    plt.xlabel('Iterations')
    plt.ylabel('Mean Returns')

    labels = ['Fundamentalist', 'Chartist']
    data = math.aggregate(info.types, info.returns, labels)
    for k, v in data.items():
        try:
            plt.plot(range(rolling, len(v)), math.rolling(v, rolling), label=k)
        except:
            plt.plot(v, label=k)

    plt.plot(range(rolling, (len(info.returns))), [info.exchange.risk_free] * (len(info.returns) - rolling),
             ls='--', color='black', label='risk-free rate')

    plt.legend()
    plt.show()
