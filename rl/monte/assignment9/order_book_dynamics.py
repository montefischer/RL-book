from rl.markov_process import MarkovProcess, NonTerminal, Terminal, State
from rl.chapter9.order_book import DollarsAndShares, OrderBook
from rl.distribution import Distribution, SampledDistribution, Poisson, Constant, Gaussian

import numpy as np
import random



class UniformLimitOrderBookMDP(MarkovProcess[OrderBook]):
    """
    Order Book MDP where new buy and sell limit orders arrive randomly at Poisson(\lambda)
    amounts at strike prices uniformly at random at integer price chosen uniformly at random
    between a and b
    """
    a: int
    b: int
    lambd: float

    def __init__(self, a: int, b: int, lambd: float):
        assert 0 < a < b
        self.a = a
        self.b = b
        self.lambd = lambd

    def transition(self, state: NonTerminal[OrderBook]) -> Distribution[State[OrderBook]]:
        def sampler() -> State[OrderBook]:
            # generate a random order
            isAsk = bool(random.randint(0, 1))
            strike = random.randint(self.a, self.b)
            shares = Poisson(self.lambd).sample()
            if isAsk:
                return NonTerminal(state.state.sell_limit_order(strike, shares)[1])
            else:
                return NonTerminal(state.state.buy_limit_order(strike, shares)[1])
        return SampledDistribution(sampler)


class UniformLimitAndMarketOrderBookMDP(MarkovProcess[OrderBook]):
    """
    Order Book MDP where at each time step, a market order arrives with probability
    p, otherwise a limit order arrives. The number of shares for a market order
    is Pois(lambda_1), and Pois(lambda_2) for a limit order. Finally, as in
    UniformLimitOrderBookMDP, limit orders arrive at integer strike chosen uniformly
    at random from [a, b]
    """
    p: float
    a: int
    b: int
    lambd1: float
    lambd2: float

    def __init__(self, p: float, a: int, b: int, lambd1: float, lambd2: float):
        assert 0 <= p <= 1
        assert 0 < a < b
        self.p = p
        self.a = a
        self.b = b
        self.lambd1 = lambd1
        self.lambd2 = lambd2

    def transition(self, state: NonTerminal[OrderBook]) -> Distribution[State[OrderBook]]:
        def sampler() -> State[OrderBook]:
            isAsk = bool(random.randint(0, 1))
            # market or limit?
            if random.random() < self.p:
                # market order
                shares = Poisson(self.lambd1).sample()
                if isAsk:
                    return NonTerminal(state.state.sell_market_order(shares)[1])
                else:
                    return NonTerminal(state.state.buy_market_order(shares)[1])
            else:
                # limit order
                strike = random.randint(self.a, self.b)
                shares = Poisson(self.lambd2).sample()
                if isAsk:
                    return NonTerminal(state.state.sell_limit_order(strike, shares)[1])
                else:
                    return NonTerminal(state.state.buy_limit_order(strike, shares)[1])
        return SampledDistribution(sampler)


class MidNormalLimitMDP(MarkovProcess[OrderBook]):
    """
    Order book MDP where limit orders arrive at price given by the nearest positive integer
    to a sample from Normal(mid, sigma^2), and amount sampled from Pois(lambda).
    """

    def __init__(self, sigma: float, lambd: float):
        assert sigma > 0
        assert lambd > 0
        self.sigma = sigma
        self.lambd = lambd
        self.latest_mid = 100 # start trading here

    def transition(self, state: NonTerminal[OrderBook]) -> Distribution[State[OrderBook]]:
        def sampler() -> State[OrderBook]:
            # generate a random order
            isAsk = bool(random.randint(0, 1))
            if len(state.state.descending_bids) > 0 and len(state.state.ascending_asks) > 0:
                self.latest_mid = state.state.mid_price()
            strike = nearest_positive_int(Gaussian(self.latest_mid, self.sigma).sample())
            shares = Poisson(self.lambd).sample()
            if isAsk:
                return NonTerminal(state.state.sell_limit_order(strike, shares)[1])
            else:
                return NonTerminal(state.state.buy_limit_order(strike, shares)[1])
        return SampledDistribution(sampler)


def nearest_positive_int(x: float):
    if x < 1:
        return 1
    else:
        return round(x)


def simulate_order_book(mdp: MarkovProcess[OrderBook], num_iter: int = 100):
    for iter, odb in enumerate(mdp.simulate(Constant(NonTerminal(OrderBook([], []))))):
        print(f"Iteration #{iter}")
        print(type(odb))
        odb.state.pretty_print_order_book()
        print()
        if iter == num_iter - 1:
            odb.state.display_order_book()
            break


if __name__ == '__main__':
    a: int = 1
    b: int = 10
    lambd: float = 10.0

    #simulate_order_book(UniformLimitOrderBookMDP(a, b, lambd), 100)
    #simulate_order_book(UniformLimitAndMarketOrderBookMDP(0.2, 1, 10, 5, 3), 1000)
    simulate_order_book(MidNormalLimitMDP(3, 1), 1000)
