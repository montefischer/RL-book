import itertools
from typing import Iterator, Tuple, TypeVar, Callable, Iterable, Mapping, Sequence
from operator import itemgetter
import numpy as np

from rl.chapter10.mc_td_experience_replay import get_fixed_episodes_from_sr_pairs_seq, \
    get_return_steps_from_fixed_episodes
from rl.markov_process import FiniteMarkovProcess, NonTerminal, \
    State, ReturnStep, TransitionStep

from rl.returns import returns
from rl.iterate import last, accumulate

S = TypeVar('S')

TabularValueFunctionApprox = Mapping[NonTerminal[S], float]


def tabular_mc_update(
        v: TabularValueFunctionApprox,
        counts: Mapping[NonTerminal[S], int],
        steps: Iterable[ReturnStep[S]],
        count_to_weight_func: Callable[[int], float]
):
    """
    Perform a Monte Carlo update of the predicted value function of a FMRP given a
    simulated trace.

    Arguments:
          v -- Current value function approximation (will be modified)
          counts -- Dictionary of how many times state S has appeared in previous traces (will be modified)
          steps -- iterator representing episode trace
          count_to_weight_func -- learning rate parameter defined as a function of how many
            times a state has been encountered previously
    """
    for step in steps:
        state, reward = step.state, step.return_
        counts[state] += 1
        v[state] = v[state] + count_to_weight_func(counts[state]) * (reward - v[state])


def tabular_mc_prediction(
        traces: Iterable[Iterable[TransitionStep]],
        approx_0: TabularValueFunctionApprox,
        gamma: float,
        episode_length_tolerance: float = 1e-6
) -> Iterator[TabularValueFunctionApprox[S]]:
    """
    Evaluate a finite Markov Reward Process using tabular Monte Carlo. Episodes
    are simulated according to the given length tolerance. Yields approximations
    of the value function for the FMRP after updating from an additional episode.

    Arguments:
        traces -- iterator of simultion traces from the FMRP
        approx_0 -- initial value function approximation
        gamma -- discount parameter for FMRP
        episode_length_tolerance -- simulated episode ends when gamma^n <= tolerance

    Returns an iterator of value functions representing successive updates with
    the simulated episodes according to Monte Carlo MDP prediction.
    """
    episodes: Iterator[Iterator[ReturnStep[S]]] = \
        (returns(trace, gamma, episode_length_tolerance) for trace in traces)
    f = approx_0
    counts: Mapping[NonTerminal[S], int] = {s: 0 for s in f.keys()}
    yield f

    for episode in episodes:
        tabular_mc_update(f, counts, episode, lambda n: 1. / n)
        yield f


if __name__ == '__main__':
    # compare tabular-specific implementation with standard FunctionApprox
    from rl.chapter2.simple_inventory_mrp import SimpleInventoryMRPFinite
    from rl.chapter10.prediction_utils import fmrp_episodes_stream
    import rl.iterate as iterate
    from pprint import pprint

    user_capacity = 2
    user_poisson_lambda = 1.0
    user_holding_cost = 1.0
    user_stockout_cost = 10.0

    mrp: SimpleInventoryMRPFinite = SimpleInventoryMRPFinite(
        capacity=user_capacity,
        poisson_lambda=user_poisson_lambda,
        holding_cost=user_holding_cost,
        stockout_cost=user_stockout_cost
    )
    num_episodes = 800
    gamma = 0.9
    episodes: Iterable[Iterable[TransitionStep[int]]] = fmrp_episodes_stream(mrp)
    mc_vfs: Iterator[TabularValueFunctionApprox] = \
        tabular_mc_prediction(
            traces=episodes,
            approx_0={s: 0.5 for s in mrp.non_terminal_states},
            gamma=gamma
        )
    final_mc_vf: TabularValueFunctionApprox = iterate.last(itertools.islice(mc_vfs, num_episodes))
    print(f"Equal-Weights-MC Value Function with {num_episodes:d} episodes")
    pprint({s: round(final_mc_vf[s], 3) for s in mrp.non_terminal_states})
    print("True Value Function")
    mrp.display_value_function(gamma=gamma)


