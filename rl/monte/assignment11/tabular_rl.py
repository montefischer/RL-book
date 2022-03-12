import itertools
from typing import Iterator, TypeVar, Callable, Iterable, Mapping
from rl.distribution import Choose
from rl.markov_process import NonTerminal, State, ReturnStep, TransitionStep
from rl.returns import returns
from rl.function_approx import learning_rate_schedule

S = TypeVar('S')
TabularValueFunctionApprox = Mapping[NonTerminal[S], float]


def extended_vf(vf: TabularValueFunctionApprox[NonTerminal[S]], s: State[S]) -> float:
    return s.on_non_terminal(lambda x: vf[x], 0.0)


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
        traces -- iterator of simulation traces from the FMRP
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


def tabular_td_update(
        v: TabularValueFunctionApprox[S],
        counts: Mapping[NonTerminal[S], int],
        transition: TransitionStep[S],
        gamma: float,
        learning_rate: Callable[[int], float]
):
    """
    Perform a TD update of the predicted value function of a FMRP given a
    simulated transition.

    Arguments:
          v -- Current value function approximation (will be modified)
          counts -- Dictionary of how many times state S has appeared in previous traces (will be modified)
          transition -- discrete transition used to update value function approximation
          gamma -- discount rate
          learning_rate -- function giving learning rate as function of number of times a state has been updated
    """
    state, reward = transition.state, transition.reward
    counts[state] += 1
    v[state] = v[state] + learning_rate(counts[state]) \
               * (reward + gamma * extended_vf(v, transition.next_state) - v[state])
    return v, counts


def tabular_td_prediction(
        transitions: Iterable[TransitionStep],
        approx_0: TabularValueFunctionApprox,
        gamma: float,
        learning_rate: Callable[[int], float]
) -> Iterator[TabularValueFunctionApprox[S]]:
    """
    Evaluate a finite Markov Reward Process using tabular Temporal Difference. Episodes
    are simulated according to the given length tolerance. Yields approximations
    of the value function for the FMRP after updating from an additional episode.

    Arguments:
        transitions -- iterator of simulated FMRP transitions
        approx_0 -- initial value function approximation
        gamma -- discount parameter for FMRP
        learning_rate -- function giving learning rate as function of # of previous updates to state

    Returns an iterator of value functions representing successive updates with
    the simulated transitions according to TD MRP prediction.
    """
    f = approx_0
    counts = {s: 0 for s in f.keys()}
    yield f

    for transition in transitions:
        tabular_td_update(f, counts, transition, gamma, learning_rate)
        yield f


if __name__ == '__main__':
    # compare tabular-specific implementation with standard FunctionApprox
    from rl.chapter2.simple_inventory_mrp import SimpleInventoryMRPFinite
    from rl.chapter10.prediction_utils import fmrp_episodes_stream, unit_experiences_from_episodes
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
    approx_0 = {s: 0.5 for s in mrp.non_terminal_states}
    episodes: Iterable[Iterable[TransitionStep[int]]] = fmrp_episodes_stream(mrp)
    mc_vfs: Iterator[TabularValueFunctionApprox] = \
        tabular_mc_prediction(
            traces=episodes,
            approx_0=approx_0,
            gamma=gamma
        )
    final_mc_vf: TabularValueFunctionApprox = iterate.last(itertools.islice(mc_vfs, num_episodes))
    print(f"Equal-Weights-MC Value Function with {num_episodes:d} episodes")
    pprint({s: round(final_mc_vf[s], 3) for s in mrp.non_terminal_states})

    td_episode_length: int = int(round(sum(len(list(returns(
        trace=mrp.simulate_reward(Choose(mrp.non_terminal_states)),
        γ=gamma,
        tolerance=1e-6
    ))) for _ in range(num_episodes)) / num_episodes))

    td_vfs = tabular_td_prediction(
        unit_experiences_from_episodes(episodes, td_episode_length),
        approx_0=approx_0,
        gamma=gamma,
        learning_rate=learning_rate_schedule(0.03, 1000, 0.5)
    )
    final_td_vf: TabularValueFunctionApprox = iterate.last(itertools.islice(td_vfs, num_episodes*td_episode_length))
    print(f"TD Value Function with transitions drawn from {num_episodes:d} episodes")
    pprint({s: round(final_td_vf[s],3) for s in mrp.non_terminal_states})

    print("True Value Function")
    mrp.display_value_function(gamma=gamma)


