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


def tabular_update(
        v: TabularValueFunctionApprox,
        steps: Iterable[ReturnStep[S]],
        count_to_weight_func: Callable[[int], float]
) -> TabularValueFunctionApprox:
    v_next: TabularValueFunctionApprox = v
    for n, step in enumerate(steps):
        state, reward = step.state, step.reward
        v_next[state] = v[state] + count_to_weight_func(n) * (reward - v[state])
    return v_next


def tabular_mc_prediction(
        traces: Iterable[Iterable[TransitionStep]],
        approx_0: TabularValueFunctionApprox,
        gamma: float,
        episode_length_tolerance: float = 1e-6
) -> Iterator[TabularValueFunctionApprox[S]]:
    episodes: Iterator[Iterator[ReturnStep[S]]] = \
        (returns(trace, gamma, episode_length_tolerance) for trace in traces)
    f = approx_0
    yield f

    for episode in episodes:
        f = tabular_update(f, episode, lambda n: 1./n)
        yield f

if __name__ == '__main__':
    # compare tabular-specific implementation with standard FunctionApprox
    pass


