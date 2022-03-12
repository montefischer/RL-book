import itertools
from typing import Iterator, TypeVar, Callable, Iterable, Mapping
from rl.distribution import Choose
from rl.markov_process import NonTerminal, State, ReturnStep, TransitionStep
from rl.returns import returns
from rl.function_approx import learning_rate_schedule
from rl.monte.assignment11.tabular_rl import extended_vf

S = TypeVar('S')
TabularValueFunctionApprox = Mapping[NonTerminal[S], float]


def tabular_td_lambda_update(
        vf: TabularValueFunctionApprox,
        counts: Mapping[NonTerminal[S], int],
        trace: Iterable[TransitionStep],
        lambd: float,
        gamma: float,
        learning_rate: Callable[[int], float]
):
    """
    Update a tabular value function approximation using a trace experience using TD(lambda).

    Arguments:
        vf: (mutable) Tabular value function approximation to be updated in place
        counts (mutable) Dictionary to track number of times a state has previously been
            updated. Used to compute the learning rate. Updates in place.
        trace: Trace experience to update value function with
        lambd: \lambda parameter of TD(lambda), 0 <= lambd <= 1
        gamma: Discount parameter
        learning_rate: Function to compute learning rate from number of times a state has
            been encountered across all trace experiences
    """
    eligibility_traces: Mapping[NonTerminal[S], float] = {s: 0 for s in vf.keys()}
    for t, transition_step in enumerate(trace):
        state, next_state, reward = transition_step.state, transition_step.next_state, transition_step.reward
        counts[state] += 1
        for s in vf.keys():
            eligibility_traces[s] *= gamma * lambd
            eligibility_traces[s] += 1 if s == state else 0
            vf_s_update = (reward + gamma * extended_vf(vf, next_state) - vf[state]) * eligibility_traces[s]
            if counts[s] > 0:
                vf[s] += learning_rate(counts[s]) * vf_s_update


def tabular_td_lambda(
        trace_experiences: Iterable[Iterable[TransitionStep]],
        lambd: float,
        initial_vf_approx: TabularValueFunctionApprox,
        gamma: float,
        learning_rate: Callable[[int], float]
) -> Iterable[TabularValueFunctionApprox]:
    """
    Perform TD(lambda) on a tabular value function approximation using an iterable of trace experiences

    Arguments:
        trace_experiences: Series of trace experiences to be used for each value function update
        lambd: \lambda parameter of TD(lambda), 0 <= lambd <= 1
        initial_vf_approx: Starting tabular value function approximation
        gamma: Discount parameter, 0 < gamma <= 1
        learning_rate: Function to compute learning rate from number of times a state has
            been encountered across all trace experiences

    """
    assert 0 <= lambd and lambd <= 1
    v = initial_vf_approx
    counts: Mapping[NonTerminal[S], int] = {s: 0 for s in v.keys()}
    yield v

    for trace in trace_experiences:
        tabular_td_lambda_update(v, counts, trace, lambd, gamma, learning_rate)
        yield v


if __name__ == '__main__':
    pass
