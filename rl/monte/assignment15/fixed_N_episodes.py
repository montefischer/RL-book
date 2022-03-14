from typing import Sequence, Tuple, Mapping
import numpy as np
from itertools import cycle

S = str
DataType = Sequence[Sequence[Tuple[S, float]]]
ProbFunc = Mapping[S, Mapping[S, float]]
RewardFunc = Mapping[S, float]
ValueFunc = Mapping[S, float]


def get_state_return_samples(
    data: DataType
) -> Sequence[Tuple[S, float]]:
    """
    prepare sequence of (state, return) pairs.
    Note: (state, return) pairs is not same as (state, reward) pairs.
    """
    return [(s, sum(r for (_, r) in l[i:]))
            for l in data for i, (s, _) in enumerate(l)]


def get_mc_value_function(
    state_return_samples: Sequence[Tuple[S, float]]
) -> ValueFunc:
    """
    Implement tabular MC Value Function compatible with the interface defined above.
    """
    total_returns: ValueFunc = {}
    counts: Mapping[S, int] = {}
    for state, return_ in state_return_samples:
        if state not in counts.keys() or state not in total_returns.keys():
            counts[state] = 1
            total_returns[state] = 0
        total_returns[state] += return_
        counts[state] += 1
    return {s: total_returns[s] / counts[s] for s in counts.keys()}


def get_state_reward_next_state_samples(
    data: DataType
) -> Sequence[Tuple[S, float, S]]:
    """
    prepare sequence of (state, reward, next_state) triples.
    """
    return [(s, r, l[i+1][0] if i < len(l) - 1 else 'T')
            for l in data for i, (s, r) in enumerate(l)]


def get_probability_and_reward_functions(
    srs_samples: Sequence[Tuple[S, float, S]]
) -> Tuple[ProbFunc, RewardFunc]:
    """
    Implement code that produces the probability transitions and the
    reward function compatible with the interface defined above.
    """
    prob_scaffold: Mapping[S, Mapping[S, int]] = {}
    reward_scaffold: Mapping[S, Mapping[str, float]] = {}
    for state, reward, next_state in srs_samples:
        if state not in prob_scaffold.keys():
            prob_scaffold[state] = {}
            reward_scaffold[state] = {'sum': 0, 'count': 0}
        if next_state not in prob_scaffold[state].keys():
            prob_scaffold[state][next_state] = 0
        prob_scaffold[state][next_state] += 1
        reward_scaffold[state]['sum'] += reward
        reward_scaffold[state]['count'] += 1
    print("prob scaffold")
    print(prob_scaffold)
    probfunc: ProbFunc = {s: {t: prob_scaffold[s][t] / sum(prob_scaffold[s][x] for x in prob_scaffold[s])\
                              for t in prob_scaffold[s]} for s in prob_scaffold}
    rewardfunc: RewardFunc = {s: reward_scaffold[s]['sum'] / reward_scaffold[s]['count'] for s in reward_scaffold}
    return probfunc, rewardfunc


def get_mrp_value_function(
    prob_func: ProbFunc,
    reward_func: RewardFunc
) -> ValueFunc:
    """
    Implement code that calculates the MRP Value Function from the probability
    transitions and reward function, compatible with the interface defined above.
    Hint: Use the MRP Bellman Equation and simple linear algebra
    """
    sorted_keys = sorted(reward_func.keys())
    n = len(sorted_keys)
    P = np.array([[prob_func[i][j] for j in sorted_keys] for i in sorted_keys])
    r_vec = np.array([reward_func[x] for x in sorted_keys])
    v = np.linalg.inv(np.eye(n) - P) @ r_vec
    return {s: v[i] for i, s in enumerate(sorted_keys)}


def get_td_value_function(
    srs_samples: Sequence[Tuple[S, float, S]],
    num_updates: int = 300000,
    learning_rate: float = 0.3,
    learning_rate_decay: int = 30
) -> ValueFunc:
    """
    Implement tabular TD(0) (with experience replay) Value Function compatible
    with the interface defined above. Let the step size (alpha) be:
    learning_rate * (updates / learning_rate_decay + 1) ** -0.5
    so that Robbins-Monro condition is satisfied for the sequence of step sizes.
    """
    updates = 1
    vf: ValueFunc = {}

    for state, reward, next_state in cycle(srs_samples):
        alpha: float = learning_rate * (updates / learning_rate_decay + 1) ** -0.5
        if state not in vf.keys():
            vf[state] = 0
        if next_state not in vf.keys():
            vf[next_state] = 0
        vf[state] += alpha * (reward + vf[next_state] - vf[state])
        updates += 1
        if updates > num_updates:
            break
    return {s: vf[s] for s in vf.keys() if s != 'T'}


def get_lstd_value_function(
    srs_samples: Sequence[Tuple[S, float, S]]
) -> ValueFunc:
    """
    Implement LSTD Value Function compatible with the interface defined above.
    Hint: Tabular is a special case of linear function approx where each feature
    is an indicator variables for a corresponding state and each parameter is
    the value function for the corresponding state.
    """
    sorted_keys = list(sorted(set(s for s, _, _ in srs_samples)))
    n = len(sorted_keys)
    A = np.zeros((n, n))
    b = np.zeros(n)
    for state, reward, next_state in srs_samples:
        if next_state == 'T':
            continue
        vec1, vec2 = np.zeros(n), np.zeros(n)
        vec1[sorted_keys.index(state)] = 1
        vec2[sorted_keys.index(next_state)] = 1
        A += vec1.reshape((n, 1)) @ (vec1 - vec2).reshape((1,n))
        print(A)
        b += vec1 * reward
    print(A)
    opt_weights = np.linalg.inv(A) @ b
    return {state: opt_weights[i] for i, state in enumerate(sorted_keys)}


if __name__ == '__main__':
    given_data: DataType = [
        [('A', 2.), ('A', 6.), ('B', 1.), ('B', 2.)],
        [('A', 3.), ('B', 2.), ('A', 4.), ('B', 2.), ('B', 0.)],
        [('B', 3.), ('B', 6.), ('A', 1.), ('B', 1.)],
        [('A', 0.), ('B', 2.), ('A', 4.), ('B', 4.), ('B', 2.), ('B', 3.)],
        [('B', 8.), ('B', 2.)]
    ]

    sr_samps = get_state_return_samples(given_data)
    print(sr_samps)

    print("------------- MONTE CARLO VALUE FUNCTION --------------")
    print(get_mc_value_function(sr_samps))

    srs_samps = get_state_reward_next_state_samples(given_data)

    pfunc, rfunc = get_probability_and_reward_functions(srs_samps)
    print("-------------- MRP VALUE FUNCTION ----------")
    print(get_mrp_value_function(pfunc, rfunc))

    print("------------- TD VALUE FUNCTION --------------")
    print(get_td_value_function(srs_samps))

    print("------------- LSTD VALUE FUNCTION --------------")
    print(get_lstd_value_function(srs_samps))