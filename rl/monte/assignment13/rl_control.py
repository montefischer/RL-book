import itertools
from operator import itemgetter
from typing import Iterator, TypeVar, Callable, Iterable, Mapping, Tuple
from rl.distribution import Choose, FiniteDistribution, Categorical, Constant
from rl.markov_process import NonTerminal, State
from rl.markov_decision_process import FiniteMarkovDecisionProcess, TransitionStep, ReturnStep
from rl.returns import returns
from rl.function_approx import learning_rate_schedule
from rl.policy import Policy, FinitePolicy, FiniteDeterministicPolicy, UniformPolicy, RandomPolicy

S = TypeVar('S')
A = TypeVar('A')

TabularActionFunctionApprox = Mapping[NonTerminal[S], Mapping[A, float]]


def greedy_policy_from_tabular_qvf(
        q: TabularActionFunctionApprox[S, A],
        actions: Callable[[NonTerminal[S]], Iterable[A]]
) -> FinitePolicy[S, A]:
    action_for: Mapping[S, A] = {}
    for s in q.keys():
        best_value: float = float('-inf')
        best_action: A = None
        for a, value in q[s].items():
            if value > best_value:
                best_action = a
        action_for[s] = best_action
    return FinitePolicy({s: Constant(action_for[s]) for s in q.keys()})


def get_epsilon_greedy_policy(
        qvf: TabularActionFunctionApprox[S,A],
        mdp: FiniteMarkovDecisionProcess,
        epsilon: float
) -> Policy[S, A]:
    def explore(s: S, mdp=mdp) -> Iterable[A]:
        return mdp.actions(NonTerminal(s))
    return RandomPolicy(Categorical(
        {UniformPolicy(explore): epsilon,
            greedy_policy_from_tabular_qvf(qvf, mdp.actions): 1 - epsilon}
    ))


def tabular_mc_glie(
        mdp: FiniteMarkovDecisionProcess[S, A],
        initial_approx: TabularActionFunctionApprox[S, A],
        gamma: float,
        tolerance: float = 1e-6
) -> Iterator[TabularActionFunctionApprox]:
    qvf: TabularActionFunctionApprox = initial_approx
    counts: Mapping[Tuple[S, A], int] = {}
    yield qvf
    num_episodes = 1
    # generate trace experience with actions sampled from epsilon-greedy policy
    # obtained from current estimate of the Q-value function
    while True:
        eps_greedy_policy: Policy[S, A] = get_epsilon_greedy_policy(qvf, mdp, epsilon=1/num_episodes)
        episode: Iterable[TransitionStep] = mdp.simulate_actions(Choose(mdp.non_terminal_states), eps_greedy_policy)
        episode_returns: Iterable[ReturnStep] = returns(episode, gamma, tolerance)
        for step in episode_returns:
            state, action, return_ = step.state, step.action, step.return_
            if (state, action) not in counts.keys():
                counts[(state, action)] = 1
            else:
                counts[(state,action)] += 1
            qvf[state][action] += 1/counts[(state,action)] * (return_ - qvf[state][action])
        yield qvf
        num_episodes += 1
        yield


def tabular_sarsa_glie():
    pass


if __name__ == '__main__':
    # test implementations against SimpleInventoryMDPCap and AssetAllocDiscrete
    from rl.chapter3.simple_inventory_mdp_cap import SimpleInventoryMDPCap, InventoryState
    from pprint import pprint

    user_capacity = 2
    user_poisson_lambda = 1.0
    user_holding_cost = 1.0
    user_stockout_cost = 10.0

    user_gamma = 0.9

    si_mdp: FiniteMarkovDecisionProcess[InventoryState, int] = \
        SimpleInventoryMDPCap(
            capacity=user_capacity,
            poisson_lambda=user_poisson_lambda,
            holding_cost=user_holding_cost,
            stockout_cost=user_stockout_cost
        )

    state_action_reward_map = si_mdp.get_action_transition_reward_map()

    qvf: TabularActionFunctionApprox[InventoryState, int] = {NonTerminal(s): {a: 0} for s, A in state_action_reward_map.items() for a in A.keys()}
    print(qvf)

    iter = 0
    for q in tabular_mc_glie(si_mdp, qvf, user_gamma):
        iter += 1
        if iter == 100:
            break
    print(qvf)
