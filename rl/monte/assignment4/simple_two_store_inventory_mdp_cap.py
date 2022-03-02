from dataclasses import dataclass
from typing import Tuple, Dict, Mapping
from rl.markov_decision_process import FiniteMarkovDecisionProcess
from rl.policy import FiniteDeterministicPolicy
from rl.markov_process import FiniteMarkovProcess, FiniteMarkovRewardProcess
from rl.distribution import Categorical
from scipy.stats import poisson


@dataclass(frozen=True)
class InventoryState:
    on_hand_1: int
    on_order_1: int
    on_hand_2: int
    on_order_2: int

    def inventory_position_1(self) -> int:
        return self.on_hand_1 + self.on_order_1

    def inventory_position_2(self) -> int:
        return self.on_hand_2 + self.on_order_2


InvOrderMapping = Mapping[
    InventoryState,
    Mapping[
        Tuple[int, int, int],
        Categorical[Tuple[InventoryState, float]]
    ]
]


class SimpleTwoStoreInventoryMDPCap(FiniteMarkovDecisionProcess[InventoryState, int]):

    def __init__(
        self,
        capacity_1: int,
        lambda_1: float,
        holding_cost_1: float,
        stockout_cost_1: float,
        capacity_2: int,
        lambda_2: float,
        holding_cost_2: float,
        stockout_cost_2: float,
        fixed_order_cost: float,
        fixed_transfer_cost: float
    ):
        self.capacity_1: int = capacity_1
        self.lambda_1: float = lambda_1
        self.holding_cost_1: float = holding_cost_1
        self.stockout_cost_1: float = stockout_cost_1
        self.capacity_2: int = capacity_2
        self.lambda_2: float = lambda_2
        self.holding_cost_2: float = holding_cost_2
        self.stockout_cost_2: float = stockout_cost_2

        self.fixed_order_cost: float = fixed_order_cost
        self.fixed_transfer_cost: float = fixed_transfer_cost

        self.poisson_distr_1 = poisson(lambda_1)
        self.poisson_distr_2 = poisson(lambda_2)

        super().__init__(self.get_action_transition_reward_map())

    def get_action_transition_reward_map(self) -> InvOrderMapping:
        d: Dict[InventoryState, Dict[Tuple[int,int,int], Categorical[Tuple[InventoryState,
                                                                           float]]]] = {}

        for alpha_1 in range(self.capacity_1 + 1):
            for beta_1 in range(self.capacity_1 + 1 - alpha_1):
                for alpha_2 in range(self.capacity_2 + 1):
                    for beta_2 in range(self.capacity_2 + 1 - alpha_2):
                        state: InventoryState = InventoryState(alpha_1, beta_1, alpha_2, beta_2)
                        ip_1: int = state.inventory_position_1()
                        ip_2: int = state.inventory_position_2()
                        base_reward: float = - self.holding_cost_1 * alpha_1 \
                                             - self.holding_cost_2 * alpha_2 \
                                             - self.fixed_order_cost * int(beta_1 + beta_2 > 0)
                        d1: Dict[Tuple[int, int, int], Categorical[Tuple[InventoryState, float]]] = {}

                        for order_1 in range(self.capacity_1 - ip_1 + 1):
                            for order_2 in range(self.capacity_2 - ip_2 + 1):
                                for transfer in range(-(self.capacity_2 - alpha_2 - beta_2 - order_2),
                                                      self.capacity_1 - alpha_1 - beta_1 - order_1 + 1):
                                    # transfer denotes net inventory moved locally with respect to store 1
                                    ip_1_net = ip_1 + transfer
                                    ip_2_net = ip_2 - transfer

                                    local_base_reward = base_reward - self.fixed_transfer_cost * int(transfer > 0)
                                    # simulate poisson demand
                                    sr_probs_dict: Dict[Tuple[InventoryState, float], float] = \
                                        {
                                            (
                                                InventoryState(ip_1_net - i, order_1, ip_2_net - j, order_2),
                                                local_base_reward
                                            ): self.poisson_distr_1.pmf(i) * self.poisson_distr_2.pmf(j)
                                            for i in range(ip_1_net) for j in range(ip_2_net)
                                        }

                                    # account for poisson demand overwhelming supply at store 1 or 2 or both
                                    probability_1: float = 1 - self.poisson_distr_1.cdf(ip_1_net - 1)
                                    reward_1: float = local_base_reward - self.stockout_cost_1\
                                                       * (probability_1 * self.lambda_1 - ip_1_net\
                                                       + ip_1_net * self.poisson_distr_1.pmf(ip_1_net))
                                    for j in range(ip_2_net):
                                        sr_probs_dict[(InventoryState(0, order_1, ip_2_net - j, order_2), reward_1)] = \
                                            probability_1 * self.poisson_distr_2.pmf(j)

                                    probability_2: float = 1 - self.poisson_distr_2.cdf(ip_2_net - 1)
                                    reward_2: float = local_base_reward - self.stockout_cost_2\
                                                       * (probability_2 * self.lambda_2 - ip_2_net\
                                                       + ip_2_net * self.poisson_distr_2.pmf(ip_2_net))
                                    for i in range(ip_1_net):
                                        sr_probs_dict[(InventoryState(ip_1_net - i, order_1, 0, order_2), reward_2)] = \
                                            probability_2 * self.poisson_distr_1.pmf(i)

                                    sr_probs_dict[(InventoryState(0, order_1, 0, order_2),
                                                   reward_1 + reward_2 - local_base_reward)]\
                                        = probability_1 * probability_2
                        d[state] = d1
        return d

# todo: write tests