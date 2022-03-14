import numpy as np
from rl.function_approx import LinearFunctionApprox
from rl.markov_process import NonTerminal
from rl.td import least_squares_td, least_squares_policy_iteration
from rl.chapter8.optimal_exercise_bin_tree import OptimalExerciseBinTree
from rl.chapter12.optimal_exercise_rl import OptimalExerciseRL, fitted_lspi_put_option
from typing import Callable, Sequence, Tuple, List

TrainingDataType = Tuple[int, float, float]

if __name__ == '__main__':
    # LSPI for American Options Pricing
    import matplotlib.pyplot as plt
    from rl.gen_utils.plot_funcs import plot_list_of_curves
    spot_price_val: float = 100.0
    strike: float = 100.0
    expiry_val: float = 1.0
    rate_val: float = 0.05
    vol_val: float = 0.25
    num_steps_val: int = 200

    opt_ex_bin_tree: OptimalExerciseBinTree = OptimalExerciseBinTree(
        spot_price=spot_price_val,
        payoff=lambda _, x: max(strike - x, 0),
        expiry=expiry_val,
        rate=rate_val,
        vol=vol_val,
        num_steps=num_steps_val
    )

    vf_seq, policy_seq = zip(*opt_ex_bin_tree.get_opt_vf_and_policy())
    ex_boundary: Sequence[Tuple[float, float]] = \
        opt_ex_bin_tree.option_exercise_boundary(policy_seq, False)
    time_pts, ex_bound_pts = zip(*ex_boundary)

    label = ("Binomail Tree Put Option Exercise Boundary")
    plot_list_of_curves(
        list_of_x_vals=[time_pts],
        list_of_y_vals=[ex_bound_pts],
        list_of_colors=["b"],
        list_of_curve_labels=[label],
        x_label="Time",
        y_label="Underlying Price",
        title=label
    )

    am_price: float = vf_seq[0][NonTerminal(0)]
    print(f"Binomial Tree American Price = {am_price:.3f}")

    dt: float = expiry_val / num_steps_val

    num_training_paths: int = 5000
    spot_price_frac_val: float = 0.0

    lspi_training_iters: int = 8

    split_val: int = 1000

    num_scoring_paths: int = 10000

    def payoff_func(_: float, s: float) -> float:
        return max(strike - s, 0.)

    opt_ex_rl: OptimalExerciseRL = OptimalExerciseRL(
        spot_price=spot_price_val,
        payoff=payoff_func,
        expiry=expiry_val,
        rate=rate_val,
        vol=vol_val,
        num_steps=num_steps_val
    )

    european_price: float = opt_ex_rl.european_put_price(strike)
    print(f"European Price = {european_price:.3f}")

    training_data: Sequence[TrainingDataType] = opt_ex_rl.training_sim_data(
        num_paths=num_training_paths,
        spot_price_frac=spot_price_frac_val
    )
    print("Generated Training Data")

    flspi: LinearFunctionApprox[Tuple[float, float]] = fitted_lspi_put_option(
        obj=opt_ex_rl,
        strike=strike,
        expiry=expiry_val,
        training_data=training_data,
        training_iters=lspi_training_iters,
        split=split_val
    )

    print("Fitted LSPI Model")

    for step in [0, int(num_steps_val / 2), num_steps_val - 1]:
        prices: np.ndarray = np.arange(120.0)
        exer_curve: np.ndarray = opt_ex_rl.exercise_curve(
            step=step,
            prices=prices
        )
        cont_curve_lspi: np.ndarray = opt_ex_rl.continuation_curve(
            func=flspi,
            step=step,
            prices=prices
        )
        plt.plot(
            prices,
            cont_curve_lspi,
            "r",
            prices,
            exer_curve,
            "b"
        )
        time: float = step * expiry_val / num_steps_val
        plt.title(f"LSPI Curve for Time = {time:.3f}")
        plt.show()

    ex_boundary_lspi: Sequence[float] = opt_ex_rl.put_option_exercise_boundary(
        func=flspi,
        strike=strike
    )
    time_pts: Sequence[float] = [i * dt for i in range(num_steps_val + 1)]
    plt.plot(time_pts, ex_boundary_lspi, "r")
    plt.title("LSPI Exercise Boundary")
    plt.show()

    scoring_data: np.ndarray = opt_ex_rl.scoring_sim_data(
        num_paths=num_scoring_paths
    )

    lspi_opt_price: float = opt_ex_rl.option_price(
        scoring_data=scoring_data,
        func=flspi
    )
    print(f"LSPI Option Price = {lspi_opt_price:.3f}")
