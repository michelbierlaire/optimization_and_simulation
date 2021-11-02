"""
Use of bootstrapping to calculate Monte Carlo integrals
Author: Michel Bierlaire
Date: Mon Nov  1 17:05:43 2021
"""

import numpy as np


def bootstrap(the_function, uniform_draws, sample_size=100, CI_size=0.9):
    """Calculate an indicator using simulation, and uses bootstrapping to
    calculate the Mean Squared Error (error) and a confidence
    interval.

    :param the_function: function that take draws as an arguments, and
        returns an indicator calculate from these draws.
    :type the_function: fct(np.array)

    :param uniform_draws: draws to use for the simulation
    :type uniform_draws: np.array

    :param sample_size: number of bootstrap draws to perform
    :type sample_size: int

    :param CI_size: size of the confidence interval
    :type CI_size: float

    :return: the estimate from simulation, MSI, the lower and the
        upper bound of the confidence interval.
    :rtype: tuple(float, float, float, float)
    """
    result = the_function(uniform_draws)
    bootstrap_results = np.array(
        [
            the_function(
                np.random.choice(uniform_draws, size=uniform_draws.size)
            )
            for _ in range(sample_size)
        ]
    )

    the_mse = np.mean((bootstrap_results - result) ** 2)
    CI_lower_bound = (1 - CI_size) / 2
    CI_upper_bound = (1 + CI_size) / 2
    the_CI_low = np.quantile(bootstrap_results, CI_lower_bound)
    the_CI_high = np.quantile(bootstrap_results, CI_upper_bound)
    return result, the_mse, the_CI_low, the_CI_high
