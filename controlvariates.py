"""
Control variates estimate of the mean
Author: Michel Bierlaire
Date: Tue Nov  2 10:54:55 2021
"""
from scipy.stats import linregress


def controlvariates(the_draws, the_control_draws, the_control_mean):
    """Calculates the control variate estimate on the mean

    :param the_draws: original draws used to calculate the mean
    :type the_draws: numpy.array

    :param the_control_draws: control draws, correlated with the
        original draws, and such that the true mean is known.
    :type the_control_draws: numpy.array

    :param the_control_mean: the true mean of the control process that
        generated the control draws.
    :type the_control_mean: float

    :return: control variates estimate of the mean
    :rtype: float
    """
    if not the_draws:
        return 0
    linregress_result = linregress(x=the_control_draws, y=the_draws)
    a = linregress_result.slope
    b = linregress_result.intercept
    return a * the_control_mean + b
