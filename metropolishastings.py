"""
Generic implementation of the Metropolis Hastings algorithm
Author: Michel Bierlaire and Rico Krueger
Date: Thu Nov  4 15:10:11 2021
"""

# pylint: disable=invalid-name,

import numpy as np
import tqdm


def generate_draws(iterator, indicators, numberOfDraws):
    """Function that generates one sequence of draws. Implemented as a
        separate function in order to implement it for parallel
        processing (not implemented yet).

    :param iterator: iterator on the MH draws
    :type iterator: MetropolisHastingsIterator

    :param indicators: list of outputs of the simulation of interest. Are
        used to check for stationarity and calculate the effective
        number of draws.
    :type indicators: list(float = fct(state))

    :param numberOfDraws: number of draws to generate
    :type numberOfDraws: int

    :return: the generated draws
    :rtype: numpry.array
    """
    return np.array(
        [indicators(next(iterator)) for _ in range(numberOfDraws)]
    )


class MetropolisHastingsIterator:
    """Implements the Markov chain for the MH algorithm"""

    def __init__(self, initialState, userProcess, logweight):
        """Constructor

        :param initialState: the initial state of the Markov processe
        :type initialState: state

        :param userProcess: function that takes a state as input, and
            returns a new state, the forward and the backward
            transition probabilities.
        :type userProcess: state, pij, pji = fct(state)

        :param logweight: log of the unnormalized target sampling
            probability. Function that takes a state as input.
        :type logweight: float = fct(state)

        """

        self.currentState = initialState
        self.accepted = 0
        self.rejected = 0
        self.userProcess = userProcess
        self.logweight = logweight
        self.sequence = np.array([])

    def __iter__(self):
        """As the object is both an iterable and an iterator, it returns
        itself.
        """
        return self

    def getSuccessRate(self):
        """Computes the percentage of accepted draws

        :return: percentage of accepted draws
        :rtype: float
        """
        total = self.accepted + self.rejected
        if total == 0:
            return 0
        return float(self.accepted) / float(total)

    def __next__(self):
        """Generate the next state of the Markov process"""
        candidate, logqij, logqji = self.userProcess(self.currentState)
        logwi = self.logweight(self.currentState)
        logwj = self.logweight(candidate)
        logratio = logwj + logqji - logwi - logqij
        logalpha_ij = min(logratio, 0)
        r = np.random.uniform()
        if np.log(r) < logalpha_ij:
            self.currentState = candidate
            self.accepted += 1
        else:
            self.rejected += 1
        return self.currentState


class AutoCorrelation:
    """Calculates the autocorrelations defined by formula (11.7)
    in Gelman et al.
    """

    def __init__(self, draws, var):
        """Constructor

        :param draws: m arrays or n draws
        :type draws: np.array([np.array(float)])

        :param var: estimate of the posterio variance, defined by (11.3)
        :type var: float

        """
        self.m, self.n = draws.shape
        self.draws = draws
        self.var = var
        self.reset()

    def reset(self):
        """Initialize the variables at the beginning of each loop."""
        self.t = 1
        self.last_rho = None
        self.negative_autocorrelation = False

    def __iter__(self):
        """As the object is both an iterable and an iterator, it returns
        itself.
        """
        self.reset()
        return self

    def variogram(self):
        """Calculates the variogram on p. 286 for the current value of t"""
        return (
            (self.draws[:, self.t :] - self.draws[:, : (self.n - self.t)]) ** 2
        ).sum() / (self.m * (self.n - self.t))

    def __next__(self):
        """Implements one iteration

        :return: current autocorrelation
        :rtype: float

        :raise StopIteration: when the list is exhausted or negative
            autocorrelation has been detected

        """
        if self.negative_autocorrelation:
            raise StopIteration
        if self.t >= self.n:
            raise StopIteration

        rho = 1 - self.variogram() / (2 * self.var)
        if not self.t % 2:
            if rho + self.last_rho < 0:
                self.negative_autocorrelation = True
        self.last_rho = rho
        self.t += 1
        return rho


def AnalyzeDraws(draws):
    """Calculates the potential scale reduction and the effective number
    of simulation draws. See Gelman et al. Chapter 11.

    :param draws: S x R array of draws, where S is the number of
        sequences, and R the number of draws per sequence.

    :type draws: numpy.array

    :return: potential scale reduction, effective numner of draws and
        mean of the indicators
    :rtype: float, int, numpy.array(float)
    """

    nbrOfSequences, nbrOfDraws = draws.shape
    m = 2 * nbrOfSequences
    n = int(nbrOfDraws / 2)
    draws = draws.reshape(m, n)

    # The name of the variables below refer to the notations in
    # Gelman et al.

    # Means
    phi_bar_dot_j = [np.mean(d) for d in draws]
    phi_bar = np.mean(phi_bar_dot_j)
    B = np.var(phi_bar_dot_j, ddof=1) * n

    # Variances. ddof=1 means that we divide by n-1 and not by
    # n. See numpy documentation
    s_j_squared = [np.var(d, ddof=1) for d in draws]
    W = np.mean(s_j_squared)

    # Calculation of the marginal posterior variance (11.3) and
    # the potential scale reduction (11.4)
    var_plus = (n - 1) * W / n + B / n
    R_hat = np.sqrt(var_plus / W)

    # Calculation of the effective number of simulation draws
    ac = AutoCorrelation(draws, var_plus)
    neff = m * n / (1 + 2 * sum(ac))

    return R_hat, neff, phi_bar


def MetropolisHastings(
    initialStates,
    userProcess,
    weight,
    indicators,
    numberOfDraws=1000,
    maxNumberOfIterations=10,
):
    """Implements the Metropolis Hastings algorithm, checking for stationarity

    :param initialStates: list of inintial states for the sequences
    :type initialStates: list(state)

    :param userProcess: function that takes a state as input, and
        return a new state, the forward and the backward
        treansition probabilities.
    :type userProcess: state, pij, pji = fct(state)

    :param weight: unnormalized target sampling probability. Function
        that takes a state as input.
    :type weight: float = fct(state)

    :param indicators: list of outputs of the simulation of interest. Are
        used to check for stationarity and calculate the effective
        number of draws.
    :type indicators: list(float = fct(state))

    :param numberOfDraws: numberOfDraws requested by the user
    :type numberOfDraws: int

    :param maxNumberOfIterations: Draws are generated at each
        iteration until stationarity is detected. This parameter sets
        a maximum number of these iterations.
    :type maxNumberOfIterations: int

    :return: the draws, the estimated average, the status of
        stationarity and the number of iterations
    :rtype: numpy.array, float, bool, int

    """
    m = 2 * len(initialStates)
    iterators = [
        MetropolisHastingsIterator(init_state, userProcess, weight)
        for init_state in initialStates
    ]

    # Warmup
    print('Warmup')
    for iterator in iterators:
        # We first generate draws that are not stored for the warmup
        # of the Markov processes
        for _ in tqdm.tqdm(range(numberOfDraws)):
            next(iterator)

    currentNumberOfDraws = numberOfDraws
    for trials in range(maxNumberOfIterations):
        print(f'Trial {trials} with {currentNumberOfDraws} draws')
        # Generate the draws
        draws = [
            generate_draws(i, indicators, currentNumberOfDraws)
            for i in iterators
        ]
        draws = np.array(draws)
        print(f'Generated draws: {draws.shape}')

        # The dimensions of draws are: #sequences x #draws x #indicators
        # We change it to obtain: #indicators x #sequences x #draws

        draws = np.swapaxes(np.swapaxes(draws, 0, 2), 1, 2)

        # Check for stationarity for each indicator

        R_hat, neff, phi_bar = zip(
            *[AnalyzeDraws(draw_per_indicator) for draw_per_indicator in draws]
        )
        R_hat = np.array(R_hat)
        neff = np.array(neff)
        target_neff = 5 * m
        phi_bar = np.array(phi_bar)

        print(f'Potential scale reduction: {R_hat}')
        print('    should be at most 1.1')
        print(f'Effective number of simulation draws: {neff}')
        print(f'    should be at least {target_neff}')
        if np.all(R_hat <= 1.1) and np.all(neff >= target_neff):
            # We merge the draws from the various sequences before
            # returning them
            final_draws = np.array([t.flatten() for t in draws])
            return final_draws, phi_bar, True, trials + 1
        min_neff = np.min(neff)
        if min_neff >= target_neff:
            inflate = 2
        else:
            inflate = 1 + int(target_neff / min_neff)
        currentNumberOfDraws = currentNumberOfDraws * inflate
    print(
        f'Warning: the maximum number ({maxNumberOfIterations}) '
        f'of iterations has been reached before stationarity.'
    )
    # We merge the draws from the various sequences before returning
    # them.
    final_draws = np.array([t.flatten() for t in draws])
    return final_draws, phi_bar, False, maxNumberOfIterations
