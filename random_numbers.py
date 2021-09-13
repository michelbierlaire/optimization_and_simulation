"""
Illustrations of pseudo-random number generators
Author: Michel Bierlaire
Date: Sat Sep 11 18:57:14 2021
"""
import numpy as np


class PseudoRandom:
    """Class generating pseudo-random numbers using a simple
    procedure. It is implemented as an iterable/iterator.
    """

    def __init__(self, size, seed=90267):
        """Ctor

        :param size: number of pseudo-random numbers to generate
        :type size: int

        :param seed: initial value of the sequence
        :type seed: int
        """
        self.size = size
        self.count = 0
        self.xn = seed
        self.m = 2 ** 31 - 1
        self.a = 7 ** 5

    def __iter__(self):
        """As the object is both an iterable and an iterator, it returns
        itself.
        """
        return self

    def __next__(self):
        """Generate the next number"""
        if self.count >= self.size:
            raise StopIteration
        current = self.xn / self.m
        self.xn = self.a * self.xn % self.m
        self.count += 1
        return current


class InverseTranformDiscrete:
    """Class illustrating the simulation of discrete random variable
    using the inverse transform method.
    """

    def __init__(self, pmf, size, seed=None):
        """Ctor

        :param pmf: dict where the keys are the values of the random
            variable and the values are the corresponding probability
        :type pmf: dict(int: float)

        :param size: number of draws to generate
        :type size: int

        :param seed: if different from None, seed used to initialize
           the random number generator. Default: None
        :type seed: int

        :raise ValueError: if the probabilities do not sum up to 1.0.
        """
        sum_proba = np.sum(list(pmf.values()))
        if np.abs(sum_proba - 1.0) > 1.0e-6:

            error_msg = (
                f'The probabilities do not sum up to one: '
                f'{list(pmf.values())}. Sum={sum_proba}.'
            )
            raise ValueError(error_msg)

        if seed is not None:
            np.random.seed(seed)
        self.size = size
        self.pmf = pmf
        self.count = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.count >= self.size:
            raise StopIteration
        r = np.random.rand()
        p = 0
        for value, proba in self.pmf.items():
            p += proba
            if r < p:
                self.count += 1
                return value
        raise ValueError(f'Wrong cumulative probability {p}')


class AcceptRejectDiscrete:
    """Class illustrating the simulation of discrete random variables
    using the accept-reject algorithm.

    """

    def __init__(self, pmf, draw_q, size, seed=None):
        """Ctor

        :param pmf: dict where the keys are the values of the random
            variable and the values are the corresponding probability
        :type pmf: dict(int: float)

        :param draw_q: iterator that generates draws from a random
            variable with the same values, but different pmf. It must
            implement the __next__ function and must have a pmf
            attribute, that is a dict that associates the values with
            the probabilities.
        :type draw_q: object

        :param size: number of draws to generate
        :type size: int

        :param seed: if different from None, seed used to initialize
           the random number generator. Default: None
        :type seed: int

        :raise ValueError: if the probabilities do not sum up to 1.0.
        """
        sum_proba = np.sum(list(pmf.values()))
        if np.abs(sum_proba - 1.0) > 1.0e-6:

            error_msg = (
                f'The probabilities do not sum up to one: '
                f'{list(pmf.values())}. Sum={sum_proba}.'
            )
            raise ValueError(error_msg)

        if seed is not None:
            np.random.seed(seed)
        self.size = size
        self.pmf = pmf
        self.draw_q = draw_q
        self.count = 0
        self.c = max(
            [x / y for x, y in zip(pmf.values(), draw_q.pmf.values())]
        )

    def one_draw(self):
        """Generate one draw from the random variable, using the accept-reject
        method
        """
        while True:
            next_draw_q = next(self.draw_q)
            q = self.draw_q.pmf[next_draw_q]
            p = self.pmf[next_draw_q]
            r = np.random.rand()
            if r < (p / (self.c * q)):
                return next_draw_q

    def __iter__(self):
        return self

    def __next__(self):
        if self.count >= self.size:
            raise StopIteration
        self.count += 1
        return self.one_draw()


class DiscreteFromNumpy:
    """Class illustrating the simulation of discrete random variables
    using the method implemented in numpy. For the sake of
    consistency, we give it the same structure as the other classes.
    """

    def __init__(self, pmf, size, seed=None):
        """Ctor

        :param pmf: dict where the keys are the values of the random
            variable and the values are the corresponding probability
        :type pmf: dict(int: float)

        :param size: number of draws to generate
        :type size: int

        :param seed: if different from None, seed used to initialize
           the random number generator. Default: None
        :type seed: int

        :raise ValueError: if the probabilities do not sum up to 1.0.
        """
        sum_proba = np.sum(list(pmf.values()))
        if np.abs(sum_proba - 1.0) > 1.0e-6:

            error_msg = (
                f'The probabilities do not sum up to one: '
                f'{list(pmf.values())}. Sum={sum_proba}.'
            )
            raise ValueError(error_msg)

        if seed is not None:
            np.random.seed(seed)
        self.size = size
        self.pmf = pmf
        self.count = 0
        self.draws = np.random.choice(
            list(self.pmf.keys()), p=list(self.pmf.values()), size=size
        )

    def __iter__(self):
        return self

    def __next__(self):
        if self.count >= self.size:
            raise StopIteration
        the_draw = self.draws[self.count]
        self.count += 1
        return the_draw


class InverseTranformContinuous:
    """Class illustrating the simulation of continuous random variables
    using the inverse transform method.
    """

    def __init__(self, inverse_cdf, size, seed=None):
        """Ctor

        :param inverse_cdf: inverse of the cumulative distribution function.
        :type pmf: function

        :param size: number of draws to generate
        :type size: int

        :param seed: if different from None, seed used to initialize
           the random number generator. Default: None
        :type seed: int

        :raise ValueError: if the probabilities do not sum up to 1.0.
        """
        if seed is not None:
            np.random.seed(seed)
        self.size = size
        self.inverse_cdf = inverse_cdf
        self.count = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.count >= self.size:
            raise StopIteration
        r = np.random.rand()
        p = self.inverse_cdf(r)
        self.count += 1
        return p


class AcceptRejectContinuous:
    """Class illustrating the simulation of continuous random variables
    using the accept-reject algorithm.

    """

    def __init__(self, pdf, draw_q, upper_bound, size, seed=None):
        """Ctor

        :param pdf: probability density function of the target distribution
        :type pdf: fct

        :param draw_q: iterator that generates draws from an auxiliary
            random variable with different pdf. It must implement the
            __next__ function and must have a pdf attribute, that is
            the probability density function of the random variable.
        :type draw_q: object

        :param upper_bound: constant c such that the ratio between the
            pdf of the target distribution and the auxiliary distribution
            is always lower or equal than c.
        :type upper_bound: float

        :param size: number of draws to generate
        :type size: int

        :param seed: if different from None, seed used to initialize
           the random number generator. Default: None
        :type seed: int

        """
        if seed is not None:
            np.random.seed(seed)
        self.c = upper_bound
        self.size = size
        self.pdf = pdf
        self.draw_q = draw_q
        self.count = 0

    def one_draw(self):
        """Generate one draw from the random variable, using the accept-reject
        method
        """
        while True:
            next_draw_q = next(self.draw_q)
            q = self.draw_q.pdf(next_draw_q)
            p = self.pdf(next_draw_q)
            r = np.random.rand()
            if r < (p / (self.c * q)):
                return next_draw_q

    def __iter__(self):
        return self

    def __next__(self):
        if self.count >= self.size:
            raise StopIteration
        self.count += 1
        return self.one_draw()


class MultivariateNormal:
    """Class illustrating the generation of draws from multivariate normal
    distributions, combining independent draws and the Cholesky factor
    for the variance-covariance matrix.

    """

    def __init__(self, mu, sigma, size, seed=None):
        """Ctor

        :param mu: location vector
        :type mu: numpy.array, dimension nx1

        :param sigma: variance-covariance matrix. Must be positive semi-definite.
        :type sigma: numpy.array dimension nxn

        :param size: number of draws to generate
        :type size: int

        :param seed: if different from None, seed used to initialize
           the random number generator. Default: None
        :type seed: int
        """
        if seed is not None:
            np.random.seed(seed)
        self.mu = mu
        self.n = len(mu)
        self.sigma = sigma
        self.size = size
        self.L = np.linalg.cholesky(sigma)
        self.independent_draws = np.random.normal(size=(size, self.n))
        self.count = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.count >= self.size:
            raise StopIteration
        the_draw = self.mu + self.L @ self.independent_draws[self.count]
        self.count += 1
        return the_draw
