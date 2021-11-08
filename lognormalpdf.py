""" Implementation of the log of the multivariate normal pdf

Author: Michel Bierlaire
Date: Mon Nov  8 10:47:06 2021
"""
def lognormpdf(x, mu=None, S=None):
    """Calculate gaussian probability density of x, when x ~ N(mu,sigma)"""
    nx = x.size
    if mu is None:
        mu = np.array([0] * nx)
    if S is None:
        S = np.identity(nx)

    if sp.issparse(S):
        lu = spln.splu(S)
        diagL = lu.L.diagonal()
        diagU = lu.U.diagonal()
        logdet = np.log(diagL).sum() + np.log(diagU).sum()
    else:
        logdet = np.linalg.slogdet(S)[1] 
    norm_coeff = nx * np.log(2 * np.pi) + logdet

    err = x - mu
    if sp.issparse(S):
        numerator = spln.spsolve(S, err).T.dot(err)
    else:
        numerator = np.linalg.solve(S, err).T.dot(err)

    return -0.5 * (norm_coeff + numerator)

