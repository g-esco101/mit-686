"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    n, d = X.shape
    K, _ = mixture.mu.shape
    post = np.zeros((n, K))
    log_likelihood = 0.0

    for u in range(n):
        user = X[u]
        mask = (user != 0)
        observed = user[mask]
        observed_count = observed.size
        func = np.zeros(K)

        for k in range(K):
            mu = mixture.mu[k, mask]
            var = mixture.var[k]

            # squared distance over movies watched
            diff = observed - mu
            sq_dist = np.dot(diff, diff)

            # log pi (with small epsilon for stability as hint suggests)
            log_pi = np.log(mixture.p[k] + 1e-16)

            # log gaussian over movies watched
            # -|C_u|/2 * log(2*pi*var) - (1/(2*var)) * sum (x - mu)^2
            if observed_count > 0:
                log_gauss = -0.5 * observed_count * np.log(2*np.pi * var) \
                            - 0.5 * sq_dist / var
            else:
                # No observed entries: Gaussian term is 0 (empty product)
                log_gauss = 0.0

            func[k] = log_pi + log_gauss

        # log p(x^{(u)} | theta) = logsumexp_k f(u, k)
        log_func = logsumexp(func)
        log_likelihood += log_func

        # responsibilities: p(k | u) = exp(f(u,k) - logsumexp)
        post[u, :] = np.exp(func - log_func)

    return post, log_likelihood


def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    # number of mixture components
    _, K = post.shape

    # new mixture weights (shape (K,))
    p_new = post.mean(axis=0)

    # boolean mask of observed entries (n, d)
    mask = (X != 0)

    # number of observed movies per user (n,)
    user_watch_count = mask.sum(axis=1)

    # initialize new means and variances
    mu_new = mixture.mu.copy()
    var_new = np.zeros(K)

    for k in range(K):
        # responsibilities of all users for component k (n,)
        weights = post[:, k]

        # weighted mask used to sum only over observed entries (n, d)
        weighted_mask = weights[:, None] * mask

        # weighted sums and counts per movie (d,)
        mu_numerator = (weighted_mask * X).sum(axis=0)
        mu_denominator = weighted_mask.sum(axis=0)

        # start from previous mean for component k
        mu = mu_new[k].copy()

        # update means only where denominator is sufficiently large
        good = mu_denominator >= 1.0
        mu[good] = mu_numerator[good] / mu_denominator[good]
        mu_new[k] = mu

        # squared errors computed only on observed entries (n, d)
        diff = (X - mu) * mask
        sse = diff ** 2

        # weighted variance numerator and denominator
        var_numerator = (weights[:, None] * sse).sum()
        var_denominator = (weights * user_watch_count).sum()

        # fallback to current variance if denominator is zero
        if var_denominator > 0:
            var = var_numerator / var_denominator
        else:
            var = mixture.var[k]

        # enforce minimum variance
        var_new[k] = max(var, min_variance)

    return GaussianMixture(mu_new, var_new, p_new)


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        z: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    log_likelihood_prev = None

    while True:
        # e-step
        post, log_likelihood = estep(X, mixture)

        # m-step
        mixture = mstep(X, post, mixture)

        # Convergence check
        if log_likelihood_prev is not None:
            if log_likelihood - log_likelihood_prev <= 1e-6 * abs(log_likelihood):
                break

        log_likelihood_prev = log_likelihood

    return mixture, post, log_likelihood

def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    n, d = X.shape
    K, _ = mixture.mu.shape
    X_complete = X.copy()

    for u in range(n):
        user = X[u]
        mask = (user != 0)
        movies_observed = user[mask]
        movies_observed_count = movies_observed.size

        # Compute f(u,k) for all clusters
        f_u = np.zeros(K)

        for k in range(K):
            mu = mixture.mu[k, mask]
            var = mixture.var[k]

            # If user has at least one observed entry
            if movies_observed_count > 0:
                diff = movies_observed - mu
                sse = np.dot(diff, diff)
                log_gauss = -0.5 * movies_observed_count * np.log(2.0 * np.pi * var) \
                            - 0.5 * sse / var
            else:
                # No observed entries → uniform likelihood across clusters
                log_gauss = 0.0

            log_pi = np.log(mixture.p[k] + 1e-16)
            f_u[k] = log_pi + log_gauss

        # soft assignments p(k|u)
        log_px = logsumexp(f_u)
        post_u = np.exp(f_u - log_px)   # shape (K,)

        for i in range(d):
            if X_complete[u, i] == 0:
                X_complete[u, i] = np.dot(post_u, mixture.mu[:, i])

    return X_complete