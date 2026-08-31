"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture



def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    n, d = X.shape
    K, _ = mixture.mu.shape
    post = np.zeros((n, K))
    const = (2 * np.pi) ** (-d / 2)

    # calculate probabilities
    for j in range(K):
        difference = X - mixture.mu[j]  # (n, d)
        exponent = -0.5 * np.sum(difference ** 2, axis=1) / mixture.var[j]
        gaussian = const * mixture.var[j] ** (-d / 2) * np.exp(exponent)
        post[:, j] = mixture.p[j] * gaussian

    # normalization
    post_sum = np.sum(post, axis=1, keepdims=True)
    post /= post_sum

    # calculate log-likelihood
    log_likelihood = np.sum(np.log(post_sum))

    return post, log_likelihood


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    _, K = post.shape
    n_hat = post.sum(axis=0)
    p = n_hat / n
    mu = np.zeros((K, d))
    var = np.zeros(K)

    # calculate Gaussian Parameters
    for j in range(K):
        mu[j, :] = post[:, j] @ X / n_hat[j]
        sse = ((mu[j] - X)**2).sum(axis=1) @ post[:, j]
        var[j] = sse / (d * n_hat[j])

    return GaussianMixture(mu, var, p)

def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    log_likelihood = None

    while True:
        post, new_log_likelihood = estep(X, mixture)
        # check convergence
        if log_likelihood is not None:
            if new_log_likelihood - log_likelihood <= 1e-6 * abs(new_log_likelihood):
                break
        log_likelihood = new_log_likelihood
        mixture = mstep(X, post)

    return mixture, post, log_likelihood