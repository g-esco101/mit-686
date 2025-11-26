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
    _, K = post.shape  # K = number of mixture components
    p_new = post.mean(axis=0)  # new mixture weights: average responsibility per component (shape (K,))

    mask = (X != 0)  # boolean mask of observed entries, shape (n, d)
    user_watch_count = mask.sum(axis=1)  # number of observed movies per user, shape (n,)

    mu_new = mixture.mu.copy()  # initialize new means with current values (shape (K, d))
    var_new = np.zeros(K)  # container for new variances per component (shape (K,))

    for k in range(K):
        weights = post[:, k]  # responsibilities of all users for component k (shape (n,))

        # weighted_mask: for each user u and movie j -> weight_u * observed(u,j)
        # shape (n, d). Used to compute weighted sums only over observed entries.
        weighted_mask = weights[:, None] * mask

        # numerator for mu: sum_u weight_u * x_{u,j} over observed entries, per movie j (shape (d,))
        mu_numerator = (weighted_mask * X).sum(axis=0)

        # denominator for mu: sum_u weight_u over users who observed movie j (shape (d,))
        mu_denominator = weighted_mask.sum(axis=0)

        mu = mu_new[k].copy()  # start from previous mean for component k

        # only update means for movies with enough effective weight (denominator >= 1.0)
        good = mu_denominator >= 1.0
        mu[good] = mu_numerator[good] / mu_denominator[good]  # elementwise division for observed movies
        mu_new[k] = mu  # store updated mean for component k

        # compute squared deviations only for observed entries:
        # (X - mu) broadcasted across users, then masked to ignore missing entries
        diff = (X - mu) * mask
        sse = diff ** 2  # squared errors for observed entries, shape (n, d)

        # numerator for variance: sum_u weight_u * sum_j (x_{u,j} - mu_{k,j})^2 over observed j
        var_numerator = (weights[:, None] * sse).sum()

        # denominator for variance: effective number of observed entries weighted by responsibilities
        # equals sum_u weight_u * #observed_movies_of_user_u
        var_denominator = (weights * user_watch_count).sum()

        if var_denominator > 0:
            var = var_numerator / var_denominator  # weighted average squared error
        else:
            var = mixture.var[k]  # fallback to previous variance when no data

        # enforce minimum variance to avoid numerical issues
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

        # Predict missing entries: E[x_i] = Σ_k p(k|u) μ_i^{(k)}
        for i in range(d):
            if X_complete[u, i] == 0:       # missing entry
                X_complete[u, i] = np.dot(post_u, mixture.mu[:, i])

    return X_complete