# MIT 6.86x --- Project 4: Gaussian Mixture Models and Matrix Completion

## Overview

This project implements **Gaussian Mixture Models (GMMs)** and the
**Expectation-Maximization (EM)** algorithm, progressing from a toy
clustering problem to matrix completion on a partially observed Netflix
movie-rating dataset.

The project implements the core algorithms from scratch with NumPy and
extends EM to handle missing observations. The learned mixture model is
then used to predict missing movie ratings.

### What the project covers

-   K-means clustering as a baseline
-   Mixture models with spherical Gaussian components
-   Expectation-Maximization (E-step and M-step)
-   Numerically stable log-domain computation
-   Bayesian Information Criterion (BIC)
-   EM with partially observed data
-   Netflix rating matrix completion
-   RMSE evaluation against known ratings

## Technologies

-   **Python**
-   **NumPy** --- numerical computation and matrix operations
-   **SciPy** --- `logsumexp` for numerical stability
-   **Matplotlib** --- mixture-model and clustering visualizations
-   **Python type hints**
-   **Conda** --- development environment

The core clustering, EM, BIC, and matrix-completion logic is implemented
directly rather than using a pre-built GMM library.

------------------------------------------------------------------------

## How to Run

```bash
cd project_4
python main.py
```

Each experiment can be enabled by uncommenting the relevant call in the
`if __name__ == "__main__":` block at the bottom of `main.py`:

```python
if __name__ == "__main__":
    run_kmeans_experiment()       # K-means on toy data
    run_naive_em_experiment()     # Naive EM + BIC on toy data
    run_em_experiment()           # EM on Netflix incomplete data
    run_rmse_experiment()         # Matrix completion + RMSE evaluation
```

| Function | Data | Description |
|---|---|---|
| `run_kmeans_experiment()` | `toy_data.txt` | Runs K-means for K ∈ {1,2,3,4} over 5 seeds, prints distortion costs, and plots the best clustering per K |
| `run_naive_em_experiment()` | `toy_data.txt` | Runs naive EM for K ∈ {1,2,3,4} over 5 seeds, prints log-likelihoods and BIC scores, and plots the best mixture per K |
| `run_em_experiment()` | `netflix_incomplete.txt` | Runs EM for K ∈ {1,12} over 5 seeds and reports the best log-likelihood per K |
| `run_rmse_experiment()` | `netflix_incomplete.txt` / `netflix_complete.txt` | Trains EM with K=12, fills the incomplete Netflix matrix, and computes RMSE against the ground-truth ratings |

------------------------------------------------------------------------

## Gaussian Mixture Model

The project models observations as a mixture of `K` spherical Gaussians:

$$
P(x \mid \theta)
=
\sum_{j=1}^{K}
\pi_j
\mathcal{N}(x;\mu_j,\sigma_j^2 I)
$$

where:

-   $\pi_j$ is the mixture probability
-   $\mu_j$ is the component mean
-   $\sigma_j^2$ is the component variance
-   $K$ is the number of mixture components

------------------------------------------------------------------------

## Expectation-Maximization

EM alternates between two steps.

### E-step

The E-step computes the soft assignment of observation $u$ to component
$j$:

$$
p(j\mid u)
=
\frac{
\pi_j\mathcal{N}(x^{(u)};\mu_j,\sigma_j^2I)
}{
\sum_{k=1}^{K}
\pi_k\mathcal{N}(x^{(u)};\mu_k,\sigma_k^2I)
}
$$

Unlike K-means, these are probabilistic assignments rather than hard
cluster labels.

The implementation performs the calculation in the **log domain** and
uses `scipy.special.logsumexp` to avoid numerical underflow.

### M-step

The M-step uses the soft assignments to update:

-   component means
-   component variances
-   mixture probabilities

The implementation also enforces a minimum variance and handles
coordinates that are missing from the observations.

### Convergence

The EM implementation follows the cycle:

``` text
E-step
  ↓
Check log-likelihood improvement
  ↓
M-step
  ↓
Repeat
```

The algorithm stops when the improvement in log-likelihood is
sufficiently small.

------------------------------------------------------------------------

## Numerical Stability

The Netflix dataset is large and high-dimensional, making numerical
stability particularly important.

The implementation uses:

### Log probabilities

Products of very small probabilities are replaced with sums of log
probabilities:

$$
\log(ab)=\log(a)+\log(b)
$$

### LogSumExp

Expressions such as

$$
\log\left(\sum_j e^{f_j}\right)
$$

are computed with `scipy.special.logsumexp`.

### Minimum variance

A variance floor prevents Gaussian components from collapsing toward
zero variance.

These techniques allow EM to run reliably on the large incomplete
Netflix dataset.

------------------------------------------------------------------------

# Model Selection with BIC

Increasing the number of mixture components generally improves
likelihood, but also increases model complexity.

The project implements the Bayesian Information Criterion:

$$
BIC(M)=\ell-\frac{p}{2}\log(n)
$$

where:

-   $\ell$ is the log-likelihood
-   $p$ is the number of free parameters
-   $n$ is the number of observations

For the spherical Gaussian mixture used here:

$$
p = Kd + K + (K-1)
$$

The preferred model is the one with the **largest BIC**.

------------------------------------------------------------------------

## K-means vs. EM

The toy-data experiments evaluate:

$$
K \in \{1,2,3,4\}
$$

using seeds `0, 1, 2, 3, 4`.

### K-means results

    K   Best cost
  --- -----------
    1   5462.2975
    2   1684.9080
    3   1329.5949
    4   1035.4998

### Naive EM results

    K   Best log-likelihood         Best BIC
  --- --------------------- ----------------
    1            -1307.2234       -1315.5056
    2            -1175.7150       -1195.0401
    3            -1138.8917   **-1169.2597**
    4            -1138.6022       -1180.0132

The BIC therefore selects:

$$
\boxed{K=3}
$$

for the toy dataset.

This illustrates the difference between optimizing likelihood and
selecting a model: the likelihood improves when moving from $K=3$ to
$K=4$, but BIC penalizes the additional parameters and prefers $K=3$.

------------------------------------------------------------------------

# Handling Missing Data

The Netflix rating matrix contains many missing values, represented by
`0`.

For each user $u$:

-   $C_u$ contains the movies the user has rated.
-   $H_u$ contains the movies with missing ratings.

Missing entries are **not treated as zero-valued ratings**. They are
excluded from the Gaussian likelihood calculation.

For example:

``` text
[5, 4, 0, 0, 2]
```

is evaluated using the observed ratings:

``` text
[5, 4, 2]
```

The E-step creates an observation mask and evaluates each Gaussian only
over the observed coordinates.

------------------------------------------------------------------------

## EM with Incomplete Observations

For a partially observed user, the posterior is based only on the
observed coordinates:

$$
p(j\mid u)
=
\frac{
\pi_j
\mathcal{N}(x_{C_u}^{(u)};
\mu_{C_u}^{(j)},
\sigma_j^2I)
}{
\sum_{k=1}^{K}
\pi_k
\mathcal{N}(x_{C_u}^{(u)};
\mu_{C_u}^{(k)},
\sigma_k^2I)
}
$$

The M-step likewise updates each coordinate using only users for whom
that coordinate is observed.

This lets the model learn from incomplete rating vectors without
incorrectly interpreting missing entries as actual ratings.

------------------------------------------------------------------------

# Netflix Matrix Completion

Once the mixture model has been trained, missing ratings can be
predicted from the posterior component probabilities.

For an unobserved movie $i$:

$$
x_i^{(u)}
=
\sum_{j=1}^{K}
p(j\mid u)\mu_i^{(j)}
$$

Thus, each missing rating is a posterior-weighted average of the
corresponding component means.

------------------------------------------------------------------------

## Netflix Results

The incomplete Netflix dataset was evaluated with $K=1$ and $K=12$,
using seeds `0, 1, 2, 3, 4`.

     K   Best log-likelihood
  ---- ---------------------
     1       -1,521,060.9540
    12   **-1,390,234.4223**

The best $K=12$ model was then used to complete the missing ratings.

### Matrix-completion performance

$$
\boxed{RMSE = 0.48047}
$$

The predictions therefore differed from the known target ratings by
approximately 0.48 rating points on average in RMSE terms.

------------------------------------------------------------------------

## Implementation Structure

The project is organized around the following components:

### `common.py`

Shared data structures and utilities, including Gaussian-mixture
representation, initialization, plotting, BIC, and RMSE functionality.

### `kmeans.py`

K-means implementation used as the baseline clustering method.

### `naive_em.py`

EM implementation for the complete-data toy mixture model.

### `em.py`

Extension of EM for partially observed vectors.

Key functions include:

-   `estep()` --- computes soft assignments using observed coordinates
-   `mstep()` --- updates means, variances, and mixture probabilities
-   `run()` --- iterates E and M steps until convergence
-   `fill_matrix()` --- predicts missing matrix entries

### `main.py`

Runs the experiments and reports:

-   K-means results
-   naive EM results
-   BIC/model selection
-   Netflix EM results
-   matrix-completion predictions
-   RMSE

------------------------------------------------------------------------

## Overall Workflow

``` text
                 Input data
                     │
          ┌──────────┴──────────┐
          │                     │
       K-means             Gaussian Mixture
          │                     │
    clustering cost          E-step
                                │
                              M-step
                                │
                           convergence
                                │
                              BIC
                                │
                       Select model / K
                                │
                     Netflix incomplete data
                                │
                         Train EM (K=12)
                                │
                       Predict missing values
                                │
                              RMSE
```

------------------------------------------------------------------------

## Key Takeaways

-   **K-means** produces hard assignments, while EM produces soft
    probabilistic assignments.
-   EM estimates mixture parameters by alternating between the E-step
    and M-step.
-   Increasing `K` can improve likelihood, but BIC provides a
    model-selection criterion that penalizes additional parameters.
-   The incomplete-data EM implementation evaluates likelihoods only
    over observed coordinates.
-   Log-domain calculations and `logsumexp` are important for numerical
    stability on the Netflix dataset.
-   The learned Gaussian mixture can be used as a probabilistic
    matrix-completion model.
-   On the toy dataset, BIC selected **K = 3**.
-   On the Netflix dataset, the best **K = 12** model achieved a
    log-likelihood of approximately **-1.390 million** and an RMSE of
    approximately **0.4805**.

------------------------------------------------------------------------

## Course

This project was completed as part of **MIT 6.86x --- Machine Learning
with Python: From Linear Models to Deep Learning**.

It provides hands-on experience with unsupervised learning,
probabilistic modeling, EM optimization, model selection, numerical
stability, and recommendation-style matrix completion.
