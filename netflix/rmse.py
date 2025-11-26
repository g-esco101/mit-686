import numpy as np
import common
from em import run, fill_matrix

# Load incomplete and complete data
X = np.loadtxt("netflix_incomplete.txt")
X_gold = np.loadtxt("netflix_complete.txt")

# Load or compute the best Gaussian mixture for K = 12
# (This assumes you already did the seed search and stored it)
K = 12

best_ll = -np.inf
best_mixture = None
best_post = None

for seed in range(5):
    mixture, post = common.init(X, K, seed)
    mixture, post, ll = run(X, mixture, post)
    if ll > best_ll:
        best_ll = ll
        best_mixture = mixture
        best_post = post

# Fill the matrix
X_pred = fill_matrix(X, best_mixture)

# Compute RMSE
rmse_val = common.rmse(X_gold, X_pred)

print("Best K=12 log-likelihood:", best_ll)
print("RMSE =", rmse_val)
