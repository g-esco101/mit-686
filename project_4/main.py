import numpy as np
import kmeans
import common
import naive_em
import em
import os


# TODO: Your code here

toy_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'toy_data.txt')
netflix_incomplete_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'netflix_incomplete.txt')
netflix_complete_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'netflix_complete.txt')


def run_kmeans_experiment():
    # load dataset
    X = np.loadtxt(toy_data_path)

    Ks = [1, 2, 3, 4]
    num_seeds = 5
    best_costs = {}

    for K in Ks:
        best_cost = float("inf")
        best_mixture = None
        best_post = None
        best_seed = None

        # 2. Try multiple random seeds for this K
        for seed in range(num_seeds):
            # Initialize mixture model (means, variances, mixing proportions)
            mixture, post = common.init(X, K, seed)

            # # Run K-means (E-step + M-step loop) starting from this initialization
            mixture, post, cost = kmeans.run(X, mixture, post)

            # Keep track of the best run for this K
            if cost < best_cost:
                best_cost = cost
                best_mixture = mixture
                best_post = post
                best_seed = seed

        best_costs[K] = best_cost

        # 3. Print the best result for this K
        title = f"k-means - K = {K}, best seed = {best_seed}, cost = {best_cost:.4f}"
        print(title)

        # 4. Plot the clustering / mixture for the best run
        common.plot(X, best_mixture, best_post, title)

    print("\nSummary of best costs - k-means:")
    for K in Ks:
        print(f"K = {K}: cost = {best_costs[K]:.4f}")


def run_naive_em_experiment():
    # load dataset
    X = np.loadtxt(toy_data_path)

    Ks = [1, 2, 3, 4]
    num_seeds = 5

    best_lls = {}
    best_bics = {}

    for K in Ks:
        best_ll = float("-inf")
        best_bic = float("-inf")
        best_mixture = None
        best_post = None
        best_seed = None

        # 2. Try multiple random seeds for this K
        for seed in range(num_seeds):
            # Initialize mixture model (means, variances, mixing proportions)
            mixture, post = common.init(X, K, seed)

            # # Run naive em (E-step + M-step loop) starting from this initialization
            mixture, post, ll = naive_em.run(X, mixture, post)
            bic = common.bic(X, mixture, ll)
            print(
                f"K={K}, seed={seed}, "
                f"ll={ll:.4f}, bic={bic:.4f}"
            )

            # EM selects the highest log-likelihood
            if ll > best_ll:
                best_ll = ll
                best_bic = bic
                best_mixture = mixture
                best_post = post
                best_seed = seed

        best_lls[K] = best_ll
        best_bics[K] = best_bic

        title = (
            f"naive EM - K={K}, best seed={best_seed}, "
            f"LL={best_ll:.4f}, BIC={best_bic:.4f}"
        )
        print(title)

        # 4. Plot the clustering / mixture for the best run
        common.plot(X, best_mixture, best_post, title)

    print("\nSummary of best results - naive em:")
    for K in Ks:
        print(
            f"K={K}: LL={best_lls[K]:.4f}, "
            f"BIC={best_bics[K]:.4f}"
        )


def run_em_experiment():
    X = np.loadtxt(netflix_incomplete_path)

    Ks = [1, 12]
    num_seeds = 5
    best_lls = {}

    for K in Ks:
        best_ll = -float("inf")
        best_seed = None

        for seed in range(num_seeds):
            mixture, post = common.init(X, K, seed)
            mixture, post, ll = em.run(X, mixture, post)

            # MAXIMIZE log-likelihood
            if ll > best_ll:
                best_ll = ll
                best_seed = seed

        best_lls[K] = best_ll
        title = f"em - K = {K}, best seed = {best_seed}, log-likelihood = {best_ll:.4f}"
        print(title)

    print("\nSummary of best log-likelihoods - em:")
    for K in Ks:
        print(f"K = {K}: log-likelihood = {best_lls[K]:.4f}")



def run_rmse_experiment():
    # Load incomplete and complete data
    X = np.loadtxt(netflix_incomplete_path)
    X_gold = np.loadtxt(netflix_complete_path)

    # Load or compute the best Gaussian mixture for K = 12
    K = 12

    best_ll = -np.inf
    best_mixture = None

    for seed in range(5):
        mixture, post = common.init(X, K, seed)
        mixture, post, ll = em.run(X, mixture, post)
        if ll > best_ll:
            best_ll = ll
            best_mixture = mixture

    # Fill the matrix
    X_pred = em.fill_matrix(X, best_mixture)

    # Compute RMSE
    rmse_val = common.rmse(X_gold, X_pred)

    print("Best K=12 log-likelihood:", best_ll)
    print("RMSE =", rmse_val)


if __name__ == "__main__":
    # run_kmeans_experiment()
    # run_naive_em_experiment()
    # run_em_experiment()
    run_rmse_experiment()