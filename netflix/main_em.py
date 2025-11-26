import numpy as np
import em
import common
import naive_em  # will be used in later parts of the project
import em        # will be used in later parts of the project


def run_kmeans_experiment():
    X = np.loadtxt("netflix_incomplete.txt")

    Ks = [1, 12]
    num_seeds = 5
    best_lls = {}

    for K in Ks:
        best_ll = -float("inf")
        best_mixture = None
        best_post = None
        best_seed = None

        for seed in range(num_seeds):
            mixture, post = common.init(X, K, seed)
            mixture, post, ll = em.run(X, mixture, post)

            # MAXIMIZE log-likelihood
            if ll > best_ll:
                best_ll = ll
                best_mixture = mixture
                best_post = post
                best_seed = seed

        best_lls[K] = best_ll
        title = f"em - K = {K}, best seed = {best_seed}, log-likelihood = {best_ll:.4f}"
        print(title)

    print("\nSummary of best log-likelihoods - em:")
    for K in Ks:
        print(f"K = {K}: log-likelihood = {best_lls[K]:.4f}")




if __name__ == "__main__":
    run_kmeans_experiment()
