import numpy as np
# import kmeans
import naive_em as kmeans
import common
import naive_em  # will be used in later parts of the project
import em        # will be used in later parts of the project


def run_kmeans_experiment():
    # load dataset
    X = np.loadtxt("toy_data.txt")

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
            bic = common.bic(X, mixture, best_cost)
            print(f"K={K}, bic={bic:.4f}")

            # Keep track of the best run for this K
            if cost < best_cost:
                best_cost = cost
                best_mixture = mixture
                best_post = post
                best_seed = seed

        best_costs[K] = best_cost

        # 3. Print the best result for this K
        title = f"naive em - K = {K}, best seed = {best_seed}, cost = {best_cost:.4f}"
        print(title)

        # # 4. Plot the clustering / mixture for the best run
        # common.plot(X, best_mixture, best_post, title)

    print("\nSummary of best costs:")
    for K in Ks:
        print(f"K = {K}: cost = {best_costs[K]:.4f}")



if __name__ == "__main__":
    run_kmeans_experiment()
