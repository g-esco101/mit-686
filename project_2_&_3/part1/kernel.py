import numpy as np
import scipy.sparse as sparse

### Functions for you to fill in ###



def polynomial_kernel(X, Y, c, p):
    """
        Compute the polynomial kernel between two matrices X and Y::
            K(x, y) = (<x, y> + c)^p
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            c - a coefficient to trade off high-order and low-order terms (scalar)
            p - the degree of the polynomial kernel

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    # Compute (n, m) Gram matrix: A[i, j] = dot(X[i], Y[j])
    A = np.matmul(X, Y.T)

    # Evaluate polynomial kernel K(x, y) = (x·y + c)^p element-wise;
    # c shifts the decision boundary, p sets the polynomial degree
    kernel_matrix = (A + c) ** p

    return kernel_matrix



def rbf_kernel(X, Y, gamma):
    """
        Compute the Gaussian RBF kernel between two matrices X and Y::
            K(x, y) = exp(-gamma ||x-y||^2)
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            gamma - the gamma parameter of gaussian function (scalar)

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    # Compute squared Euclidean distances efficiently using vectorization
    X_norm = np.sum(X ** 2, axis=1).reshape(-1, 1)  # shape (n, 1)
    Y_norm = np.sum(Y ** 2, axis=1).reshape(1, -1)  # shape (1, m)
    sq_dists = X_norm + Y_norm - 2 * np.dot(X, Y.T)  # shape (n, m)

    # Compute the RBF kernel
    kernel_matrix = np.exp(-gamma * sq_dists)

    return kernel_matrix

def compute_probabilities(kernel_matrix, alpha, temp_parameter):
    """
    Kernelized version of softmax.compute_probabilities.

    Args:
        kernel_matrix - (n_eval, n_train) matrix where entry (i, t)
            is K(x_eval_i, x_train_t)
        alpha - (k, n_train) coefficient matrix. Row j contains the
            coefficients for class j in theta_j = sum_t alpha[j,t] phi(x_t)
        temp_parameter - softmax temperature

    Returns:
        probabilities - (k, n_eval) matrix, where probabilities[j, i]
            is P(label j | x_eval_i)
    """
    scores = np.dot(alpha, kernel_matrix.T) / temp_parameter  # (k, n_eval)
    c = np.max(scores, axis=0, keepdims=True)
    scores -= c           # numerical stability
    exp_scores = np.exp(scores)
    probabilities = exp_scores / np.sum(exp_scores, axis=0, keepdims=True)
    return probabilities


def compute_cost_function(kernel_matrix, Y, alpha, lambda_factor, temp_parameter):
    """
    Kernelized version of softmax.compute_cost_function.

    Args:
        kernel_matrix - (n_train, n_train) training kernel matrix
        Y - (n_train,) labels
        alpha - (k, n_train) coefficient matrix
        lambda_factor - L2 regularization strength
        temp_parameter - softmax temperature

    Returns:
        cost - scalar regularized softmax cost
    """
    n = kernel_matrix.shape[0]
    probabilities = compute_probabilities(kernel_matrix, alpha, temp_parameter)
    true_class_probs = probabilities[Y, np.arange(n)]
    correct_probs_log = np.log(true_class_probs + 1e-12)
    loss = -np.mean(correct_probs_log)

    # ||theta_j||^2 = alpha_j K alpha_j^T in the implicit feature space.
    reg = (lambda_factor / 2.0) * np.sum(alpha * np.dot(alpha, kernel_matrix))
    return loss + reg


def run_gradient_descent_iteration(kernel_matrix, Y, alpha, learning_rate,
                                   lambda_factor, temp_parameter):
    """
    One batch gradient-descent step for kernelized softmax regression.

    Instead of updating explicit weights theta_j in feature space, this updates
    alpha_j, where theta_j = sum_i alpha[j, i] phi(x_i).
    """
    n = kernel_matrix.shape[0]
    k = alpha.shape[0]
    probabilities = compute_probabilities(kernel_matrix, alpha, temp_parameter)
    M = sparse.coo_matrix((np.ones(n), (Y, np.arange(n))), shape=(k, n)).toarray()

    # From explicit softmax gradient:
    # theta <- (1 - lr*lambda) theta + lr/(n*T) * sum_i (M-H)_ji phi(x_i)
    # Therefore the training-example coefficients update as below.
    alpha *= (1.0 - learning_rate * lambda_factor)
    alpha += (learning_rate / (n * temp_parameter)) * (M - probabilities)
    return alpha


def softmax_regression(kernel_matrix, Y, temp_parameter, alpha, lambda_factor,
                       k, num_iterations):
    """
    Kernelized softmax regression using batch gradient descent.

    Args:
        kernel_matrix - (n_train, n_train) matrix from a kernel function
        Y - (n_train,) labels
        temp_parameter - softmax temperature
        alpha - learning rate, kept as this name to match softmax.py's API
        lambda_factor - regularization strength
        k - number of classes
        num_iterations - number of gradient-descent steps

    Returns:
        coefficients - (k, n_train) alpha/coefficient matrix
        cost_function_progression - list of costs, one per iteration
    """
    coefficients = np.zeros((k, kernel_matrix.shape[0]))
    cost_function_progression = []
    for _ in range(num_iterations):
        cost_function_progression.append(
            compute_cost_function(kernel_matrix, Y, coefficients, lambda_factor, temp_parameter)
        )
        coefficients = run_gradient_descent_iteration(
            kernel_matrix, Y, coefficients, alpha, lambda_factor, temp_parameter
        )
    return coefficients, cost_function_progression


def get_classification(kernel_matrix, alpha, temp_parameter):
    """
    Predict labels for examples represented by kernel_matrix.

    Args:
        kernel_matrix - (n_eval, n_train) matrix K(X_eval, X_train)
        alpha - (k, n_train) trained coefficient matrix
        temp_parameter - softmax temperature

    Returns:
        predicted labels - (n_eval,) NumPy array
    """
    probabilities = compute_probabilities(kernel_matrix, alpha, temp_parameter)
    return np.argmax(probabilities, axis=0)


def compute_test_error(kernel_matrix, Y, alpha, temp_parameter):
    """Return error rate for kernelized softmax predictions."""
    assigned_labels = get_classification(kernel_matrix, alpha, temp_parameter)
    return 1.0 - np.mean(assigned_labels == Y)
