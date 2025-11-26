import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("..")
from utils import *
from linear_regression import *
from svm import *
from softmax import *
from features import *
from kernel import *

from sklearn.svm import SVC


#######################################################################
# 1. Introduction
#######################################################################

# Load MNIST data:
train_x, train_y, test_x, test_y = get_MNIST_data()
# Plot the first 20 images of the training set.
plot_images(train_x[0:20, :])

#######################################################################
# 2. Linear Regression with Closed Form Solution
#######################################################################

# TODO: first fill out functions in linear_regression.py, otherwise the functions below will not work


def run_linear_regression_on_MNIST(lambda_factor=1):
    """
    Trains linear regression, classifies test data, computes test error on test set

    Returns:
        Final test error
    """
    train_x, train_y, test_x, test_y = get_MNIST_data()
    train_x_bias = np.hstack([np.ones([train_x.shape[0], 1]), train_x])
    test_x_bias = np.hstack([np.ones([test_x.shape[0], 1]), test_x])
    theta = closed_form(train_x_bias, train_y, lambda_factor)
    test_error = compute_test_error_linear(test_x_bias, test_y, theta)
    return test_error


# Don't run this until the relevant functions in linear_regression.py have been fully implemented.
# print('Linear Regression test_error =', run_linear_regression_on_MNIST(lambda_factor=1))

#######################################################################
# 3. Support Vector Machine
#######################################################################

# TODO: first fill out functions in svm.py, or the functions below will not work

def run_svm_one_vs_rest_on_MNIST():
    """
    Trains svm, classifies test data, computes test error on test set

    Returns:
        Test error for the binary svm
    """
    train_x, train_y, test_x, test_y = get_MNIST_data()
    train_y[train_y != 0] = 1
    test_y[test_y != 0] = 1
    pred_test_y = one_vs_rest_svm(train_x, train_y, test_x)
    test_error = compute_test_error_svm(test_y, pred_test_y)
    return test_error


# print('SVM one vs. rest test_error:', run_svm_one_vs_rest_on_MNIST())


def run_multiclass_svm_on_MNIST():
    """
    Trains svm, classifies test data, computes test error on test set

    Returns:
        Test error for the binary svm
    """
    train_x, train_y, test_x, test_y = get_MNIST_data()
    pred_test_y = multi_class_svm(train_x, train_y, test_x)
    test_error = compute_test_error_svm(test_y, pred_test_y)
    return test_error


# print('Multiclass SVM test_error:', run_multiclass_svm_on_MNIST())

#######################################################################
# 4. Multinomial (Softmax) Regression and Gradient Descent
#######################################################################

# TODO: first fill out functions in softmax.py, or run_softmax_on_MNIST will not work


def run_softmax_on_MNIST(temp_parameter=1):
    """
    Trains softmax, classifies test data, computes test error, and plots cost function

    Runs softmax_regression on the MNIST training set and computes the test error using
    the test set. It uses the following values for parameters:
    alpha = 0.3
    lambda = 1e-4
    num_iterations = 150

    Saves the final theta to ./theta.pkl.gz

    Returns:
        Final test error
    """
    train_x, train_y, test_x, test_y = get_MNIST_data()
    theta, cost_function_history = softmax_regression(train_x, train_y, temp_parameter, alpha=0.3, lambda_factor=1.0e-4, k=10, num_iterations=150)
    plot_cost_function_over_time(cost_function_history)
    test_error = compute_test_error(test_x, test_y, theta, temp_parameter)
    # Save the model parameters theta obtained from calling softmax_regression to disk.
    write_pickle_data(theta, "./theta.pkl.gz")

    # TODO: add your code here for the "Using the Current Model" question in tab 6.
    #      and print the test_error_mod3
    return test_error


# print('softmax test_error=', run_softmax_on_MNIST(temp_parameter=1))

# TODO: Find the error rate for temp_parameter = [.5, 1.0, 2.0]
#      Remember to return the tempParameter to 1, and re-run run_softmax_on_MNIST

#######################################################################
# 6. Changing Labels
#######################################################################



def run_softmax_on_MNIST_mod3(temp_parameter=1):
    """
    Trains Softmax regression on digit (mod 3) classifications.

    See run_softmax_on_MNIST for more info.
    """
    # train_x, train_y, test_x, test_y = get_MNIST_data()
    # theta, cost_function_history = softmax_regression(train_x, train_y, temp_parameter, alpha=0.3, lambda_factor=1.0e-4, k=10, num_iterations=150)
    # plot_cost_function_over_time(cost_function_history)
    # test_error = compute_test_error_mod3(test_x, test_y, theta, temp_parameter)
    # # Save the model parameters theta obtained from calling softmax_regression to disk.
    # write_pickle_data(theta, "./theta.pkl.gz")
    # return test_error


    train_x, train_y, test_x, test_y = get_MNIST_data()

    # Convert labels to mod 3 for training (3 classes)
    train_y_mod3 = train_y % 3

    # Train a k=3 softmax model on the mod-3 labels
    theta, cost_function_history = softmax_regression(
        train_x,
        train_y_mod3,
        temp_parameter,
        alpha=0.3,
        lambda_factor=1e-4,
        k=3,                   # <-- 3 classes, not 10
        num_iterations=150
    )

    # (Optional) plot; some graders prefer you skip plotting
    # plot_cost_function_over_time(cost_function_history)

    # Evaluate with the provided mod-3 error function
    test_error = compute_test_error_mod3(test_x, test_y, theta, temp_parameter)

    # Save parameters (now shape 3 x d)
    write_pickle_data(theta, "./theta.pkl.gz")
    return test_error

# TODO: Run run_softmax_on_MNIST_mod3(), report the error rate

# print('softmax mod 3 test_error=', run_softmax_on_MNIST_mod3(temp_parameter=1))


def run_softmax_on_MNIST_PCA18(temp_parameter=1.0):
    # 1) Load data
    train_x, train_y, test_x, test_y = get_MNIST_data()   # shapes: (n,d), d=784

    # 2) Fit PCA on the TRAINING data only
    pcs = principal_components(train_x)                   # (d,d), cols are unit eigenvectors
    feature_means = np.mean(train_x, axis=0)              # (d,)

    # 3) Project train/test onto first 18 PCs using TRAIN means
    train_proj = project_onto_PC(train_x, pcs, 18, feature_means)  # (n,18)
    test_proj  = project_onto_PC(test_x,  pcs, 18, feature_means)  # (m,18)

    # 4) Train softmax on 10 classes with projected features
    theta, _ = softmax_regression(
        train_proj, train_y, temp_parameter,
        alpha=0.3, lambda_factor=1e-4, k=10, num_iterations=150
    )

    # 5) Evaluate on projected test set
    return compute_test_error(test_proj, test_y, theta, temp_parameter)

# print('softmax PCA 18 test_error=', run_softmax_on_MNIST_PCA18(temp_parameter=1))


def run_softmax_on_MNIST_PCA10(temp_parameter=1.0):
    # 1) Load data
    train_x, train_y, test_x, test_y = get_MNIST_data()   # shapes: (n,d), d=784

    # 2) Fit PCA on the TRAINING data only
    train_x = cubic_features(train_x)
    pcs = principal_components(train_x)                   # (d,d), cols are unit eigenvectors
    feature_means = np.mean(train_x, axis=0)              # (d,)

    # 3) Project train/test onto first 18 PCs using TRAIN means
    train_proj = project_onto_PC(train_x, pcs, 10, feature_means)  # (n,18)
    test_proj  = project_onto_PC(test_x,  pcs, 10, feature_means)  # (m,18)

    # 4) Train softmax on 10 classes with projected features
    theta, _ = softmax_regression(
        train_proj, train_y, temp_parameter,
        alpha=0.3, lambda_factor=1e-4, k=10, num_iterations=150
    )

    # 5) Evaluate on projected test set
    return compute_test_error(test_proj, test_y, theta, temp_parameter)


# print('softmax PCA 10 test_error=', run_softmax_on_MNIST_PCA10(temp_parameter=1))


def run_softmax_on_MNIST_cubic_PCA10(temp_parameter=1.0):
    """
    Train/evaluate softmax on cubic features of 10-D PCA representation.
    Steps:
      - PCA to 10 dims (train-only fit)
      - Expand with cubic_features (your features.py)
      - Train softmax on 10 classes
    Expect test error ~0.08 (± a bit).
    """
    # 1) Load data
    train_x, train_y, test_x, test_y = get_MNIST_data()

    # 2) Fit PCA on TRAIN ONLY
    centered_train_x, feature_means = center_data(train_x)
    pcs = principal_components(centered_train_x)

    # 3) Project to 10 PCs
    train_pca10 = project_onto_PC(train_x, pcs, n_components=10, feature_means=feature_means)
    test_pca10  = project_onto_PC(test_x,  pcs, n_components=10, feature_means=feature_means)

    # 4) Expand with explicit cubic map from features.py
    train_cubic = cubic_features(train_pca10)  # shape (n, 286)
    test_cubic  = cubic_features(test_pca10)   # shape (m, 286)

    # 5) Train softmax
    theta, _ = softmax_regression(
        train_cubic, train_y, temp_parameter,
        alpha=0.3, lambda_factor=1e-4, k=10, num_iterations=150
    )

    # 6) Evaluate
    return compute_test_error(test_cubic, test_y, theta, temp_parameter)

# print('softmax cubic PCA 10 test_error=', run_softmax_on_MNIST_cubic_PCA10(temp_parameter=1))


def run_cubic_poly_svm_on_MNIST_PCA10():
    """
    Train an SVM with a cubic polynomial kernel on 10-D PCA features of MNIST.
    Expect test error ≈ 0.06.
    """

    # 1) Load MNIST data
    train_x, train_y, test_x, test_y = get_MNIST_data()

    # 2) Fit PCA on training data only
    centered_train_x, feature_means = center_data(train_x)
    pcs = principal_components(centered_train_x)

    # 3) Project both train and test to 10-D using training means
    train_pca10 = project_onto_PC(train_x, pcs, n_components=10, feature_means=feature_means)
    test_pca10  = project_onto_PC(test_x,  pcs, n_components=10, feature_means=feature_means)

    # 4) Train cubic polynomial SVM
    svm = SVC(
        kernel='poly',
        degree=3,
        random_state=0,    # ensure reproducibility
        gamma='scale',     # default scaling
        coef0=1.0,         # include bias term (same as +1 in (x·x'+1)^3)
    )

    print("Training cubic polynomial SVM...")
    svm.fit(train_pca10, train_y)

    # 5) Compute accuracy and error on test set
    test_accuracy = svm.score(test_pca10, test_y)
    test_error = 1 - test_accuracy

    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test error: {test_error:.4f}")

    return test_error

print('softmax cubic poly svm PCA 10 test_error=', run_cubic_poly_svm_on_MNIST_PCA10())



def run_rbf_svm_on_MNIST_PCA10():
    """
    Train an SVM with RBF kernel on 10-D PCA features of MNIST and return test error.
    Uses random_state=0 and default values for other SVC params.
    Expected test error ≈ 0.05.
    """
    # 1) Load data
    train_x, train_y, test_x, test_y = get_MNIST_data()

    # 2) Fit PCA on training data only
    centered_train_x, feature_means = center_data(train_x)
    pcs = principal_components(centered_train_x)  # columns are eigenvectors

    # 3) Project train/test to 10 PCs using training means and PCs
    train_pca10 = project_onto_PC(train_x, pcs, n_components=10, feature_means=feature_means)
    test_pca10  = project_onto_PC(test_x,  pcs, n_components=10, feature_means=feature_means)

    # 4) Train RBF SVM (default gamma='scale', C=1.0)
    clf = SVC(kernel='rbf', random_state=0)
    clf.fit(train_pca10, train_y)

    # 5) Evaluate
    test_accuracy = clf.score(test_pca10, test_y)
    test_error = 1.0 - test_accuracy
    print(f"RBF SVM — test accuracy: {test_accuracy:.4f}, test error: {test_error:.4f}")
    return test_error


print('rbf svm PCA 10 test_error=', run_rbf_svm_on_MNIST_PCA10())

#######################################################################
# 7. Classification Using Manually Crafted Features
#######################################################################

## Dimensionality reduction via PCA ##

# TODO: First fill out the PCA functions in features.py as the below code depends on them.


n_components = 18

###Correction note:  the following 4 lines have been modified since release.
train_x_centered, feature_means = center_data(train_x)
pcs = principal_components(train_x_centered)
train_pca = project_onto_PC(train_x, pcs, n_components, feature_means)
test_pca = project_onto_PC(test_x, pcs, n_components, feature_means)

# train_pca (and test_pca) is a representation of our training (and test) data
# after projecting each example onto the first 18 principal components.


# TODO: Train your softmax regression model using (train_pca, train_y)
#       and evaluate its accuracy on (test_pca, test_y).


# TODO: Use the plot_PC function in features.py to produce scatterplot
#       of the first 100 MNIST images, as represented in the space spanned by the
#       first 2 principal components found above.
plot_PC(train_x[range(000, 100), ], pcs, train_y[range(000, 100)], feature_means)#feature_means added since release


# TODO: Use the reconstruct_PC function in features.py to show
#       the first and second MNIST images as reconstructed solely from
#       their 18-dimensional principal component representation.
#       Compare the reconstructed images with the originals.
firstimage_reconstructed = reconstruct_PC(train_pca[0, ], pcs, n_components, train_x, feature_means)#feature_means added since release
plot_images(firstimage_reconstructed)
plot_images(train_x[0, ])

secondimage_reconstructed = reconstruct_PC(train_pca[1, ], pcs, n_components, train_x, feature_means)#feature_means added since release
plot_images(secondimage_reconstructed)
plot_images(train_x[1, ])


## Cubic Kernel ##
# TODO: Find the 10-dimensional PCA representation of the training and test set


# TODO: First fill out cubicFeatures() function in features.py as the below code requires it.

train_cube = cubic_features(train_pca10)
test_cube = cubic_features(test_pca10)
# train_cube (and test_cube) is a representation of our training (and test) data
# after applying the cubic kernel feature mapping to the 10-dimensional PCA representations.


# TODO: Train your softmax regression model using (train_cube, train_y)
#       and evaluate its accuracy on (test_cube, test_y).
