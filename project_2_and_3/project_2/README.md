# MIT 6.86x — Project 2: MNIST Digit Classification

## Overview

This project implements and compares a suite of supervised learning algorithms
for classifying handwritten digits from the
[MNIST dataset](http://yann.lecun.com/exdb/mnist/). Every algorithm is built
from scratch in NumPy and evaluated end-to-end, progressing from simple linear
regression through kernelized SVMs. Feature engineering techniques — including
**Principal Component Analysis (PCA)** and an explicit **cubic feature map** —
are used to reduce dimensionality and improve accuracy.

This is **Project 2** of
[MIT 6.86x: Machine Learning with Python — From Linear Models to Deep Learning](https://www.edx.org/course/machine-learning-with-python-from-linear-models-to).

---

## Dataset

**MNIST** — 70,000 grayscale images of handwritten digits (0–9).

| Split      | Samples | Shape      |
|------------|---------|------------|
| Train + Val | 60,000 | (60000, 784) |
| Test        | 10,000 | (10000, 784) |

Each image is **28×28 pixels**, flattened to a 784-dimensional feature vector
with pixel values normalised to `[0, 1]`. The dataset is stored as a compressed
pickle file at `../Datasets/mnist.pkl.gz`.

---

## Project Structure

```
project_2/
├── main.py               # Driver: runs every experiment end-to-end
├── linear_regression.py  # Closed-form L2-regularised linear regression
├── svm.py                # Binary (one-vs-rest) and multiclass linear SVM
├── softmax.py            # Multinomial softmax regression (gradient descent)
├── kernel.py             # Polynomial & RBF kernels + kernelised softmax
├── features.py           # PCA, cubic feature map, reconstruction & plotting
├── utils.py              # Data loading, preprocessing, and visualisation
├── cubic_features_checker.py  # Unit tests for the cubic feature expansion
└── test.py               # Project test suite
```

---

## Algorithms & Experiments

### 1. Linear Regression (`linear_regression.py`)
Treats digit labels as continuous values and solves directly via the
**closed-form L2-regularised normal equation**:

$$\theta = (X^\top X + \lambda I)^{-1} X^\top Y$$

Predictions are rounded to the nearest integer and clipped to `[0, 9]`.

---

### 2. Support Vector Machine (`svm.py`)
Two SVM variants are trained using `sklearn.svm.LinearSVC` (C = 0.1):

| Mode | Description |
|------|-------------|
| **One-vs-Rest** | Binary classifier — digit `0` vs. all others |
| **Multiclass** | Full 10-class classification using one-vs-rest internally |

---

### 3. Softmax (Multinomial Logistic) Regression (`softmax.py`)
A full implementation of softmax regression trained with **batch gradient
descent**, including:

- **Temperature scaling** — a `temp_parameter` controls the sharpness of the
  softmax distribution, explored at τ ∈ {0.5, 1.0, 2.0}
- **L2 regularisation** (`lambda_factor`)
- Numerically stable probability computation (subtracts column-wise max)
- Sparse one-hot indicator matrix for efficient gradient construction
- **Mod-3 label transformation** — remaps digit labels to `{0, 1, 2}` (digit
  mod 3) to train a 3-class classifier on top of a 10-class model, examining
  how well the learned representation transfers

Training hyperparameters used throughout: `alpha=0.3`, `lambda=1e-4`,
`num_iterations=150`, `k=10`.

---

### 4. Kernel Methods (`kernel.py`)
Kernelised softmax regression is implemented from scratch for two kernels:

| Kernel | Formula |
|--------|---------|
| **Polynomial** | $K(x, y) = (\langle x, y \rangle + c)^p$ |
| **RBF (Gaussian)** | $K(x, y) = \exp(-\gamma \|x - y\|^2)$ |

Pairwise squared distances in the RBF kernel are computed efficiently via
`‖x−y‖² = ‖x‖² + ‖y‖² − 2xᵀy` to avoid an O(n²d) loop.

---

### 5. Feature Engineering (`features.py`)

#### Principal Component Analysis (PCA)
PCA is fitted **on training data only** to prevent data leakage. The
covariance scatter matrix is decomposed with `np.linalg.eigh` (guarantees
real-valued, sorted eigenvalues). Three dimensionality reduction experiments
are run in `main.py`:

| Experiment | Components | Input to Softmax |
|------------|------------|-----------------|
| PCA-18 | 18 | 18-D projected features |
| PCA-10 | 10 | 10-D projected features |
| PCA-10 + Cubic | 10 → 286 | Cubic expansion of 10-D PCA |

Additional PCA utilities:
- `center_data(X)` — zero-centres features, returns means for later reuse
- `plot_PC(X, pcs, labels)` — 2D scatter plot with digit annotations
- `reconstruct_PC(x_pca, pcs, n_components)` — reconstructs images from
  their low-dimensional PCA representation

#### Cubic Feature Map (`cubic_features`)
Implements the explicit feature map corresponding to the **cubic polynomial
kernel**, expanding a d-dimensional input into all monomials of degree ≤ 3
(with appropriate √6 / √3 scaling for cross-terms). For d = 10 (post-PCA)
this produces **286 features**, enabling a linear classifier to learn
cubic decision boundaries without the kernel trick.

---

## Pipeline Summary (`main.py`)

```
MNIST raw (784-D)
       │
       ├─► Linear Regression (closed form)
       │
       ├─► LinearSVC  ──► One-vs-Rest  /  Multiclass
       │
       ├─► Softmax Regression ──► 10 classes
       │                      └─► 3 classes (mod-3 labels)
       │
       ├─► PCA-18 ──► Softmax
       │
       ├─► PCA-10 ──► Softmax
       │          └─► Cubic features (286-D) ──► Softmax
       │          └─► Cubic polynomial kernel SVM
       │
       └─► Kernelised Softmax (Polynomial / RBF)
```

---

## Technologies Used

| Technology | Purpose |
|------------|---------|
| **Python 3** | Core language |
| **NumPy** | Vectorised linear algebra, array operations |
| **SciPy** (`sparse`) | Efficient sparse one-hot matrix construction in gradient steps |
| **scikit-learn** (`LinearSVC`, `SVC`) | SVM baselines |
| **Matplotlib** | Digit visualisation, cost curve plotting, PCA scatter plots |
| **pickle + gzip** | Compressed serialisation of the MNIST dataset and trained θ |

---

## Getting Started

### Prerequisites

```bash
pip install numpy scipy matplotlib scikit-learn
```

### Run All Experiments

Uncomment the desired `print(...)` line at the bottom of each section in
`main.py` and run:

```bash
cd project_2_and_3/project_2
python main.py
```

### Quick Data Check

```python
from utils import get_MNIST_data, plot_images

train_x, train_y, test_x, test_y = get_MNIST_data()
print(train_x.shape)   # (60000, 784)
print(test_y.shape)    # (10000,)
plot_images(train_x[:20])
```

---

## Typical Results

| Method                                      | Test Error |
|---------------------------------------------|------------|
| Linear Regression                           | 0.7697     |
| SVM One vs. Rest                            | 0.0075     |
| Multiclass SVM                              | 0.0819     |
| Softmax Regression (temp=1)                 | 0.1005     |
| Softmax Regression — mod3 labels (current model) | 0.0768 |
| Softmax Regression — mod3 trained (temp=1) | 0.1881     |
| Softmax + PCA 18 components                 | 0.1476     |
| Softmax + PCA 10 components                 | 0.2089     |
| Softmax + Cubic Features + PCA 10          | 0.0839     |
| SVM Cubic Polynomial Kernel + PCA 10       | 0.0603     |
| SVM RBF Kernel + PCA 10                    | 0.0636     |
| Kernelized Softmax Polynomial + PCA 10     | 0.0935     |
| Kernelized Softmax Gaussian RBF + PCA 10   | 0.0938     |


> Exact numbers depend on random seed and any hyperparameter tuning performed.

---
