# MIT 6.86x — Project 1: Sentiment Analysis

A machine learning project that builds a sentiment analysis classifier for Amazon product reviews using linear classifiers implemented from scratch in Python.

## Overview

This project implements and compares several linear classification algorithms to classify Amazon product reviews as **positive** (+1) or **negative** (-1). The pipeline covers everything from raw text feature extraction to training, evaluation, and hyperparameter tuning.

## Technologies Used

- **Python 3.11**
- **NumPy** — vectorized math and matrix operations
- **Matplotlib** — plotting decision boundaries and accuracy curves
- **Seaborn** — enhanced data visualization
- **Scikit-Learn** — used in demo/recitation notebooks for comparison and cross-validation

## Project Structure
```
sentiment_analysis/ 
├── project1.py # Core implementation (algorithms + feature extraction) 
├── utils.py # Data loading utilities 
├── test.py # Automated test suite 
├── main.py # Entry point for running experiments 
├── reviews.tsv # Amazon product review dataset 
├── toy_data.tsv # 2D synthetic dataset for visualization 
├── stopwords.txt # Common English stopwords for text preprocessing 
├── tumor-diagnosis-demo.py # Scikit-Learn SVM demo on breast cancer dataset 
└── unit1-recitation.py # Recitation notebook
```

## Implemented Algorithms

All classifiers are implemented **from scratch** using NumPy:

| Algorithm | Description |
|---|---|
| **Perceptron** | Classic online linear classifier |
| **Average Perceptron** | Averaged weights for better generalization |
| **Pegasos** | Stochastic gradient descent with L2 regularization (SVM) |

## NLP Pipeline

1. **Bag of Words** — builds a vocabulary dictionary from raw review text
2. **Stopword Removal** — filters common words (e.g., "the", "is", "and") using `stopwords.txt`
3. **Feature Extraction** — converts reviews into binary or count-based feature vectors
4. **Classification** — trains linear classifiers on the feature vectors
5. **Accuracy Evaluation** — measures train and validation accuracy

## Loss Functions

- **Hinge Loss (single)** — computes loss for a single labeled example
- **Hinge Loss (full)** — averages hinge loss over the entire dataset

## Dataset

- **`reviews.tsv`** — Amazon product reviews with labels (+1 positive, −1 negative), review titles, and text
- **`toy_data.tsv`** — 2D labeled dataset used to visualize decision boundaries

## Key Concepts Demonstrated

- Linear classification and decision boundaries
- Online vs. batch learning
- Regularization and the bias-variance tradeoff
- Hyperparameter tuning (number of epochs `T`, regularization strength `λ`)
- Text preprocessing and bag-of-words representation
- Binary vs. count-based feature vectors

## Running the Tests

```bash
python test.py
```

The test suite validates each component including loss functions, update steps, full training loops, bag-of-words construction, and feature extraction.

## Running Experiments

Different experiments can be enabled by commenting and uncommenting the relevant sections in `main.py`. For example:

- To run a specific algorithm (perceptron, average perceptron, or pegasos), uncomment the corresponding training and plotting calls.
- To tune hyperparameters (e.g., `T` for number of iterations, `L` for the regularization parameter in pegasos), uncomment the `tune_*` function calls.
- To switch between binary and non-binary bag-of-words features, modify the `extract_bow_feature_vectors` call in the feature extraction section.
- To test on the toy dataset vs. the sentiment dataset, comment/uncomment the respective `load_toy_data` or `load_data` calls.

After configuring `main.py`, run the project with:

```bash
python main.py
```

## Example Results
The classifier is evaluated on held-out Amazon reviews. Performance is measured by accuracy on both training and validation sets across varying hyperparameters T and λ.