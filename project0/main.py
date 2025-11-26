import numpy as np
import sys

def randomization(n):
    """
    Arg:
      n - an integer
    Returns:
      A - a randomly-generated nx1 Numpy array.
    """
    return np.random.random([n,1])

def operations(h, w):
    """
    Takes two inputs, h and w, and makes two Numpy arrays A and B of size
    h x w, and returns A, B, and s, the sum of A and B.

    Arg:
      h - an integer describing the height of A and B
      w - an integer describing the width of A and B
    Returns (in this order):
      A - a randomly-generated h x w Numpy array.
      B - a randomly-generated h x w Numpy array.
      s - the sum of A and B.
    """
    #Your code here
    A = np.random.random([h,w])
    B = np.random.random([h, w])
    s = A+B
    return A, B, s


def norm(A, B):
    """
    Takes two Numpy column arrays, A and B, and returns the L2 norm of their
    sum.

    Arg:
      A - a Numpy array
      B - a Numpy array
    Returns:
      s - the L2 norm of A+B.
    """
    s = A + B
    s = np.linalg.norm(s)
    return s


def neural_network(inputs, weights):
    """
     Takes an input vector and runs it through a 1-layer neural network
     with a given weight matrix and returns the output.

     Arg:
       inputs - 2 x 1 NumPy array
       weights - 2 x 1 NumPy array
     Returns (in this order):
       out - a 1 x 1 NumPy array, representing the output of the neural network
    """
    return np.tanh(np.matmul(weights.transpose(), inputs))

def scalar_function(x, y):
    """
    Returns the f(x,y) defined in the problem statement.
    """
    return x/y if x>y else x*y

def vector_function(x, y):
    """
    Make sure vector_function can deal with vector input x,y 
    """
    vectorized = np.vectorize(scalar_function)
    test = vectorized(x,y)
    print(test)
    return vectorized(x,y)

if __name__ == "__main__":
    vector_function([2, 2], [4,6])
    vector_function([4, 6], [2, 2])
    print(sys.executable)

