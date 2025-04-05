import numpy as np
from numpy import array, linalg, dot
from time import time

from lib.eigenvector import Eigenvector


def power_method(matrix, eps=1e-6, max_iter=100):
    n = matrix.shape[0]

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix A must be square.")
    
    b = np.random.rand(n)
    
    lambda_approx = 0.0
    
    for i in range(max_iter):
        b_new = matrix @ b
        
        index = np.argmax(np.abs(b_new))
        lambda_new = b_new[index]
        
        if np.abs(lambda_new - lambda_approx) < eps:
            eigenvalue = (b.T @ matrix @ b) / (b.T @ b)
            result = Eigenvector(b, eigenvalue, i + 1)
            break
        
        lambda_approx = lambda_new
        b = b_new / lambda_new
    else:
        result = None
    
    return result
    

def product_method(matrix, eps=1e-6, max_iter=100):
    n = matrix.shape[0]

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix A must be square.")
    
    x = np.random.rand(n)
    y = np.random.rand(n)
    
    lambda_approx = 0.0
    
    for i in range(max_iter):
        x_new = matrix @ x
        y_new = np.transpose(matrix) @ y

        lambda_new = dot(x_new, y_new) / dot(x, y_new)
        
        if np.abs(lambda_new - lambda_approx) < eps:
            eigenvalue = lambda_new
            result = Eigenvector(x_new, eigenvalue, i + 1)
            break
        
        lambda_approx = lambda_new
        x = x_new  / lambda_new
        y = y_new  / lambda_new
    else:
        result = None
    
    return result

def library_method(matrix):
    evs = np.linalg.eig(matrix)
    result = Eigenvector(None, 0, 0)

    for eval, evec in zip(evs.eigenvalues, evs.eigenvectors):
        if abs(eval) > abs(result.value):
            result = Eigenvector(evec, eval, 0)

    return result