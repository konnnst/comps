import numpy as np
from numpy import array, linalg, dot
from time import time

from lib.eigenvector import Eigenvector
from lib.generators import get_hilbert_matrix, get_identity_vector


def power_method(matrix, library_vector: Eigenvector, eps, iter_limit=20000):
    n = matrix.shape[0]
    x_prev = get_identity_vector(n)

    iter_count = 0
    while iter_count < iter_limit:
        x = matrix @ x_prev
        evalue = (x @ x_prev) / (x_prev @ x_prev)
        x /= linalg.norm(x)

        if abs(library_vector.value - evalue) < eps:
            result = Eigenvector(x, evalue, iter_count)
            break
        result = Eigenvector(x, evalue, iter_count)


        iter_count += 1
        x_prev = x
    else:
        result = Eigenvector(None, None, iter_count)

    return result


def product_method(matrix, library_vector, eps, iter_limit=20000):
    n = matrix.shape[0]
    x_prev = get_identity_vector(n)
    y_prev = get_identity_vector(n)

    iter_count = 0
    while iter_count < iter_limit:
        iter_count += 1

        x = matrix @ x_prev
        y = matrix.transpose() @ y_prev
        evalue = (x @ y) / (x_prev @ y)
        x /= linalg.norm(x)
        y /= linalg.norm(y)

        if abs(library_vector.value - evalue) < eps:
            result = Eigenvector(x, evalue, iter_count)
            break
        result = Eigenvector(x, evalue, iter_count)

        x_prev = x
    else:
        result = Eigenvector(None, None, iter_count)

    return result


def library_method(matrix):
    evs = np.linalg.eig(matrix)
    eigenvalues, eigenvectors = evs.eigenvalues, evs.eigenvectors.transpose()
    print("Eigenvalues: ", eigenvalues)
    print("Eigenvectors: ", eigenvectors)

    result = Eigenvector(None, 0, 0)
    for evalue, evec in zip(eigenvalues, eigenvectors):
        if abs(evalue) > abs(result.value):
            result = Eigenvector(evec / linalg.norm(evec), evalue, 0)

    return result
