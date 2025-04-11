import numpy as np
from numpy import linalg
from enum import Enum
from lib.eigenvector import Eigenvector


class ChoiceMethod(Enum):
    MAX_OFF_DIAG = "max_off_diag"
    ONE_BY_ONE = "one_by_one"


def get_max_off_diag(matrix, n, prev_p=None, prev_q=None):
    max_off_diag = 0.0
    p, q = 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            if abs(matrix[i, j]) > max_off_diag:
                max_off_diag = abs(matrix[i, j])
                p, q = i, j
    return p, q


def get_one_by_one():
    pass


def eigen_jacobi(matrix, method, library, eps=1e-10, max_iter=10000):
    n = matrix.shape[0]

    V = np.eye(n)
    iterations = 0

    for _ in range(max_iter):
        if method == ChoiceMethod.MAX_OFF_DIAG:
            p, q = get_max_off_diag(matrix, n)
            if matrix[p][q] < eps:
                break
        else:
            p, q = get_max_off_diag()

        if matrix[p, p] == matrix[q, q]:
            phi = np.pi / 4
        else:
            phi = 0.5 * np.arctan(2 * matrix[p, q] / (matrix[p, p] - matrix[q, q]))

        c = np.cos(phi)
        s = np.sin(phi)
        R = np.eye(n)
        R[p, p] = c
        R[q, q] = c
        R[p, q] = -s
        R[q, p] = s

        matrix = R.T @ matrix @ R

        V = V @ R
        iterations += 1

        evs = []
        for i in range(n):
            ev = Eigenvector(V[i], matrix[i][i], iterations)
            evs.append(ev)

    return evs


def eigen_lib(matrix):
    result = []
    evs = linalg.eig(matrix)
    for eval, evec in zip(evs.eigenvalues, evs.eigenvectors):
        ev = Eigenvector(evec, eval, 0)
        result.append(ev)

    return result


def gg_radius(matrix, i):
    r = sum(matrix[i]) - matrix[i][i]
    return r


def print_evs(type, evs):
    print(type)
    for i, ev in enumerate(evs, 1):
        print(f"{i}.\n{ev}")
