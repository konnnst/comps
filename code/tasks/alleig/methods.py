import numpy as np
from time import time
from numpy import linalg
from enum import Enum
from lib.eigenvector import Eigenvector


class ChoiceMethod(Enum):
    MAX_OFF_DIAG = "max_off_diag"
    ONE_BY_ONE = "one_by_one"
    BARRIER = "barrier"


def get_max_off_diag(matrix, n, prev_p=None, prev_q=None):
    max_off_diag = 0.0
    p, q = 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            if abs(matrix[i, j]) > max_off_diag:
                max_off_diag = abs(matrix[i, j])
                p, q = i, j
    return p, q


def gg_radius(matrix, i):
    r = sum(matrix[i]) - matrix[i][i]
    return r


def gg_condition_fit(matrix, eps):
    for i in range(len(matrix)):
        if gg_radius(matrix, i) > eps:
            return False

    return True


def eigen_jacobi(matrix, method, eps=1e-10, max_iter=10000):
    n = matrix.shape[0]

    V = np.eye(n)
    iterations = 0

    p, q = 0, 0
    evs = []
    for i in range(n):
        ev = Eigenvector(V[i], matrix[i][i], iterations)
        evs.append(ev)

    start = time()
    while time() - start < 10:
        if method == ChoiceMethod.MAX_OFF_DIAG:
            p, q = get_max_off_diag(matrix, n)
            if matrix[p][q] < eps:
                break
        elif method == ChoiceMethod.ONE_BY_ONE:
            if q == n - 1:
                p, q = p + 1, 0
            else:
                q += 1

            if p == n:
                p, q = 0, 0

            if p == q:
                continue
        elif method == ChoiceMethod.BARRIER:
            raise NotImplementedError()
        else:
            raise NotImplementedError()

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

        if gg_condition_fit(matrix, eps):
            break

    return evs


def eigen_lib(matrix):
    result = []
    evs = linalg.eig(matrix)
    for eval, evec in zip(evs.eigenvalues, evs.eigenvectors):
        ev = Eigenvector(evec, eval, 0)
        result.append(ev)

    return result


def print_evs(method, evs):
    print(f"{method}: {sorted([float(round(ev.value, 3)) for ev in evs])}")
