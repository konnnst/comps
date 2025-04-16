import numpy as np
from time import time
from numpy import linalg
from lib.eigenvector import Eigenvector


def gg_radius(matrix, i):
    r = sum(matrix[i]) - matrix[i][i]
    return r

def gg_radiuses_fit(matrix, eps):
    for i in range(len(matrix)):
        if gg_radius(matrix, i) >= eps:
            return False
    return True


class Answer:
    def __init__(self, success, epsilon, eigenvalues, iterations):
        self.success = success
        self.epsilon = epsilon
        self.eigenvalues = sorted([round(float(ev), 4) for ev in eigenvalues])
        self.iterations = iterations

    def __str__(self):
        s = f"Epsilon: {self.epsilon}, "
        s += "success" if self.success else "fail"
        s += f"({self.iterations}) {self.eigenvalues}"
        return s


def eigen_jacobi(matrix, choice_method, eps):
    n = matrix.shape[0]
    V = np.eye(n)
    iterations = 0
    p, q = 0, 0

    start = time()
    success_flag = True
    while not gg_radiuses_fit(matrix, eps):
        if time() - start > 5:
            success_flag = False
            break

        p, q = choice_method.next(matrix, p, q)

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

    evs = [matrix[i][i] for i in range(n)]
    answer =  Answer(success_flag, eps, evs, iterations)
    return answer


def eigen_lib(matrix):
    evs = [ev for ev in linalg.eig(matrix).eigenvalues]
    result = Answer(True, 0, evs, 0)
    return result
