from numpy import linalg
from copy import deepcopy
from lib.slau import Problem

def get_spectral_criterion(matrix):
    return linalg.norm(matrix) * linalg.norm(linalg.inv(matrix))


def get_volume_criterion(matrix):
    acc = 1
    for i in range(len(matrix)):
        curr = 1
        for k in range(len(matrix[i])):
            curr += matrix[i][k] ** 2
        acc *= curr ** 0.5

    return acc / abs(linalg.det(matrix))


def get_angular_criterion(matrix):
    def norm(vector):
        return sum([x * x for x in vector]) ** 0.5

    criterion = -1
    inv = linalg.inv(matrix)

    for n in range(len(matrix)):
        matrix_row, inverted_column = [], []
        for k in range(len(matrix)):
            matrix_row.append(matrix[n][k])
            inverted_column.append(inv[k][n])
        criterion = max(criterion, norm(matrix_row) * norm(inverted_column))

    return criterion


def get_da_solution(problem: Problem):
    modified = deepcopy(problem)
    modified.a[0][0] *= 1.01
    return modified.solve()

def get_db_solution(problem: Problem):
    modified = deepcopy(problem)
    modified.b[0] *= 1.01
    return modified.solve()

def get_da_db_solution(problem: Problem):
    modified = deepcopy(problem)
    modified.a[0][0] *= 1.01
    modified.b[0] *= 1.01
    return modified.solve()

def norm(vector):
    return linalg.norm(vector)
