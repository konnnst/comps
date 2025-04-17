from scipy import linalg
from numpy import array
import random


def get_random_matrix(n):
    return [[random.random() for i in range(n)] for k in range(n)]


def get_random_symmetric_matrix(n):
    m = array([[random.random() for i in range(n)] for k in range(n)])
    return m + m.transpose()


def get_random_vector(n):
    return [random.random() for i in range(n)]


def get_hilbert_matrix(n):
    return linalg.hilbert(n)


def get_identity_matrix(n):
    return [[int(k == i) for k in range(n)] for i in range(n)]


def get_identity_vector(n):
    return array([0.5 for _ in range(n)], dtype="float64")


def get_random_diagonally_dominant_matrix(n):
    matrix = []
    for i in range(n):
        row = []
        rowsum = 0
        for j in range(n):
            now = random.random()
            rowsum += abs(now)
            row.append(now)
        row[i] = random.choice([-1, 1]) * random.randrange(2, 10) * rowsum
        matrix.append(row)
    return array(matrix)

def get_zero_matrix(n):
    return [[0 for _ in range(n)] for _ in range(n)]


def get_random_sparse_matrix(n):
    matrix = get_zero_matrix(n)
    count = random.randint(int(n / 100), int(n / 50))

    for _ in range(count):
        i, j = random.randint(0, n - 1), random.randint(0, n - 1)
        value = random.randint(-1000, 1000)
        matrix[i][j] = value

    return matrix

def get_random_sparse_vector(n):
    vector = [0 for _ in range(n)]
    count = random.randint(int(n / 100), int(n / 50))
    for _ in range(count):
        i = random.randint(0, n - 1)
        value = random.randint(-1000, 1000)
        vector[i] = value
    return vector
