from numpy import linalg

from lib.slau import Problem

from lib.generators import get_identity_matrix, get_zero_matrix


def get_lu_decomposition(matrix):
    l, u = get_identity_matrix(len(matrix)), matrix
    m = matrix

    for k in range(len(matrix) - 1):
        m = get_identity_matrix(len(matrix))

        for i in range(k + 1, len(matrix)):
            m[i][k] = -u[i][k] / u[k][k]

        l = linalg.matmul(l, linalg.inv(m))
        u = linalg.matmul(m, u)

    return l, u


def print_decomposition(name, l, r):
    print(name)
    for i in range(len(l)):
        for k in range(len(l[0])):
            print(f"{l[i][k]:6.3f}", end="")
        print("|", end="")
        for k in range(len(r[0])):
            print(f"{r[i][k]:6.3f}", end="")
        print()


def get_holetsky_decomposition(matrix):
    l = get_zero_matrix(len(matrix))

    for i in range(len(matrix)):
        diag = matrix[i][i]
        for k in range(i):
            diag -= matrix[i][k] ** 2
        diag **= 0.5
        l[i][i] = diag

        for k in range(i + 1, len(matrix)):
            diff = 0
            for j in range(k):
                diff += l[i][j] * l[k][j]
            l[i][k] = (matrix[i][k] - diff) / l[i][i]

    return linalg.matrix_transpose(l), l


def get_qr_decomposition(matrix):
    q = get_identity_matrix(len(matrix))

    for i in range(len(matrix)):
        for k in range(i + 1, len(matrix)):
            t = get_identity_matrix(len(matrix))
            cos = matrix[i][i] / (matrix[i][i] ** 2 + matrix[i][k] ** 2) ** 0.5
            sin = -matrix[i][k] / (matrix[i][i] ** 2 + matrix[i][k] ** 2) ** 0.5

            t[i][i] = cos
            t[k][k] = cos
            t[i][k] = -sin
            t[k][i] = sin

            q = linalg.matmul(q, linalg.inv(t))

    r = linalg.matmul(linalg.inv(q), matrix)
    return linalg.qr(matrix)

def solve(m, t, b):
    answer = []

    for i in range(len(m)):
        prevsum = 0
        for k in range(len(answer)):
            prevsum += 

        answer.append((b[i] - prevsum) / m[i][i])

    return answer
