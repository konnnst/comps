from numpy import linalg, array
from time import time

from lib.slau import Problem
from lib.generators import get_zero_matrix
from .solution import Solution


def iterate(problem, b, c, eps):
    solution = [1 for _ in range(len(problem.a))]

    initial_time = time()
    while problem.get_solution_delta(solution) > eps and time() - initial_time < 10:
        solution = b @ solution + c
    return solution


def library(problem: Problem, eps=0.01):
    solution = Solution(linalg.solve(problem.a, problem.b), "library", 0)
    return solution


def simple_iter(problem: Problem, eps=0.01):
    n = len(problem.a)

    b, c = [], []

    for i in range(n):
        b_row = []
        for j in range(n):
            b_row.append(0 if j == i else -problem.a[i][j] / problem.a[i][i])

        b.append(b_row)
        c.append(problem.b[i] / problem.a[i][i])
    solution = iterate(problem, array(b), array(c), eps)

    return Solution(solution, "simple iter", eps)


def zeidel(problem: Problem, eps=0.01):
    n = len(problem.a)
    l = array(get_zero_matrix(n), dtype="float64")
    d = array(get_zero_matrix(n), dtype="float64")
    u = array(get_zero_matrix(n), dtype="float64")
    for i in range(n):
        for j in range(n):
            to_edit = l if i > j else u if i < j else d
            to_edit[i][j] = problem.a[i][j]
    m = d + l
    n = -u

    inv = linalg.inv(m)
    b = inv @ n
    c = inv @ problem.b

    solution = iterate(problem, b, c, eps)

    return Solution(solution, "zeidel", eps)
