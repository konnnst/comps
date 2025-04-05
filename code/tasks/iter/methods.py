from numpy import linalg, matrix, add, diff, array
from time import time

from lib.slau import Problem
from lib.generators import get_identity_matrix, get_zero_matrix


def solve_simple_iter(problem: Problem, eps):
    m = get_identity_matrix(len(problem.a))
    n = array(get_identity_matrix(len(problem.a))) -  array(problem.a)
    b = linalg.matmul(linalg.inv(m), n)
    g = linalg.matmul(linalg.inv(m), problem.b)

    solution = [0 for _ in range(len(problem.a))]
    initial_time = time()
    while problem.get_solution_delta(solution) > eps and time() - initial_time < 15:
        solution = add(linalg.matmul(b, solution), g)

    return solution


def solve_zeidel_iter(problem: Problem, eps):
    n = len(problem.a)
    l, d, u = array(get_zero_matrix(n), dtype="float64"), \
        array(get_zero_matrix(n), dtype="float64"), array(get_zero_matrix(n), dtype="float64")
    for i in range(n):
        for j in range(n):
            to_edit = l if i > j else u if i < j else d
            print(id(to_edit))
            to_edit[i][j] = problem.a[i][j]
    m = array(linalg.inv(d + l))
    b = - m * u
    g = m * problem.b

    solution = [0 for _ in range(len(problem.a))]
    initial_time = time()
    while problem.get_solution_delta(solution) > eps and time() - initial_time < 15:
        solution = add(linalg.matmul(b, solution), g)
    
    return solution

def iterate(problem, b, g):
    solution = [0 for _ in range(len(problem.a))]
    initial_time = time()
    while problem.get_solution_delta(solution) > eps and time() - initial_time < 15:
        solution = add(linalg.matmul(b, solution), g)

    return solution