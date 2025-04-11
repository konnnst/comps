from lib.slau import Problem

from lib.generators import get_hilbert_matrix, get_random_matrix, \
        get_identity_matrix, get_random_sparse_matrix, get_random_sparse_vector

from .methods import solve_simple_iter, solve_zeidel_iter


problems = [
        Problem([[1, 0.99], [0.99, 0.98]], [1.99, 1.97]),
        Problem([[1, 0.99], [0.99, 0.98]], [2, 2]),
        Problem(get_hilbert_matrix(4), [1, 2, 3, 4]),
        Problem(get_random_matrix(4), [4, 3, 2, 1]),
        Problem(get_identity_matrix(4), [1, 2, 3, 4]),
        #Problem(get_random_sparse_matrix(1000), get_random_sparse_vector(1000)),
]


def run():
    for i, problem in enumerate(problems, 1):
        print(f"Problem #{i}")
        print(problem)

        for eps in [1000, 100, 1, 0.1, 0.001]:
            print(f"Solve with eps = {eps}")

            si_solution = solve_simple_iter(problem, eps)
            print(f"Simple iteration solution:\n{si_solution}")
            print(f"Simple iteration delta = {problem.get_solution_delta(si_solution)}")

            #zd_solution = solve_zeidel_iter(problem, eps)
            #print(f"Zeidel solution:\n{zd_solution}")
            #print(f"Zeidel delta = {problem.get_solution_delta(zd_solution)}")

            print()


        print("\n\n\n")
