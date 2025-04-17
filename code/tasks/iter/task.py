from lib.slau import Problem

from lib.generators import get_random_diagonally_dominant_matrix, \
        get_random_vector

from .methods import simple_iter, zeidel, library


problems = [
    Problem(get_random_diagonally_dominant_matrix(2), get_random_vector(2)),
    Problem(get_random_diagonally_dominant_matrix(3), get_random_vector(3)),
    Problem(get_random_diagonally_dominant_matrix(9), get_random_vector(9)),
]

methods = [
      simple_iter,
      zeidel,
]

epsilons = [
    10,
    1,
    0.1,
    0.001,
]


def run():
    for i, problem in enumerate(problems, 1):
        print(f"Problem #{i}")
        print(problem)

        library_solution = library(problem)
        print(library_solution)
        print()

        for method in methods:
            for eps in epsilons:
                solution = method(problem, eps)
                print(solution)
                print(
                    f"||lib - {solution.method}|| = ",
                    round(solution.distance(library_solution), 4)
                )
            print()

        print("\n\n")
