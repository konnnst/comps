from lib.slau import Problem


from tasks.criteria.methods import get_spectral_criterion, \
        get_volume_criterion, get_angular_criterion, \
        get_da_solution, get_db_solution, get_da_db_solution, \
        norm

from lib.generators import get_hilbert_matrix, get_random_matrix, \
        get_identity_matrix


problems = [
        Problem([[1, 0.99], [0.99, 0.98]], [1.99, 1.97]),
        Problem([[1, 0.99], [0.99, 0.98]], [2, 2]),
        Problem(get_hilbert_matrix(4), [1, 2, 3, 4]),
        Problem(get_random_matrix(4), [4, 3, 2, 1]),
        Problem(get_identity_matrix(4), [1, 2, 3, 4]),
]


def run():
    for i, problem in enumerate(problems, 1):
        # Print problem
        print(f"Problem #{i}\n{problem}")

        # Calculating criteria
        sc = get_spectral_criterion(problem.a)
        vc = get_volume_criterion(problem.a)
        ac = get_angular_criterion(problem.a)

        # Print criteria
        print(f"Spectral: {sc}")
        print(f"Volume: {vc}")
        print(f"Angular: {ac}")

        # Get solution
        solution = problem.solve()

        # Print solution
        print(f"Solution: {solution}")

        # Get modified solutions
        da_solution = get_da_solution(problem)
        db_solution = get_db_solution(problem)
        da_db_solution = get_da_db_solution(problem)

        # Print differences between original and modified solutions
        print(f"|solution - da_solution| = {norm(solution - da_solution)}"
              f" [{(norm(solution - da_solution) / norm(solution)):10.6f}]")
        print(f"|solution - db_solution| = {norm(solution - db_solution)}"
                f" [{(norm(solution - db_solution) / norm(solution)):10.6f}]")
        print(f"|solution - da_db_solution| = {norm(solution - da_db_solution)}"
                f" [{(norm(solution - da_db_solution) / norm(solution)):10.6f}]")

        print("\n\n")
