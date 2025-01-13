from numpy import linalg

from lib.printers import print_matrix
from lib.slau import Problem

from lib.generators import get_hilbert_matrix, \
        get_random_matrix, get_identity_matrix, \
        get_random_vector
from tasks.slau_solver.methods import get_lu_decomposition, \
        print_decomposition, get_qr_decomposition, \
        solve_lu, solve_qr

problems = [
        Problem([[1, 0.99], [0.99, 0.98]], [1.99, 1.97]),
        Problem([[1, 0.99], [0.99, 0.98]], [2, 2]),
        Problem(get_hilbert_matrix(4), [0, 0, 0, 0]),
        Problem(get_random_matrix(4), [0, 0, 0, 0]),
        Problem(get_identity_matrix(4), [1, 2, 3, 4]),
        Problem([[4, 3], [6, 3]], [1, 2]),
        Problem(get_random_matrix(4), get_random_vector(4)),
        # Problem(get_random_matrix(100), get_random_vector(100)),
]


def run():
    for i, problem in enumerate(problems, 1):
        print(f"#{i}\n{problem}")
        real_solution = linalg.solve(problem.a, problem.b)
        print(f"Real solution:\n{real_solution}")

        # Get LU decomposition matrices
        l, u = get_lu_decomposition(problem.a)
        print_decomposition("LU decomposition", l, u)
        # Check if LU decomposition correct
        lu_diff = linalg.norm(linalg.matmul(l, u) - problem.a)
        print(f"||A - LU|| = {lu_diff}")
        # Solve equation using LU decomposition
        lu_solution = solve_lu(problem, l, u)
        print(f"LU solution:\n{lu_solution}")
        print(f"LU norm delta: {linalg.norm(real_solution - lu_solution)}")


        # Get QR decomposition matrices
        q, r = get_qr_decomposition(problem.a)
        print_decomposition("QR decomposition", q, r)
        # Check if QR decomposition correct
        qr_diff = linalg.norm(linalg.matmul(q, r) - problem.a)
        qqt = linalg.matmul(q, linalg.matrix_transpose(q))
        print("q * qt")
        print_matrix(qqt)
        print(f"||I - q * qt|| = {linalg.norm(get_identity_matrix(len(qqt)) - qqt)}")
        # Solve equation using QR decomposition
        qr_solution = solve_qr(problem, q, r)
        print(f"QR solution:\n{qr_solution}")
        print(f"QR norm delta: {linalg.norm(real_solution - qr_solution)}")


        print("\n\n")
