from numpy import array

from lib.generators import get_identity_matrix, get_hilbert_matrix, \
        get_random_matrix
from lib.printers import print_matrix

from .methods import eigen_jacobi, eigen_lib, print_evs, ChoiceMethod

matrices = [
    array(get_hilbert_matrix(2)),
    array(get_hilbert_matrix(5)),
    array(get_identity_matrix(3)),
    array([[1, 0, 0], [0, 2, 0], [0, 0, 3]]),
    array(get_random_matrix(3)),
]

def run():
    for i, matrix in enumerate(matrices, 1):
        print(f"Matrix #{i}")
        print_matrix(matrix)
        print()

        el = eigen_lib(matrix)
        print_evs("Library", el)
        print()

        for epsilon in [10, 1, 0.1, 0.001]:
            print(f"Epsilon: {epsilon}")

            ej_mod = eigen_jacobi(matrix, ChoiceMethod.MAX_OFF_DIAG, epsilon)
            print_evs("Jacobi, max not diagonal", ej_mod)

            ej_obo = eigen_jacobi(matrix, ChoiceMethod.ONE_BY_ONE, epsilon)
            print_evs("Jacobi, one by one", ej_obo)

            print()

        print("\n\n\n")
