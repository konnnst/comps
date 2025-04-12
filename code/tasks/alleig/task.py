from numpy import array

from lib.generators import get_identity_matrix, get_hilbert_matrix, \
        get_random_matrix
from lib.printers import print_matrix

from .methods import eigen_jacobi, eigen_lib
from .choice_methods import max_off_diag, gg_opt

matrices = [
    array(get_hilbert_matrix(2)),
    array(get_hilbert_matrix(5)),
    array(get_identity_matrix(3)),
    array([[1, 0, 0], [0, 2, 0], [0, 0, 3]]),
    array(get_random_matrix(3)),
]

choice_methods = [
    max_off_diag,
    gg_opt,
]

epsilons = [
    10,
    1,
    0.1,
    0.001,
]

def run():
    for i, matrix in enumerate(matrices, 1):
        print(f"Matrix #{i}")
        print_matrix(matrix)
        print()

        print("Library")
        el = eigen_lib(matrix)
        print(el)

        for method in choice_methods:
            print("Method: ", method)
            for epsilon in epsilons:
                ev = eigen_jacobi(matrix, method, epsilon)
                print(ev)

        print("\n")
