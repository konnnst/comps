from numpy import array

from lib.generators import get_identity_matrix, get_hilbert_matrix, get_random_matrix
from lib.printers import print_matrix

from .methods import power_method, product_method, library_method

matrices = [
    array(get_hilbert_matrix(2)),
    array(get_hilbert_matrix(5)),
    array(get_identity_matrix(3)),
    array(get_random_matrix(3)),
    array([[1, 0, 0], [0, 2, 0], [0, 0, 3]], dtype="float64"),
]


def run():
    for i, matrix in enumerate(matrices, 1):
        print(f"Matrix #{i}")
        print_matrix(matrix)
        print()

        lib_ev = library_method(matrix)
        print(f"Library method:\n{lib_ev}")

        for epsilon in [10, 1, 0.1, 0.001]:
            print(f"Epsilon: {epsilon}")

            pwm_ev = power_method(matrix, lib_ev, epsilon)
            print(f"Power method:\n{pwm_ev}")

            pdc_ev = product_method(matrix, lib_ev, epsilon)
            print(f"Product method:\n{pdc_ev}")

            print()

        print("\n\n\n")
