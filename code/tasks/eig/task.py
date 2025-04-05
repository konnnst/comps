from numpy import array

from lib.generators import get_identity_matrix, get_hilbert_matrix
from lib.printers import print_matrix

from .methods import power_method, product_method, library_method

matrices = [
    array(get_hilbert_matrix(2)),
    array(get_hilbert_matrix(5)),
    array(get_identity_matrix(3)),
    array([[1, 0, 0], [0, 2, 0], [0, 0, 3]]),
]

def run():
    for i, matrix in enumerate(matrices, 1):
        print(f"Matrix #{i}")
        print_matrix(matrix)
        print()

        for epsilon in [10, 1, 0.1, 0.001]:
            print(f"Epsilon: {epsilon}")

            pwm_ev = power_method(matrix, epsilon)
            print(f"Power method:\n{pwm_ev}")

            pdc_ev = product_method(matrix, epsilon)
            print(f"Product method:\n{pdc_ev}")

            lib_ev = library_method(matrix)
            print(f"Library method:\n{lib_ev}")
            
            print()

        print("\n\n\n")