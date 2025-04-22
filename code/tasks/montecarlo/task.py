from math import sin, pi, sqrt, cos

from .integral import Integral


integrals = [
    Integral("sin x", lambda x: sin(x), [[0, pi]]),
    Integral("x + y", lambda x, y: x + y, [[0, 1], [0, 1]]),
    Integral(
        "x + y / sin(z)",
        lambda x, y, z: x + y / sin(z),
        [[0, 1], [0, 1], [pi / 4, 3 * pi / 4]],
    ),
    Integral(
        "sin(x) / sqrt(x^2 + x^3 + cos(x) + 5)",
        lambda x: sin(x) / sqrt(x ** 2 + x ** 3 + cos(x) + 5),
        [[0, 100]],
    ),
]


def run():
    for integral in integrals:
        (lib, _), mc = integral.library(), integral.monte_carlo()
        print(integral)
        print("Library: ", round(lib, 4))
        print("Monte-Carlo: ", round(mc, 4))
        print("delta = ", round(abs(lib - mc), 6), end="\n\n")
        
