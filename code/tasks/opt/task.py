from math import sin, pi

from lib.expression import Expression
from .methods import optimization_loop, gradient_descent_step, \
        nesterov_step, heavy_ball_step, newton_step


expressions = [
    Expression("y = x1 ^ 2", lambda x: x[0] ** 2, 1, [0]),
    Expression("y = sin x1", lambda x: sin(x[0]), 1, [-pi / 2]),
    Expression("y = (x1 - 5) ^ 2", lambda x: (x[0] - 5) ** 2, 1, [5]),
    Expression(
        "y = x1 ^ 2 + (x2 - 1) ^ 2",
        lambda x: x[0] ** 2 + (x[1] - 1) ** 2,
        2,
        [0, 1],
    ),
    Expression(
        "y = (x1 ^ 2 + 2) * ((x2 - 2) ^ 2 + 3) * ((x3 - 3) ^ 2 + 5)",
        lambda x: (x[0] ** 2 + 2)
        * ((x[1] - 2) ** 2 + 3)
        * ((x[2] - 3) ** 2 + 5),
        3,
        [-2, 2, 3]
    ),
]

methods = [
    gradient_descent_step,
    nesterov_step,
    heavy_ball_step,
    newton_step,
]


def run():
    for expr in expressions:
        print(expr)
        for method in methods:
            print(optimization_loop(method, expr))
        print()
