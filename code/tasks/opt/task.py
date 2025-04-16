from math import sin
from lib.expression import Expression
from .methods import gradient_descent, nesterov, \
        heavy_ball, newton


expressions = [
    Expression("y = x1 ^ 2", lambda args: args[0] ** 2, 1),
    Expression("y = sin x1", lambda args: sin(args[0]), 1),
    Expression(
        "y = x1 ^ 2 + (x2 - 1) ^ 2",
        lambda args: args[0] ** 2 + (args[1] - 1) ** 2,
        2, gamma=0.00001
    ),
    Expression(
        "y = (x1 ^ 2 + 2) * ((x2 - 239) ^ 2 + 3) * ((x3 - 30) ^ 2 + 5)",
        lambda args: (args[0] ** 2 + 2) * ((args[1] - 239) ** 2 + 3)
        * ((args[2] - 30) ** 2 + 5),
        3,
    ),
]

methods = [
        gradient_descent,
        newton,
        nesterov,
        heavy_ball,
]

epsilons = [
    1,
    0.1,
    0.01,
]


def run():
    for expr in expressions:
        print(expr)
        for method in methods:
            for epsilon in epsilons:
                minimum = method(expr, epsilon)
                print(minimum)
            print()


