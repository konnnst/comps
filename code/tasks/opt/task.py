from math import sin, pi
from lib.expression import Expression
from .methods import gradient_descent, nesterov, \
        heavy_ball, newton


expressions = [
    Expression("y = x1 ^ 2", lambda args: args[0] ** 2, 1, [0]), 
    Expression("y = sin x1", lambda args: sin(args[0]), 1, [-pi / 2]), 
    Expression("y = (x1 - 5) ^ 2", lambda args: (args[0] - 5) ** 2, 1, [5]),
    Expression(
        "y = x1 ^ 2 + (x2 - 1) ^ 2",
        lambda args: args[0] ** 2 + (args[1] - 1) ** 2,
        2, [0, 1], gamma=0.00001
    ),
    Expression(
        "y = (x1 ^ 2 + 2) * ((x2 - 2) ^ 2 + 3) * ((x3 - 3) ^ 2 + 5)",
        lambda args: (args[0] ** 2 + 2) * ((args[1] - 2) ** 2 + 3) * ((args[2] - 3) ** 2 + 5),
        3, [-2, 2, 3]
    ),
]

methods = [
    gradient_descent,
    nesterov,
    heavy_ball,
    newton,
]


def run():
    for expr in expressions:
        print(expr)
        for method in methods:
            print(method(expr))
        print()
