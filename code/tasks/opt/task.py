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
        lambda args: (args[0] ** 2 + 2) * ((args[1] - 239) ** 2 + 3) * ((args[2] - 30) ** 2 + 5),
        3,
    ), 
]

def run():
    for expr in expressions:
        print(expr)

        print(f"Gradient descent: {gradient_descent(expr)}")
        print(f"Heavy ball: {heavy_ball(expr)}")
        print(f"Nesterov: {nesterov(expr)}")
        print("Newton:")

        print()

