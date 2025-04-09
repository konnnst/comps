from scipy.linalg import norm
from lib.expression import Expression


def gradient_descent(expr: Expression, epsilon=0.01, gamma=0.00001):
    prev_x = [0 for _ in range(expr.dim())]

    while True:
        grad = expr.grad(prev_x)
        now_x = prev_x - gamma * grad
        grad_norm = norm(grad)

        if grad_norm < epsilon:
            break
        prev_x = now_x

    return now_x


def fastest_descent(expr: Expression, radius, epsilon=0.01):
    pass


def heavy_ball():
    pass


def newton():
    pass
