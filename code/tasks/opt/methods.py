from time import time
from scipy.linalg import norm
import numpy as np

from .minimum import Minimum
from lib.generators import get_identity_vector


def gradient_descent_step(expr, epsilon, x, y, x_prev):
    x_prev = x
    grad = expr.grad(x_prev)
    x = x_prev - expr.gamma * grad
    return x, None, x_prev, "Gradient descent"


def heavy_ball_step(expr, epsilon, x, y, x_prev):
    prev_x = x
    prev_prev_x = prev_x
    grad = expr.grad(prev_x)
    x = prev_x - expr.alpha * grad + expr.beta * (prev_x - prev_prev_x)

    return x, None, prev_x, "Heavy ball"


def nesterov_step(expr, epsilon, x, y, x_prev):
    x_prev = x
    y_prev = y
    grad = expr.grad(y_prev)
    x = y_prev - expr.alpha * grad
    y = x + expr.beta * (x - x_prev)

    return x, y, None, "Nesterov"


def newton_step(expr, epsilon, x, y, x_prev):
    val = expr(x)
    grad = expr.grad(x)
    hess = expr.hess(x)

    try:
        delta = np.linalg.solve(hess, -grad)
    except np.linalg.LinAlgError:
        delta = -grad

    alpha = 1.0
    while True:
        x_new = x + alpha * delta
        val_new = expr(x_new)
        if val_new < val + 1e-4 * alpha * grad.dot(delta) or alpha < 1e-10:
            break
        alpha *= 0.5
    x = x_new

    return x, y, None, "Newton"


def optimization_loop(step, expr, epsilon=0.01, timeout=5):
    x, x_prev, y = [get_identity_vector(expr.dim())] * 3

    start = time()
    iterations = 0
    while time() - start < timeout:
        iterations += 1
        x, y, x_prev, method = step(expr, epsilon, x, y, x_prev)
        if norm(expr.grad(x)) < epsilon:
            break

    minimum = Minimum(x, norm(expr.grad(x)), iterations, time() - start, method)
    return minimum
