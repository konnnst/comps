from numpy import array
import numpy as np
from time import time
from scipy.linalg import norm

from .minimum import Minimum


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

    if norm(grad) >= epsilon:
        try:
            delta = np.linalg.solve(hess, -grad)
        except np.linalg.LinAlgError:
            delta = -grad

        alpha = 1.0
        while True:
            x_new = x + alpha * delta
            try:
                val_new = expr(x_new)
                if val_new < val + 1e-4 * alpha * grad.dot(delta) or alpha < 1e-10:
                    break
            except:
                pass
            alpha *= 0.5
        x = x_new

    return x, y, None, "Newton"


def optimization_loop(optimization_step, expr, epsilon=0.01, timeout=5):
    x = array([1 for _ in range(expr.dim())])
    x_prev = array([1 for _ in range(expr.dim())])
    y = array([1 for _ in range(expr.dim())])

    start = time()
    iterations = 0
    while time() - start < timeout:
        iterations += 1
        x, y, x_prev, method_name = optimization_step(expr, epsilon, x, y, x_prev)
        if norm(expr.grad(x)) < epsilon:
            break

    minimum = Minimum(x, norm(expr.grad(x)), iterations, time() - start, method_name)
    return minimum
