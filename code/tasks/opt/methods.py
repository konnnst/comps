from numpy import array
import numpy as np
from time import time
from scipy.linalg import norm

from lib.expression import Expression
from .minimum import Minimum


def gradient_descent(expr: Expression, epsilon=0.01, timeout=5):
    x = [0 for _ in range(expr.dim())]

    start = time()
    iterations = 0
    while time() - start < timeout:
        iterations += 1

        prev_x = x
        grad = expr.grad(prev_x)
        x = prev_x - expr.gamma * grad
        grad_norm = norm(grad)

        if grad_norm < epsilon:
            break

    minimum = Minimum(x, grad_norm, iterations, time() - start, "gradient_descent")
    return minimum


def gradient_descent_step(expr, epsilon, x, y, x_prev):
    prev_x = x
    grad = expr.grad(prev_x)
    x = prev_x - expr.gamma * grad
    grad_norm = norm(grad)
    return x, None, x_prev, grad_norm


def heavy_ball(expr: Expression, epsilon=0.01, timeout=5):
    x = array([0 for _ in range(expr.dim())])
    prev_x = array([0 for _ in range(expr.dim())])

    start = time()
    iterations = 0
    while time() - start < timeout:
        iterations += 1

        prev_x = x
        prev_prev_x = prev_x
        grad = expr.grad(prev_x)
        x = prev_x - expr.alpha * grad + expr.beta * (prev_x - prev_prev_x)
        grad_norm = norm(grad)

        if grad_norm < epsilon:
            break

    minimum = Minimum(x, grad_norm, iterations, time() - start, "heavy ball")
    return minimum


def heavy_ball_step(expr, epsilon, x, y, x_prev):
    prev_x = x
    prev_prev_x = prev_x
    grad = expr.grad(prev_x)
    x = prev_x - expr.alpha * grad + expr.beta * (prev_x - prev_prev_x)
    grad_norm = norm(grad)

    return x, None, prev_x, grad_norm


def nesterov(expr, epsilon=0.01, timeout=5):
    x = array([1 for _ in range(expr.dim())])
    y = array([1 for _ in range(expr.dim())])

    start = time()
    iterations = 0
    while time() - start < timeout:
        iterations += 1

        x_prev = x
        y_prev = y
        grad = expr.grad(y_prev)
        x = y_prev - expr.alpha * grad
        y = x + expr.beta * (x - x_prev)
        grad_norm = norm(grad)

        if grad_norm < epsilon:
            break

    minimum = Minimum(x, grad_norm, iterations, time() - start, "nesterov")
    return minimum


def nesterov_step(expr, epsilon, x, y, x_prev):
    x_prev = x
    y_prev = y
    grad = expr.grad(y_prev)
    x = y_prev - expr.alpha * grad
    y = x + expr.beta * (x - x_prev)
    grad_norm = norm(grad)

    return x, y, None, grad_norm


def newton(expr, epsilon=0.001, timeout=5):
    x = [1 for _ in range(expr.dim())]

    start = time()
    iterations = 0
    while time() - start < timeout:
        val = expr(x)
        grad = expr.grad(x)
        hess = expr.hess(x)

        if norm(grad) < epsilon:
            break

        try:
            delta = np.linalg.solve(hess, -grad)
        except np.linalg.LinAlgError:
            delta = -grad

        alpha = 1.0
        while True:
            iterations += 1
            x_new = x + alpha * delta
            try:
                val_new = expr(x_new)
                if val_new < val + 1e-4 * alpha * grad.dot(delta) or alpha < 1e-10:
                    break
            except:
                pass
            alpha *= 0.5

        x = x_new

    minimum = Minimum(x, norm(expr.grad(x)), iterations, time() - start, "newton")

    return minimum


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
        try:
            val_new = expr(x_new)
            if val_new < val + 1e-4 * alpha * grad.dot(delta) or alpha < 1e-10:
                break
        except:
            pass
        alpha *= 0.5

    return x, y, None, norm(grad)




def optimization_loop(optimization_step, expr, epsilon=0.001, timeout=5):
    x = array([1 for _ in range(expr.dim())])
    x_prev = array([1 for _ in range(expr.dim())])
    y = array([1 for _ in range(expr.dim())])

    start = time()
    iterations = 0
    while time() - start < timeout:
        iterations += 1
        x, y, x_prev, grad_norm = optimization_step(expr, epsilon, x, y, x_prev)
        if grad_norm < epsilon:
            break

    minimum = Minimum(x, grad_norm, iterations, time() - start, "gradient_descent")
    return minimum
