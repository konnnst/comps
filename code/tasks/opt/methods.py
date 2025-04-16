from numpy import array
from scipy.linalg import norm
from time import time
from lib.expression import Expression


def gradient_descent(expr: Expression, epsilon=0.01):
    prev_x = [0 for _ in range(expr.dim())]
    
    start = time()
    while time() - start < 60:
        grad = expr.grad(prev_x)
        now_x = prev_x - expr.gamma * grad
        grad_norm = norm(grad)

        if grad_norm < epsilon:
            break
        prev_x = now_x

    return now_x, grad_norm



def heavy_ball(expr: Expression, epsilon=0.01):
    prev_x = array([0 for _ in range(expr.dim())])
    prev_prev_x = array([0 for _ in range(expr.dim())])

    start = time()
    while time() - start < 60:
        grad = expr.grad(prev_x)
        now_x = prev_x - expr.alpha * grad + expr.beta * (prev_x - prev_prev_x)
        grad_norm = norm(grad)

        if grad_norm < epsilon:
            break

        prev_x = now_x
        prev_prev_x = prev_prev_x

    return now_x, grad_norm


def nesterov(expr, epsilon=0.01):
    x_prev = array([1 for _ in range(expr.dim())])
    y_prev = array([1 for _ in range(expr.dim())])

    start = time()
    while time() - start < 60:
        x = y_prev - expr.alpha * expr(y_prev)
        y = x + expr.beta * (x - x_prev)

        x_diff = norm(x - x_prev)
        if x_diff < epsilon:
            break
        
        x = x_prev
        y = y_prev

    return x, x_diff



def newton(expr, eps=0.001):
    x_prev = array([0 for _ in range(expr.dim())])
    return None
