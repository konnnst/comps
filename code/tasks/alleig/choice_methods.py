from enum import Enum
from .methods import gg_radius


@staticmethod
def __get_max_off_diag(matrix, p, q):
    max_off_diag = 0.0
    n = len(matrix)
    p, q = 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            if abs(matrix[i, j]) > max_off_diag:
                max_off_diag = abs(matrix[i, j])
                p, q = i, j
    return p, q


@staticmethod
def __gg_opt(matrix, p, q):
    n = len(matrix)
    max_gg_r_row, max_gg_r = -1, -1

    for i in range(n):
        now_gg_r = gg_radius(matrix, i)
        if now_gg_r > max_gg_r:
            max_gg_r_row, max_gg_r = i, now_gg_r

    max_abs_col, max_abs = -1, -1
    for j in range(n):
        now_abs = abs(matrix[max_gg_r_row][j])
        if j != max_gg_r_row and now_abs > max_abs:
            max_abs_col, max_abs = j, now_abs

    return max_gg_r_row, max_abs_col



class ChoiceMethod:
    def __init__(self, name, method):
        self.name = name
        self.method = method

    def next(self, matrix, p, q):
        return self.method(matrix, p, q)
 
    def __str__(self):
        return self.name


max_off_diag = ChoiceMethod("max_off_diag", __get_max_off_diag)
gg_opt = ChoiceMethod("gg_opt", __gg_opt)

