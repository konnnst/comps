from numpy.linalg import solve, norm


class Solution:
    def __init__(self, x, method, epsilon):
        self.x = x
        self.method = method
        self.epsilon = epsilon

    def __str__(self):
        return f"{self.method}: " \
                f"x = {[round(float(self.x[i]), 3) for i in range(len(self.x))]}, " \
               f"eps = {self.epsilon}"

    def distance(self, solution):
        return norm(self.x - solution.x)
