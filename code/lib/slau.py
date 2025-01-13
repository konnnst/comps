from numpy import linalg


class Problem:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __str__(self):
        res = ""
        for i in range(len(self.a)):
            for k in range(len(self.a[0])):
                res += f"{self.a[i][k]:6.2f} "
            res += f"| {self.b[i]:6.2f}\n"

        return res

    def solve(self):
        return linalg.solve(self.a, self.b)

    def get_solution_delta(self, solution):
        real_solution = linalg.solve(self.a, self.b)
        delta = linalg.norm(solution - real_solution)
        return delta
