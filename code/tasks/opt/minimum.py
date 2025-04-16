class Minimum:
    def __init__(self, point, method, epsilon, iterations):
        self.point = point
        self.method = method
        self.epsilon = epsilon
        self.iterations = iterations

    def __str__(self):
        return (f"{self.method}: {self.point}, eps: {self.epsilon}, "
                f"{self.iterations} iters")
