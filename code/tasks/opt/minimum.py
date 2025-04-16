class Minimum:
    def __init__(self, coords, grad_norm, iterations, elapsed_time, method):
        self.coords = coords
        self.grad_norm = grad_norm
        self.iterations = iterations
        self.elapsed_time = elapsed_time
        self.method = method

    def __str__(self):
        s = (f"{self.method}: "
             f"minimum = {[round(float(coord), 3) for coord in self.coords]}, gradient norm = {self.grad_norm:.3f}, "
             f"{self.iterations} iterations, {self.elapsed_time:.3f} seconds elapsed")
        return s

