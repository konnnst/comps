class Eigenvector:
    def __init__(self, vector, value, iterations):
        self.vector = vector
        self.value = value
        self.iterations = iterations

    def __str__(self):
        return f"\tVector: {self.vector}\n\tValue: {self.value}\n\tIterations: {self.iterations}"
