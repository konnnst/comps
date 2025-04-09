from scipy.optimize import approx_fprime 

class Expression:
    def __init__(self, string, function, dim):
        self._string = string
        self._function = function
        self._dim = dim

    def __call__(self, *args):
        return self._function(*args)
    
    def __str__(self):
        return self._string

    def dim(self):
        return self._dim

    def grad(self, point):
        return approx_fprime(point, self._function)

