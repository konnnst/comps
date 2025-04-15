from scipy.optimize import approx_fprime 
from scipy.differentiate import hessian

class Expression:
    def __init__(
            self,
            string,
            function,
            dim,
            expected_minimum,
            alpha=0.01,
            beta=0.01,
            gamma=0.01,
        ):
        self._string = string
        self._function = function
        self._dim = dim
        self._expected_minimum = expected_minimum
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def __call__(self, *args):
        return self._function(*args)
    
    def __str__(self):
        return f"Expression: {self._string}, minimum: {self._expected_minimum}"


    def dim(self):
        return self._dim

    def grad(self, point):
        return approx_fprime(point, self._function)

    def hess(self, point):
        return hessian(self._function, point).ddf
