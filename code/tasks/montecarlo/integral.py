from copy import copy
from random import random
from scipy.integrate import nquad


class Integral:
    def __init__(self, string, f, limits):
        self.f = f
        self.limits = limits
        self.string = string

    def __str__(self):
        return f"{self.string}, limits: " \
               f"{[[round(l[0], 6), round(l[1], 6)] for l in self.limits]}"

    def library(self):
        return nquad(self.f, self.limits)

    def __global_extremum(self, comp):
        start = [limit[0] for limit in self.limits]
        end = [limit[1] for limit in self.limits]
        steps = [(limit[1] - limit[0]) / 100 for limit in self.limits]
        now = copy(start)

        extremum = 1e10
        if comp(1, 0):
            extremum = -extremum

        while True:
            if comp(self.f(*now), extremum):
                extremum = self.f(*now)

            degree = 0
            while degree < len(self.limits):
                now[degree] += steps[degree]
                if now[degree] > end[degree]:
                    now[degree] = start[degree]
                    degree += 1
                else:
                    break
            else:
                break

        return extremum

    def monte_carlo(self, count=100000):
        maximum = self.__global_extremum(lambda x, y: x > y)
        minimum = self.__global_extremum(lambda x, y: x < y)
        upper_bound = maximum + abs(maximum) / 4
        lower_bound = minimum - abs(minimum) / 4

        volume = upper_bound - lower_bound
        for limit in self.limits:
            volume *= limit[1] - limit[0]

        fit_count = 0
        for _ in range(count):
            points = []
            for limit in self.limits:
                coord = limit[0] + (limit[1] - limit[0]) * random()
                points.append(coord)
            y = lower_bound + (upper_bound - lower_bound) * random()
            try:
                if 0 <= y <= self.f(*points):
                    fit_count += 1
                elif self.f(*points) <= y <= 0:
                    fit_count -= 1
            except ArithmeticError:
                pass

        integral = volume * fit_count / count
        return integral
