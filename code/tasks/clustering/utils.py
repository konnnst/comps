from numpy import array
from random import shuffle, choice, random


def read_dataset(dataset_name):
    with open(f"tests/{dataset_name}") as points:
        data = []
        for point in points.readlines():
            data.append([int(coord) for coord in point.split()])
        return array(data)


def generate_dataset(group_count, group_size_range, group_radius, radius, dim=3):
    points = []

    for _ in range(group_count):
        count = choice(group_size_range) - 1
        center = [random() * radius for _ in range(dim)]
        points.append(center)
        for _ in range(count):
            point = [
                    center[i] + choice([-1, 1]) * random() * group_radius \
                    for i in range(dim)
            ]
            points.append(point)

    shuffle(points)
    return array(points)

