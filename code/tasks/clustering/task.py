from .utils import read_dataset, generate_dataset

from .kmeans import KMeans
from .hierarchical import Hierarchical

DATASET = "p4"

methods = {
    "kmeans": KMeans,
    "hierarchical": Hierarchical,
}


def run():
    # dataset = read_dataset(DATASET)
    dataset = generate_dataset(6, range(5, 20), 10, 100)
    print("Methods: ")

    commands = []
    for i, key in enumerate(methods, 1):
        print(f"{i}) {key}")
        commands.append(key)
    command_id = int(input()) - 1

    print("k:", end=" ")
    k = int(input())
    method = methods[commands[command_id]](k=k)
    method.fit(dataset)
    method.visualize(dataset)
