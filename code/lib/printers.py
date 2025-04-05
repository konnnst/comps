def print_matrix(matrix):
    for i in range(len(matrix)):
        for k in range(len(matrix[i])):
            print(f"{matrix[i][k]:6.4f}", end=" ")
        print()
