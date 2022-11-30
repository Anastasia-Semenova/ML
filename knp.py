import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def init_graph(n):
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if np.random.randint(0, 2) == 1:
                matrix[i][j] = matrix[j][i] = np.random.randint(0, 100)
    return matrix


def dijkstra(matrix, node):
    distances = np.ones(matrix.shape[0]) * -1
    edges = {node: [node]}
    distances[node] = 0
    for c in range(matrix.shape[0]):
        for i in range(matrix.shape[0]):
            if distances[i] == -1:
                continue
            for j in range(matrix.shape[0]):
                if matrix[i, j] == 0:
                    continue
                new_dist = distances[i] + matrix[i, j]
                if distances[j] == -1 or new_dist < distances[j]:
                    distances[j] = new_dist
                    path = edges[i].copy()
                    path.append(j)
                    edges[j] = path
                    continue
    return edges


def tree(graph):
    matrix = nx.to_numpy_matrix(graph)
    i = -1
    j = -1
    min = -1
    for x in range(matrix.shape[0]):
        for y in range(matrix.shape[1]):
            if matrix[x, y] > 0 and (min == -1 or matrix[i, j] < min):
                min = matrix[i, j]
                i = x
                j = y
    paths = dijkstra(matrix, i)
    edges = set()
    for key, val in paths.items():
        for i in range(len(val) - 1):
            edges.add((val[i], val[i + 1]))
            edges.add((val[i + 1], val[i]))
    for x in range(matrix.shape[0]):
        for y in range(matrix.shape[0]):
            if (x, y) not in edges:
                matrix[x, y] = 0
                matrix[y, x] = 0
    return nx.from_numpy_matrix(matrix)


def remove_max(graph):
    matrix = nx.to_numpy_matrix(graph)
    i = -1
    j = -1
    max = -1
    for x in range(matrix.shape[0]):
        for y in range(matrix.shape[1]):
            if matrix[x, y] > 0 and (max == -1 or matrix[i, j] > max):
                max = matrix[i, j]
                i = x
                j = y
    matrix[i, j] = 0
    matrix[j, i] = 0
    return nx.from_numpy_matrix(matrix)


if __name__ == '__main__':
    n = 15

    # граф
    matrix = init_graph(n)
    G = nx.from_numpy_matrix(matrix)
    pos = nx.spring_layout(G, seed=7)
    nx.draw_networkx(G)
    plt.show()

    # минимальное дерево
    G = tree(G)
    pos = nx.spring_layout(G, seed=2)
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_edges(G, pos)
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels)
    plt.show()

    # кластеры
    G = remove_max(G)
    pos = nx.spring_layout(G, seed=2)
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_edges(G, pos)
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels)
    plt.show()