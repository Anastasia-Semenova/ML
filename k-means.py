import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv
import random

N = 88
x = np.random.randint(1, 100, N)
y = np.random.randint(1, 100, N)

x_c = np.mean(x)
y_c = np.mean(y)
wcss = []


def k_means(k):
    R = 0
    for i in range(0, N):
        if dist(x_c, y_c, x[i], y[i]) > R:
            R = dist(x_c, y_c, x[i], y[i])
    x_cc = [R * np.cos(2 * np.pi * i / k) + x_c for i in range(k)]
    y_cc = [R * np.sin(2 * np.pi * i / k) + y_c for i in range(k)]
    cluster = [0] * N
    clusterize(x_cc, y_cc, x, y, k, cluster)
    draw(x, y, cluster, x_cc, y_cc, k)
    check(x, y, x_cc, y_cc, cluster, k)


def mean_clusters(x, y, clusters, k, x_cc, y_cc):
    for i in range(0, k):
        z_x, z_y = [], []
        for j in range(0, len(clusters)):
            if clusters[j] == i:
                z_x.append(x[j])
                z_y.append(y[j])
                draw(x, y, clusters, x_cc, y_cc, k)
        x_cc[i] = np.mean(z_x)
        y_cc[i] = np.mean(z_y)


def draw(x, y, clusters, x_cc, y_cc, k):
    cluster_length = len(clusters)
    for i in range(0, cluster_length):
        clr = (clusters[i] + 1) / k
        plt.scatter(x[i], y[i], color=(clr, 0.2, clr ** 2))
    plt.scatter(x_cc, y_cc, color='orange')
    plt.show()


def check(x, y, x_cc, y_cc, clust, k):
    x_old, y_old = x_cc, y_cc
    clusterize(x_cc, y_cc, x, y, k, clust)
    mean_clusters(x, y, clust, k, x_cc, y_cc)
    draw(x, y, clust, x_cc, y_cc, k)
    if x_old == x_cc and y_old == y_cc:
        wss = 0
        for i in range(0, k):
            clusterSum = 0
            for j in range(0, len(clust)):
                if clust[j] == i:
                    clusterSum += dist(x[j], y[j], x_cc[i], y_cc[i]) ** 2
            wss += clusterSum
        wcss.append(wss)
        return True
    else:
        return False


def clusterize(x_cc, y_cc, x, y, k, cluster):
    for i in range(0, N):
        r = dist(x_cc[0], y_cc[0], x[i], y[i])
        numb = 0
        for j in range(0, k):
            if r < dist(x_cc[j], y_cc[j], x[i], y[i]):
                numb = j
                r = dist(x_cc[j], y_cc[j], x[i], y[i])
            if j == k - 1:
                cluster[i] = numb


def dist(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


for i in range(1, 10):
    k_means(i)


def k_min(wcss):
    arr = []
    for i in range(1, len(wcss) - 1):
        arr.append((wcss[i] - wcss[i + 1]) / (wcss[i - 1] - wcss[i]))
    return arr.index(min(arr))


plt.plot(range(1, 10), wcss)
plt.xlabel('Cluster Num')
plt.ylabel('WCSS')
plt.show()
print(k_min(wcss))


def dist(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def draw(x, y, cntr_x=[], cntr_y=[]):
    plt.scatter(x, y, color='r')
    if len(cntr_x) != 0:
        plt.scatter(cntr_x, cntr_y, marker="+")
    plt.show()


def random_points(n):
    x = np.random.uniform(0, 100, n)
    y = np.random.uniform(0, 100, n)
    return x, y


def init_centroids(x, y, k):
    x_c, y_c = x.mean(), y.mean()
    R = 0
    for i in range(len(x)):
        d = dist(x_c, y_c, x[i], y[i])
        if R < d:
            R = d
    cntr_x, cntr_y = [], []
    for i in range(k):
        cntr_x.append(R * np.cos(2 * np.pi * i / k) + x_c)
        cntr_y.append(R * np.sin(2 * np.pi * i / k) + y_c)
    return cntr_x, cntr_y


def recalculate_probs(x, y, cntr_x, cntr_y, n):
    s = 2 / (1 - m)
    probs = []
    for i in range(len(x)):
        for j in range(len(cntr_x)):
            probs[i][j] = dist(x[i], y[i], cntr_x[j], cntr_y[j]) ** s
            sum += dist(x[i], y[i], cntr_x[j], cntr_y[j]) ** s
        probs[i] /= sum
    return probs


def recalculate(x, y, probs, m):
    cntr_x, cntr_y = np.zeros(len(probs)), np.zeros(len(probs))
    for i in range(len(probs)[1]):
        p = 0
        for j in range(len(x)):
            cntr_x[j] += x[i] * probs[i][j] ** m
            cntr_y[j] += y[i] * probs[i][j] ** m
            p += probs[i][j] ** m
        cntr_x[j] /= p
        cntr_y[j] /= p
    return cntr_x, cntr_y


def get_cluster_by_point(probs):
    cluster = []
    for row in probs:
        max, index = -1, -1
        for i in range(len(row)):
            if max < row[i]:
                map = row[i]
                index = i
        cluster.append(index)
    return cluster


if __name__ == "__main__":
    n = 100
    x, y = random_points(100)
    draw(x, y)
    k = 4
    cntr_x, cntr_y = init_centroids(x, y, k)
    draw(x, y, cntr_x, cntr_y)
    m = 11
    probs = recalculate_probs(x, y, cntr_x, cntr_y, m)
    get_cluster_by_point(probs)
