import math
import csv

def calc_corr(X):
    A, B, C, D, S, n = 0, 0, 0, 0, 0, 0
    for i in range(len(X)):
        for j in range(len(X)):
            S += X[i][j] * i * j
            A += X[i][j] * i
            B += X[i][j] * j
            C += X[i][j] * i * i
            D += X[i][j] * j * j
            n += X[i][j]
    r = (n * S - A * B) / (math.sqrt(n * C - A * A) * math.sqrt(n * D - B * B))
    return r


def get_from_csv(path):
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file)
        rows = list(csv_reader)
        m = rows[15:20]
        for i in range(len(m)):
            m[i].pop(0)
        for i in range(len(m)):
            m[i].pop(0)
    for i in range(len(m)):
        for j in range(len(m[i])):
            m[i][j] = float(m[i][j])
    return m