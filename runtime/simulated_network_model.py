import numpy as np
from matplotlib import pyplot as plt
import math
import random


def random_modular_graph(d, c, p, mu, sigma):

    # adjacency matrix
    adj = np.zeros((d, d))

    for i in range(1, d + 1):
        for j in range(i + 1, d + 1):
            module_i = math.ceil(c * i / d)
            module_j = math.ceil(c * j / d)

            if module_i == module_j:
                # probability of attachment within module
                if random.uniform(0, 1) <= p:
                    w = np.random.normal(mu, sigma)
                    adj[i - 1][j - 1] = abs(w)
                    adj[j - 1][i - 1] = abs(w)
                else:
                    # noise
                    w = np.random.normal(0, sigma)
                    adj[i - 1][j - 1] = abs(w)
                    adj[j - 1][i - 1] = abs(w)
            else:
                # probability of attachment between modules
                if random.uniform(0, 1) <= 1 - p:
                    w = np.random.normal(mu, sigma)
                    adj[i - 1][j - 1] = abs(w)
                    adj[j - 1][i - 1] = abs(w)
                else:
                    w = np.random.normal(0, sigma)
                    adj[i - 1][j - 1] = abs(w)
                    adj[j - 1][i - 1] = abs(w)
    return adj


if __name__ == "__main__":
    d = 60
    c = 5
    p = 0.9
    mu = 1
    sigma = 0.5
    adj = random_modular_graph(d, c, p, mu, sigma)
    plt.imshow(adj)
    plt.colorbar()
    plt.show()
