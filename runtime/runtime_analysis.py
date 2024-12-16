import numpy as np
from matplotlib import pyplot as plt
import datetime

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

from sklearn.metrics.pairwise import linear_kernel, euclidean_distances
from simulated_network_model import random_modular_graph

from persim import (
    PersistenceImager,
    bottleneck,
    wasserstein,
    sliced_wasserstein,
    heat,
)
from ripsLib import adj2ripspers

# alg = 'top'
# alg = 'pi' (PI euclidean)
# alg = 'sw' (PD sliced wasserstein)
# alg = 'wn' (PD wasserstein)
# alg = 'ht' (PD heat)
# alg = 'bk' (PD bottleneck)


def top(adj1, adj2):
    top1 = _vectorize_geo_top_info(adj1)
    top2 = _vectorize_geo_top_info(adj2)
    # top1 = top1.reshape(-1, 1)
    # top2 = top2.reshape(-1, 1)

    dist = np.linalg.norm(top1 - top2)
    return


def _vectorize_geo_top_info(adj):
    birth_set, death_set = _compute_birth_death_sets(adj)  # topological info
    return np.concatenate((birth_set, death_set), axis=0)


def _compute_birth_death_sets(adj):
    """Computes birth and death sets of a network."""
    mst, nonmst = _bd_demomposition(adj)
    birth_ind = np.nonzero(mst)
    death_ind = np.nonzero(nonmst)
    return np.sort(mst[birth_ind]), np.sort(nonmst[death_ind])


def _bd_demomposition(adj):
    """Birth-death decomposition."""
    eps = np.nextafter(0, 1)
    adj[adj == 0] = eps
    adj = np.triu(adj, k=1)
    Xcsr = csr_matrix(-adj)
    Tcsr = minimum_spanning_tree(Xcsr)
    mst = -Tcsr.toarray()  # reverse the negative sign
    nonmst = adj - mst
    return mst, nonmst


########## BASELINE ##########
# ADJ -> PD -> PI -> VEC


# def correlation_matrix_to_persistence(correlation_matrix, output_type="PD"):
#     distance_matrix = np.sqrt(abs(1 - correlation_matrix))
#     result = ripser(distance_matrix, distance_matrix=True)
#     h1 = result["dgms"][1]
#     if output_type == "PI":
#         pimgr = PersistenceImager(
#             pixel_size=0.05, birth_range=(0, 1), pers_range=(0, 1)
#         )
#         # print("Resolution before fit:", pimgr.resolution)
#         img = pimgr.transform(h1)
#         # print(img.shape)
#         # flatten the image
#         return img.flatten()
#     return h1


def correlation_matrix_to_persistence(correlation_matrix, output_type="PD"):
    h1 = adj2ripspers(correlation_matrix)[1]
    if output_type == "PI":
        pimgr = PersistenceImager(
            pixel_size=0.05, birth_range=(0, 1), pers_range=(0, 1)
        )
        img = pimgr.transform(h1, skew=True)
        # print(img.shape)
        return img.flatten()
    return h1


def pi(adj1, adj2):
    pd1 = correlation_matrix_to_persistence(adj1, output_type="PI")
    pd2 = correlation_matrix_to_persistence(adj2, output_type="PI")

    # pd1 = pd1.reshape(-1, 1)
    # pd2 = pd2.reshape(-1, 1)
    dist = np.linalg.norm(pd1 - pd2)
    # dist = euclidean_distances(np.vstack((pd1, pd2)))
    return


def sw(adj1, adj2):
    pd1 = correlation_matrix_to_persistence(adj1, output_type="PD")
    pd2 = correlation_matrix_to_persistence(adj2, output_type="PD")
    dist = sliced_wasserstein(pd1, pd2)
    return


def wn(adj1, adj2):
    pd1 = correlation_matrix_to_persistence(adj1, output_type="PD")
    pd2 = correlation_matrix_to_persistence(adj2, output_type="PD")
    dist = wasserstein(pd1, pd2)
    return


def ht(adj1, adj2):
    pd1 = correlation_matrix_to_persistence(adj1, output_type="PD")
    pd2 = correlation_matrix_to_persistence(adj2, output_type="PD")
    dist = heat(pd1, pd2)
    return


def bk(adj1, adj2):
    pd1 = correlation_matrix_to_persistence(adj1, output_type="PD")
    pd2 = correlation_matrix_to_persistence(adj2, output_type="PD")
    dist = bottleneck(pd1, pd2)
    return


# "top", "pi", "bk", "sw", "wn", "ht"
for alg in ["top"]:
    result = {}
    for nNode in [30, 60, 120, 250, 500, 1000]:

        # nNode = 120
        module1 = 3
        module2 = 5
        p = 0.55
        mu = 1
        sigma = 0.5

        adj1 = random_modular_graph(nNode, module1, p, mu, sigma)
        adj2 = random_modular_graph(nNode, module2, p, mu, sigma)

        number = 5

        # alg = 'top'
        # alg = 'pi' (PI euclidean)
        # alg = 'sw' (PD sliced wasserstein)
        # alg = 'wn' (PD wasserstein)
        # alg = 'ht' (PD heat)
        # alg = 'bk' (PD bottleneck)

        from timeit import Timer

        if alg == "top":
            t = Timer("top(adj1, adj2)", globals=globals())
            runtime = t.timeit(number)
            print(runtime)

        elif alg == "pi":
            t = Timer("pi(adj1, adj2)", globals=globals())
            runtime = t.timeit(number)
            print(runtime)

        elif alg == "sw":
            t = Timer("sw(adj1, adj2)", globals=globals())
            runtime = t.timeit(number)
            print(runtime)

        elif alg == "wn":
            t = Timer("wn(adj1, adj2)", globals=globals())
            runtime = t.timeit(number)
            print(runtime)

        elif alg == "ht":
            t = Timer("ht(adj1, adj2)", globals=globals())
            runtime = t.timeit(number)
            print(runtime)

        elif alg == "bk":
            t = Timer("bk(adj1, adj2)", globals=globals())
            runtime = t.timeit(number)
            print(runtime)

        result[nNode] = runtime / number
        result["alg"] = alg
        print(result)

        filename = (
            "runtime/"
            + datetime.datetime.now().strftime("%I_%M%p_on_%B_%d_%Y_")
            + alg
            + "_runtime_analysis.txt"
        )
        with open(filename, "w") as f:
            print(result, file=f)
