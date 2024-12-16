import numpy as np
import networkx as nx
from simulated_network_model import random_modular_graph
from matplotlib import pyplot as plt
from ripser import ripser
from persim import PersistenceImager, plot_diagrams

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.utils.metaestimators import _safe_split
from sklearn.svm import SVC
from sklearn.metrics import get_scorer


def adj2ripspers(adj):
    """inf for a single cc is removed"""
    # adj -> distance matrix
    dissimMtx = 1.0 / (adj + 1)  # from similarity to dissimilarity
    G = nx.from_numpy_array(dissimMtx)
    metricG = nx.algorithms.approximation.steinertree.metric_closure(
        G
    )  # metric graph with shortest metric
    distanceMtx = nx.linalg.graphmatrix.adjacency_matrix(metricG, weight="distance")

    # distance matrix -> Rips persistence diagram
    dgms = ripser(distanceMtx, distance_matrix=True)["dgms"]
    # dgms[0] = dgms[0][:-1] # remove inf persistence from CC's diagram

    return dgms


def ripspers2pimg2vec(dgms):
    """combine CC's and cycles into a feature vector"""
    # hyperparameter follows the original paper
    pimgr = PersistenceImager(
        pixel_size=0.05, kernel="gaussian", kernel_params={"sigma": 0.1}
    )
    pimgs = pimgr.transform(dgms, skew=True)
    return list(pimgs[0].flatten()) + list(pimgs[1].flatten())


def compute_cv_score(estimator, X, y, cv, scorer):
    avg_score = []
    for train, test in cv.split(X, y):
        X_train, y_train = _safe_split(estimator, X, y, train)
        X_test, y_test = _safe_split(estimator, X, y, test, train)
        estimator.fit(X_train, y_train)
        avg_score.append(scorer(estimator, X_test, y_test))

    print(estimator.cv_results_)

    return np.mean(avg_score)
