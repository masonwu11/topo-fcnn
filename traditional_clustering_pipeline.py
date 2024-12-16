from concurrent.futures import ProcessPoolExecutor
import os
import random
import sys
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from ripser import ripser
from persim import (
    plot_diagrams,
    PersistenceImager,
    bottleneck,
    wasserstein,
    sliced_wasserstein,
    heat,
)
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import contingency_matrix
from sklearn_extra.cluster import KMedoids
import clustering_pipeline as cp
import models as model_architectures
import top_clustering as tc


# Set seed for reproducibility
def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)


def correlation_matrix_to_persistence(correlation_matrix, output_type="PD"):
    """Converts a correlation matrix to a persistence diagram or persistence image (H1).

    Args:
        correlation_matrix: an adjacency matrix representing correlation.
        output_type: the type of output, either PD (persistence diagram) or PI (persistence image).

    Returns:
        persistence diagram or persistence image (flattened).
    """
    distance_matrix = np.sqrt(1 - correlation_matrix)
    result = ripser(distance_matrix, distance_matrix=True)
    h1 = result["dgms"][1]
    if output_type == "PI":
        pimgr = PersistenceImager(
            birth_range=(0, 1), pers_range=(0, 1), pixel_size=0.05
        )
        img = pimgr.transform(h1)
        # print(img.shape)
        # flatten the image
        return img.flatten()
    return h1


def load_persistence(
    dataset_name,
    task,
    regularization_types,
    class_labels=None,
    load_model=True,
    model_architecture=None,
    graph_data=None,
    output_type="PD",
    device="cpu",
):
    """Load correlation matrices and convert them to persistence diagrams or persistence images.

    Args:
        dataset_name: the name of the dataset.
        task: the name of the task.
        regularization_types: the regularization types of correlation matrices to load.
        class_labels: the class labels of correlation matrices to load. Defaults to None.
        load_model: whether to load the model. Defaults to True.
        model_architecture: the model architecture to use. Defaults to None.
        graph_data: the graph data to pass to the model. Defaults to None.
        output_type: the type of output, either PD (persistence diagram) or PI (persistence image). Defaults to "PD".
        device: the device to use. Defaults to "cpu".

    Returns:
        n_clusters: the number of clusters.
        n_networks_per_cluster: the number of networks per cluster.
        persistence: the persistence diagrams or persistence images.
        
    """
    correlation_matrices = []
    persistence = []
    if class_labels == None:
        n_clusters = len(regularization_types)
        for regularization_type in regularization_types:
            n_networks_per_cluster, cms = cp.load_cm_of_type(
                dataset_name,
                task,
                regularization_type,
                class_labels,
                load_model,
                model_architecture,
                graph_data,
                device,
            )
            correlation_matrices.extend(cms)
    else:
        n_clusters = len(class_labels)
        regularization_type = regularization_types
        for class_label in class_labels:
            n_networks_per_cluster, cms = cp.load_cm_of_type(
                dataset_name,
                task,
                regularization_type,
                class_label,
                load_model,
                model_architecture,
                graph_data,
                device,
            )
            correlation_matrices.extend(cms)

    for cm in correlation_matrices:
        persistence.append(correlation_matrix_to_persistence(cm, output_type))

    return n_clusters, n_networks_per_cluster, persistence


def calculate_distance(i, j, distance_metric, persistence):
    """Calculate the distance between two persistence diagrams or persistence images in a list.

    Args:
        i: an index in the list.
        j: an index in the list.
        distance_metric: the distance metric to use. Options are "bottleneck", "wasserstein", "sliced_wasserstein", "heat", and "euclidean".
        persistence: the list of persistence diagrams or persistence images.

    Returns:
        (i, j, dist): a tuple containing the indices and the distance between the persistence diagrams or persistence images.
    """
    # print(f"Calculating distance between PD {i} and {j}")
    if distance_metric == "bottleneck":
        dist = bottleneck(persistence[i], persistence[j])
    elif distance_metric == "wasserstein":
        dist = wasserstein(persistence[i], persistence[j])
    elif distance_metric == "sliced_wasserstein":
        dist = sliced_wasserstein(persistence[i], persistence[j])
    elif distance_metric == "heat":
        dist = heat(persistence[i], persistence[j])
    elif distance_metric == "euclidean":
        dist = np.linalg.norm(persistence[i] - persistence[j])
    return (i, j, dist)


def submit_all_distances(executor, n, distance_metric, persistence):
    """Submit all pairwise distances between persistence diagrams or persistence images to the executor.

    Args:
        executor: the executor to submit the tasks to.
        n: the number of persistence diagrams or persistence images.
        distance_metric: the distance metric to use.
        persistence: the list of persistence diagrams or persistence images.

    Returns:
        futures: a dictionary of futures.
    """
    futures = {}
    for i in range(n):
        for j in range(i, n):
            future = executor.submit(
                calculate_distance, i, j, distance_metric, persistence
            )
            futures[future] = (i, j)
    return futures


def persistence_to_distance_matrix(
    dataset_name, task, run_name, persistence, distance_metric="bottleneck"
):
    """Compute the distance matrix of a list of persistence diagrams or persistence images.

    Args:
        dataset_name: the name of the dataset.
        task: the name of the task.
        run_name: the name of the run.
        persistence: the list of persistence diagrams or persistence images.
        distance_metric: the distance metric to use. Options are "bottleneck", "wasserstein", "sliced_wasserstein", "heat", and "euclidean". Defaults to "bottleneck".

    Returns:
        distance_matrix: the distance matrix.
    """
    n = len(persistence)
    print(f"\nCalculating {distance_metric} distance matrix for {n} PD/PI\n")
    distance_matrix = np.zeros((n, n))
    with ProcessPoolExecutor() as executor:
        futures = submit_all_distances(executor, n, distance_metric, persistence)
        for future in futures:
            i, j, dist = future.result()
            distance_matrix[i, j] = dist
            if i != j:
                distance_matrix[j, i] = dist
    dm_file_path = f"distance_matrices/{task}/{dataset_name}/{run_name}/"
    if not os.path.exists(dm_file_path):
        os.makedirs(dm_file_path)
    np.save(dm_file_path + f"{distance_metric}.npy", distance_matrix)
    print(f"Distance matrix saved to {dm_file_path}")
    return distance_matrix

def k_medroid_clustering(distance_matrix, n_clusters, n_networks_per_cluster):
    """k-medoid clustering with distance matrix.

    Args:
        distance_matrix: the distance matrix.
        n_clusters: the number of clusters.
        n_networks_per_cluster: the number of networks per cluster.

    Returns:
        purity_mean: the mean purity score.
        purity_std: the standard deviation of the purity scores.
    """
    print(f"\nClustering Distance Matrix to {n_clusters} clusters with K-Medoids\n")
    # 20 for mnist, 20 for fashion-mnist, 20 for cifar-10
    iterations = 20
    labels_true = np.empty(n_clusters * n_networks_per_cluster)
    for i in range(n_clusters):
        labels_true[i * n_networks_per_cluster : (i + 1) * n_networks_per_cluster] = i

    scores = np.zeros(iterations)
    for j in range(iterations):
        kmedoids = KMedoids(
            n_clusters=n_clusters,
            metric="precomputed",
            init="random",
            method="pam",
            random_state=j,
        )
        kmedoids.fit(distance_matrix)
        labels_pred = kmedoids.labels_
        scores[j] = tc.purity_score(labels_true, labels_pred)
        print("Purity score: ", scores[j])
        print("Contingency matrix: ")
        print(contingency_matrix(labels_true, labels_pred))

    purity_mean = np.mean(scores)
    purity_std = np.std(scores)

    return purity_mean, purity_std



def k_mean_clustering(persistence, n_clusters, n_networks_per_cluster):
    """k-means clustering with persistence. (Used for flattened persistence images)

    Args:
        persistence: the persistence images.
        n_clusters: the number of clusters.
        n_networks_per_cluster: the number of networks per cluster.

    Returns:
        purity_mean: the mean purity score.
        purity_std: the standard deviation of the purity scores.
    """
    print(f"\nClustering Persistence to {n_clusters} clusters with K-Means\n")
    # 20 for mnist, 20 for fashion-mnist, 20 for cifar-10
    iterations = 20
    labels_true = np.empty(n_clusters * n_networks_per_cluster)
    for i in range(n_clusters):
        labels_true[i * n_networks_per_cluster : (i + 1) * n_networks_per_cluster] = i

    scores = np.zeros(iterations)
    for j in range(iterations):
        kmeans = KMeans(n_clusters=n_clusters, init="random", n_init=1, random_state=j)
        kmeans.fit(persistence)
        labels_pred = kmeans.labels_
        scores[j] = tc.purity_score(labels_true, labels_pred)
        print("Purity score: ", scores[j])
        print("Contingency matrix: ")
        print(contingency_matrix(labels_true, labels_pred))

    purity_mean = np.mean(scores)
    purity_std = np.std(scores)
    return purity_mean, purity_std


def log_clustering(
    dataset_name,
    regularization_types,
    accuracy,
    losses,
    topo_weights,
    purity_mean,
    purity_std,
    notes="",
    filename="logs/clustering_log.csv",
):
    """Log cross-validation results to a csv file (append if file exists)

    Args:
        dataset_name: the name of the dataset.
        regularization_types: regularization types used.
        accuracy: accuracy of the model.
        losses: losses of the model.
        topo_weights: topological weights.
        purity_mean: mean purity score.
        purity_std: standard deviation of purity scores.
        notes: additional notes. Defaults to "".
        filename: name of the log file. Defaults to "logs/clustering_log.csv".

    Returns:
        log: the log dataframe containing the results.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    log = pd.DataFrame(
        {
            "Timestamp": pd.Timestamp.now(),
            "Dataset": dataset_name,
            "Regularization types": str(regularization_types),
            "Accuracy": str(accuracy),
            "Loss": str(losses),
            "Topological weights": str(topo_weights),
            "Purity mean": str(purity_mean),
            "Purity std": str(purity_std),
            "Notes": notes,
        },
        index=[0],
    )

    header = not os.path.exists(filename)

    log.to_csv(filename, mode="a", header=header, index=False)

    return log


def load_distance_matrix(dataset_name, task, run_name, distance_metric):
    """Load a distance matrix from a file.

    Args:
        dataset_name: the name of the dataset.
        task: the name of the task.
        run_name: the name of the run.
        distance_metric: the distance metric used.

    Returns:
        distance_matrix: the distance matrix.
    """
    dm_file_path = f"distance_matrices/{task}/{dataset_name}/{run_name}/"
    distance_matrix = None
    # file name maybe {distance_metric}.npy or {distance_metric}_distance_matrix.npy
    if os.path.exists(dm_file_path + f"{distance_metric}.npy"):
        print(f"Loading distance matrix from {dm_file_path}{distance_metric}.npy")
        distance_matrix = np.load(dm_file_path + f"{distance_metric}.npy")
    elif os.path.exists(dm_file_path + f"{distance_metric}_distance_matrix.npy"):
        print(
            f"Loading distance matrix from {dm_file_path}{distance_metric}_distance_matrix.npy"
        )
        distance_matrix = np.load(
            dm_file_path + f"{distance_metric}_distance_matrix.npy"
        )
    else:
        print(f"Distance matrix not found in {dm_file_path}")
    return distance_matrix


if __name__ == "__main__":
    param = "mnist"
    if len(sys.argv) > 1:
        param = sys.argv[1]
        print(f"Received parameter: {param}")
    else:
        print("No parameter received.")
    dataset_name = param

    print(f"\n########## Dataset {dataset_name} ##########\n")
    task = "clustering-classes"
    regularization_types = ["vanilla", "batch_norm", "l2", "dropout"]
    class_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # class_labels = None
    set_seed(42)
    clustering_tasks = {
        "all_cluster": ["vanilla", "batch_norm", "l2", "dropout"],
        "batch_vanilla": ["vanilla", "batch_norm"],
        "l2_vanilla": ["vanilla", "l2"],
        "dropout_vanilla": ["vanilla", "dropout"],
        "batch_dropout_vanilla": ["vanilla", "batch_norm", "dropout"],
    }
    for regularization_type in regularization_types:
        print(f"\n########## Run {regularization_type} ##########\n")
        n_clusters, n_networks_per_cluster, persistence = load_persistence(
            dataset_name,
            task,
            regularization_type,
            class_labels=class_labels,
            load_model=False,
            model_architecture=model_architectures.MNIST_NN,
            output_type="PI",
            device="cpu",
        )
        purity_mean, purity_std = k_mean_clustering(
            persistence, n_clusters, n_networks_per_cluster
        )
        print(f"Mean purity: {purity_mean}, std: {purity_std}")
        log_clustering(
            dataset_name,
            regularization_type,
            None,
            None,
            None,
            purity_mean,
            purity_std,
            notes=task + "-euclidean",
            filename="logs/traditional_clustering_log.csv",
        )
