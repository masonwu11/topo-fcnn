import glob
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
import torch
import numpy as np
import pandas as pd
import os
import random
from matplotlib import pyplot as plt
import top_clustering as tc


# Set seed for reproducibility
def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

# Extract neuron outputs
def extract_neuron_values(dataset_name, model, data, device="cpu"):
    """Extract neuron values from

    Args:
        dataset_name: the name of the dataset.
        model: the trained model to extract neuron values from.
        graph_data: the graph data to pass through the model before extraction.
        device: the device to run the model on. Defaults to "cpu".

    Raises:
        ValueError: if the input data contains NaN values.

    Returns:
        neurons: the concatenated neuron values.
    """
    if torch.isnan(data).any():
        raise ValueError("Input data contains NaN values")
    activations = []

    def get_activation():
        def hook(model, input, output):
            activations.append(output.detach())

        return hook

    if dataset_name == "mnist" or dataset_name == "fashion-mnist":
        model.lrelu1.register_forward_hook(get_activation())
        model.lrelu2.register_forward_hook(get_activation())
    elif dataset_name == "cifar10":
        # modified for CIFAR10
        model.lrelu_fc1.register_forward_hook(get_activation())
        model.lrelu_fc2.register_forward_hook(get_activation())
    else:
        print("Invalid dataset name for extracting neuron values")

    model.eval()

    # Pass the graph data through the model
    with torch.no_grad():
        model(data)

    if any(torch.isnan(act).any() for act in activations):
        print("NaN values detected after activation layers")

    if device == "cuda":
        activations[0] = activations[0].cpu().numpy()
        activations[1] = activations[1].cpu().numpy()
    else:
        activations[0] = activations[0].numpy()
        activations[1] = activations[1].numpy()
    neurons = np.concatenate((activations[0].T, activations[1].T))
    return neurons


def correlation_graph(neurons):
    """Generate a correlation matrix for neurons

    Args:
        neurons: the neuron values.

    Returns:
        corr_matrix: the correlation matrix.
    """
    corr_matrix = abs(np.corrcoef(neurons))
    np.fill_diagonal(corr_matrix, 0)
    return corr_matrix

def purity_stats(iterations, top_clust, connectomes, labels_true):
    """Generate purity score for clustering connectomes

    Args:
        iterations: number of iterations to run.
        top_clust: top clustering object.
        connectomes: connectomes to cluster.
        labels_true: true labels.

    Returns:
        mean: mean purity score.
        std: standard deviation of purity scores.
    """
    scores = np.zeros(iterations)
    for i in range(iterations):
        labels_pred = top_clust.fit_predict(connectomes)
        scores[i] = tc.purity_score(labels_pred, labels_true)

    return np.mean(scores), np.std(scores)


def load_network(model_architecture, file_path, device="cpu"):
    """Load a network from a state_dict file

    Args:
        model_architecture: the model architecture to load the network into.
        file_path: the path to the state_dict file.
        device: the device to load the network on. Defaults to "cpu".

    Returns:
        network: the loaded network.
    """
    model_state = torch.load(file_path, map_location=device)
    network = model_architecture.to(device)
    network.load_state_dict(model_state)
    network.to(device)
    network.eval()
    del model_state
    return network


def load_cm_of_type(
    dataset_name,
    task,
    regularization_type,
    class_label=None,
    load_model=True,
    model_architecture=None,
    graph_data=None,
    device="cpu",
):
    """Load correlation matrices of a specific regularization type or class label

    Args:
        dataset_name: the name of the dataset. 
        task: the task name. 
        regularization_type: the regularization type of correlation matrices to load.
        class_label: the class label of correlation matrices to load. Defaults to None.
        load_model: whether to load the model to extract neuron values. Defaults to True.
        model_architecture: the model architecture to load the network into. Defaults to None.
        graph_data: the graph data to pass through the model before extraction. Defaults to None.
        graph_labels: the graph labels. Defaults to None.
        device: the device to load the network on. Defaults to "cpu".

    Returns:
        n_networks: the number of networks loaded.
        correlation_matrices: a list of loaded correlation matrices.
    """
    if load_model:
        correlation_matrices = []
        if class_label is None:
            networks_path = f"models/{task}/{dataset_name}/{regularization_type}/"
        else:
            networks_path = (
                f"models/{task}/{dataset_name}/{regularization_type}/{class_label}/"
            )

        pattern = os.path.join(networks_path, "*.pth")
        pth_files = glob.glob(pattern)

        n_networks = len(pth_files)
        print(f"Loading {n_networks} networks at {networks_path}")

        model_instance = model_architecture(regularization_type=regularization_type)

        for network_file in pth_files:
            network = load_network(model_instance, network_file, device=device)
            neuron = extract_neuron_values(
                dataset_name, network, graph_data, device=device
            )
            del network
            correlation_matrices.append(correlation_graph(neuron))
            del neuron

        if class_label is None:
            cm_file_path = (
                f"correlation_matrices/{task}/{dataset_name}/{regularization_type}/"
            )
        else:
            cm_file_path = f"correlation_matrices/{task}/{dataset_name}/{regularization_type}/{class_label}/"

        if not os.path.exists(cm_file_path):
            os.makedirs(cm_file_path)
            print(f"Directory created at {cm_file_path}")
        for i, cm in enumerate(correlation_matrices):
            cm_filename = f"{cm_file_path}{i}.npy"
            np.save(cm_filename, cm)
            print(f"Correlation matrix saved to {cm_filename}")

        del correlation_matrices, model_instance

    correlation_matrices = []

    if class_label is None:
        cms_path = f"correlation_matrices/{task}/{dataset_name}/{regularization_type}/"
    else:
        cms_path = f"correlation_matrices/{task}/{dataset_name}/{regularization_type}/{class_label}/"

    pattern = os.path.join(cms_path, "*.npy")
    npy_files = glob.glob(pattern)

    n_networks = len(npy_files)
    print(f"Loading {n_networks} correlation matrices at {cms_path}")

    for cm_file in npy_files:
        cm = np.load(cm_file)
        correlation_matrices.append(cm)

    return n_networks, correlation_matrices


def load_correlation_matrices(
    dataset_name,
    task,
    regularization_types,
    class_labels=None,
    load_model=True,
    model_architecture=None,
    graph_data=None,
    device="cpu",
):
    """Load correlation matrices of multiple regularization types or class labels

    Args:
        dataset_name: the name of the dataset. 
        task: the task name.
        regularization_types: the regularization types of correlation matrices to load.
        class_labels: the class labels of correlation matrices to load. Defaults to None.
        load_model: whether to load the model to extract neuron values. Defaults to True.
        model_architecture: the model architecture to load the network into. Defaults to None.
        graph_data: the graph data to pass through the model before extraction. Defaults to None.
        device: the device to load the network on. Defaults to "cpu".

    Returns:
        n_clusters: the number of clusters loaded.
        n_networks_per_cluster: the number of networks loaded per cluster.
        correlation_matrices: a list of loaded correlation matrices
    """
    correlation_matrices = []
    if class_labels == None:
        n_clusters = len(regularization_types)
        for regularization_type in regularization_types:
            n_networks_per_cluster, cms = load_cm_of_type(
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
            n_networks_per_cluster, cms = load_cm_of_type(
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

    return n_clusters, n_networks_per_cluster, correlation_matrices



def cluster_correlation_matrices(
    dataset_name,
    run_name,
    regularization_types,
    n_clusters,
    n_networks_per_cluster,
    correlation_matrices,
):
    """Cluster correlation matrices by regularization types

    Args:
        dataset_name: the name of the dataset.
        run_name: the name of the run.
        regularization_types: the regularization types of correlation matrices to cluster.
        n_clusters: the number of clusters.
        n_networks_per_cluster: the number of networks per cluster.
        correlation_matrices: the correlation matrices to cluster.

    Returns:
        top_weights: the topological weights used.
        purity: the purity scores.
    """
    print("\nClustering correlation matrices\n")
    max_iter_alt = 300
    max_iter_interp = 300
    # 0.05 for mnist, 0.01 for fashion-mnist, 0.01 for cifar-10
    learning_rate = 0.01

    # 20 for mnist, 20 for fashion-mnist, 20 for cifar-10
    iterations = 20
    labels_true = np.empty(n_clusters * n_networks_per_cluster)
    for i in range(n_clusters):
        labels_true[i * n_networks_per_cluster : (i + 1) * n_networks_per_cluster] = i

    top_weights = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.999]
    # top_weights = [0.999]  # DEBUG
    purity = []
    purity_means = []
    purity_stds = []
    for w in top_weights:
        print(f"Clustering for {iterations} iterations with topological weight: {w}")
        top_clust = tc.TopClustering(
            n_clusters, w, max_iter_alt, max_iter_interp, learning_rate
        )
        purity_entry = purity_stats(
            iterations, top_clust, correlation_matrices, labels_true
        )
        purity.append(np.asarray(purity_entry))
        purity_means.append(purity_entry[0])
        purity_stds.append(purity_entry[1])

    # Log purity scores to a csv file
    log_clustering(
        dataset_name,
        regularization_types,
        "None",
        "None",
        top_weights,
        purity_means,
        purity_stds,
        notes=run_name,
        filename="logs/clustering_log.csv",
    )

    return top_weights, purity



def graph_purity(dataset_name, task, run_name, top_weights, purity, save_graph=True):
    """Graph purity scores for clustering results as a line plot

    Args:
        dataset_name: the name of the dataset.
        task: the task name.
        run_name: the name of the run.
        top_weights: the topological weights used.
        purity: the purity scores.
        save_graph: whether to save the graph. Defaults to True.
    """
    plt.figure()
    purity = np.asarray(purity)
    purity_means = purity[:, 0]
    purity_stds = purity[:, 1]

    plt.scatter(top_weights, purity_means)
    plt.xlabel("Relative topological weight")
    plt.ylabel("Mean purity score")
    plt.title(f"Purity Scores for '{run_name}' Clustering Results")
    plt.ylim([0, 1.07])
    plt.errorbar(top_weights, purity_means, purity_stds)

    if save_graph == True:
        save_path = f"graphs/{task}/{dataset_name}/{run_name}_purity.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Graph saved at {save_path}")


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
    """_summary_

    Args:
        dataset_name: the name of the dataset.
        regularization_types: the regularization types of correlation matrices clustered.
        accuracy: the accuracy scores.
        losses: the loss values.
        topo_weights: the topological weights used.
        purity_mean: the purity scores mean.
        purity_std: the purity scores std.
        notes: additional notes. Defaults to "".
        filename: the filename to save the log to. Defaults to "logs/clustering_log.csv".

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


def classify_correlation_matrices(
    dataset_name,
    run_name,
    regularization_type,
    n_clusters,
    n_networks_per_cluster,
    correlation_matrices,
):
    """Cluster correlation matrices by predefined labels of a particular regularization type. Also calculate accuracy and confusion matrix.

    Args:
        dataset_name: the name of the dataset.
        run_name: the name of the run.
        regularization_type: the regularization type of correlation matrices to cluster.
        n_clusters: the number of clusters.
        n_networks_per_cluster: the number of networks per cluster.
        correlation_matrices: the correlation matrices to cluster.

    Returns:
        top_weights: the topological weights used.
        purity: the purity scores.
    """
    # Obtain predicted labels
    print("\nClassifying correlation matrices\n")
    max_iter_alt = 300
    max_iter_interp = 300
    # 0.05 for mnist, 0.01 for fashion-mnist, 0.01 for cifar-10
    learning_rate = 0.05

    # 50 for mnist, 50 for fashion-mnist, 50 for cifar-10
    iterations = 20
    labels_true = np.empty(n_clusters * n_networks_per_cluster)
    for i in range(n_clusters):
        labels_true[i * n_networks_per_cluster : (i + 1) * n_networks_per_cluster] = i

    top_weights = [0, 0.5, 0.999]
    # top_weights = [0.999]  # DEBUG
    predicted_labels = []
    confusion_matrices = []
    reassigned_labels = []
    purity_scores_list = []
    purity = []
    purity_means = []
    purity_stds = []
    for w in top_weights:
        print(f"Clustering for {iterations} iterations with topological weight: {w}")
        top_clust = tc.TopClustering(
            n_clusters, w, max_iter_alt, max_iter_interp, learning_rate
        )
        scores = np.zeros(iterations)
        for i in range(iterations):
            labels_pred = top_clust.fit_predict(correlation_matrices)
            # print("Predicted labels: ", labels_pred)

            predicted_labels.append(labels_pred)

            # Reassign labels based on majority vote within a cluster
            labels_reassigned = np.zeros(n_clusters * n_networks_per_cluster)

            for j in range(n_clusters):
                cluster_indices = np.where(labels_pred == j)[0]
                cluster_labels = labels_true[cluster_indices]
                unique_labels, counts = np.unique(cluster_labels, return_counts=True)
                majority_label = unique_labels[np.argmax(counts)]
                labels_reassigned[cluster_indices] = majority_label

            # Calculate confusion matrix
            confusion_matrices.append(
                confusion_matrix(
                    labels_true,
                    labels_reassigned,
                    labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                )
            )
            reassigned_labels.append(labels_reassigned)
            # Calculate purity stats
            scores[i] = tc.purity_score(labels_true, labels_pred)

        purity_scores_list.append(scores)
        purity.append(np.asarray(np.mean(scores), np.std(scores)))
        purity_means.append(np.mean(scores))
        purity_stds.append(np.std(scores))

    cm_file_path = f"confusion_matrices/{dataset_name}/{regularization_type}/"

    if not os.path.exists(cm_file_path):
        os.makedirs(cm_file_path)
        print(f"Directory created at {cm_file_path}")

    ps_file_path = f"logs/{run_name}/purity_scores/"
    if not os.path.exists(ps_file_path):
        os.makedirs(ps_file_path)
        print(f"Directory created at {ps_file_path}")

    print(f"Purity scores: ")
    print(purity_scores_list)
    np.save(
        f"{ps_file_path}{dataset_name}_{regularization_type}.npy", purity_scores_list
    )

    accuracies = []
    for k, cm in enumerate(confusion_matrices):
        weight = top_weights[k // iterations]
        idx = k % iterations
        cm_filename = f"{cm_file_path}{weight}_{idx}.npy"
        np.save(cm_filename, cm)
        print(f"Confusion matrix saved to {cm_filename}")
        # Calculate accuracy
        accuracy = balanced_accuracy_score(labels_true, reassigned_labels[k])
        accuracies.append(accuracy)

    log_clustering(
        dataset_name,
        regularization_type,
        accuracies,
        "None",
        top_weights,
        purity_means,
        purity_stds,
        notes=run_name,
        filename="logs/classification_log.csv",
    )

    return top_weights, purity

