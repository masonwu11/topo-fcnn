import traditional_clustering_pipeline as tcp


def run_clustering(dataset_name, task, run_name, distance_metric):
    distance_matrix = tcp.load_distance_matrix(
        dataset_name, task, run_name, distance_metric
    )
    n_sample = distance_matrix.shape[0]
    n_networks_per_cluster = 20
    n_clusters = n_sample // n_networks_per_cluster
    purity_mean, purity_std = tcp.k_medroid_clustering(
        distance_matrix, n_clusters, n_networks_per_cluster
    )
    print(f"Mean purity: {purity_mean}, std: {purity_std}")
    tcp.log_clustering(
        dataset_name,
        run_name,
        None,
        None,
        None,
        purity_mean,
        purity_std,
        notes=task + "-" + distance_metric,
        filename="logs/traditional_clustering_log.csv",
    )


if __name__ == "__main__":
    clustering_tasks = {
        "all_cluster": ["vanilla", "batch_norm", "l2", "dropout"],
        "batch_vanilla": ["vanilla", "batch_norm"],
        "l2_vanilla": ["vanilla", "l2"],
        "dropout_vanilla": ["vanilla", "dropout"],
        "batch_dropout_vanilla": ["vanilla", "batch_norm", "dropout"],
    }
    task = "clustering-classes"
    regularization_types = ["vanilla", "batch_norm", "l2", "dropout"]
    distance_metrics = ["bottleneck", "wasserstein", "sliced_wasserstein", "heat"]
    for regularization_type in regularization_types:
        print(f"\n########## Running {regularization_type} ##########\n")
        for distance_metric in distance_metrics:
            print(f"\n########## Distance Metric: {distance_metric} ########")
            run_clustering("cifar10", task, regularization_type, distance_metric)
