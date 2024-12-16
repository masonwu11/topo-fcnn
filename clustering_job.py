import sys
import torch
import clustering_pipeline as cp
import loading_pipeline as lp
import models as model_architectures


def run_clustering(
    dataset_name,
    task,
    run_name,
    regularization_types,
    class_labels=None,
    graph_data=None,
    load_model=False,
    device="cpu",
):
    for regularization_type in regularization_types:
        print(f"\n########## Running {run_name} of {regularization_type} ##########\n")

        n_clusters, n_networks_per_cluster, correlation_matrices = (
            cp.load_correlation_matrices(
                dataset_name,
                task,
                regularization_type,
                class_labels=class_labels,
                load_model=load_model,
                model_architecture=model_architectures.Fashion_MNIST_NN,
                graph_data=graph_data,
                device=device,
            )
        )
        print(
            f"Loaded {n_clusters} clusters and {n_networks_per_cluster} networks per cluster"
        )
        top_weights, purity = cp.classify_correlation_matrices(
            dataset_name,
            run_name,
            regularization_type,
            n_clusters,
            n_networks_per_cluster,
            correlation_matrices,
        )
        print(f"########## Finished {run_name} of {regularization_type} ##########\n")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    param = "mnist"
    task = "clustering-classes"
    if len(sys.argv) > 1:
        param = sys.argv[1]
        print(f"Received parameter: {param}")
    else:
        print("No parameter received.")
    dataset_name = param
    print(f"\n########## Dataset {dataset_name} ##########\n")
    stratification = True
    class_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    cp.set_seed(42)
    (
        train_dataset,
        test_dataset,
        training_loader,
        testing_loader,
        graph_data,
        graph_labels,
    ) = lp.load_dataset(dataset_name, stratification=stratification, device=device)

    del train_dataset, test_dataset, training_loader, testing_loader

    run_clustering(
        dataset_name,
        task,
        "classification",
        ["vanilla", "batch_norm", "l2", "dropout"],
        class_labels=class_labels,
        graph_data=graph_data,
        load_model=False,
        device=device,
    )


if __name__ == "__main__":
    main()
