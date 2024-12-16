import torch
import loading_pipeline as lp
import training_pipeline as tp
import sys


def run(
    dataset_name,
    task,
    stratification,
    hyperparameters,
    class_labels=None,
    k=5,
    n_networks=20,
    device="cpu",
):
    (
        train_dataset,
        test_dataset,
        training_loader,
        testing_loader,
        graph_data,
        graph_labels,
    ) = lp.load_dataset(dataset_name, stratification=stratification, device=device)
    regularization_types = ["vanilla", "batch_norm", "l2", "dropout"]
    regularization_types = ["dropout"]
    for regularization_type in regularization_types:
        print(f"\n########## Training {regularization_type} ##########\n")

        hyperparams = hyperparameters[regularization_type]
        print("\nTuning hyperparameters\n")
        best_hyperparam, best_accuracy = tp.k_fold_CV(
            train_dataset,
            hyperparams,
            k,
            dataset_name,
            task,
            stratification=stratification,
            device=device,
        )

        print(f"Best hyperparam for {regularization_type}: ", best_hyperparam)
        print(f"Best accuracy for {regularization_type}: ", best_accuracy)
        # best_hyperparam = best_hyperparam_dict[(dataset_name, regularization_type)]

        # print(f"Best hyperparam for {regularization_type}: ", best_hyperparam)
        if class_labels != None:
            for class_label in class_labels:
                print(f"\nTraining {n_networks} networks for class {class_label}\n")
                full_dataset, full_loader = lp.combine_data(train_dataset, test_dataset)
                tp.train_n_networks(
                    n_networks,
                    full_loader,
                    best_hyperparam,
                    graph_data,
                    graph_labels,
                    dataset_name,
                    task,
                    regularization_type,
                    class_label=class_label,
                    device=device,
                )
                del full_dataset, full_loader
        else:
            full_dataset, full_loader = lp.combine_data(train_dataset, test_dataset)
            print(f"\nTraining {n_networks} networks\n")
            tp.train_n_networks(
                n_networks,
                full_loader,
                best_hyperparam,
                graph_data,
                graph_labels,
                dataset_name,
                task,
                regularization_type,
                class_label=None,
                device=device,
            )
        print(f"\n########## Finished Training {regularization_type} ##########\n")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    param = "cifar10"
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
    # class_labels = None

    hyperparameters = {
        "vanilla": {
            "regularization_type": "vanilla",
            "epochs": [20, 30, 40],
            "alpha": [0.01, 0.1],
        },
        "batch_norm": {
            "regularization_type": "batch_norm",
            "epochs": [20, 30, 40],
            "alpha": [0.01, 0.1],
        },
        "l2": {
            "regularization_type": "l2",
            "epochs": [20, 30, 40],
            "l2_lambda": [0.0001, 0.001, 0.005, 0.01],
            "alpha": [0.01, 0.1],
        },
        "dropout": {
            "regularization_type": "dropout",
            "epochs": [20, 30, 40],
            "dropout_rate": [0.1, 0.2, 0.3, 0.4],
            "alpha": [0.01, 0.1],
        },
    }

    tp.set_seed(42)
    run(
        dataset_name,
        task,
        stratification,
        hyperparameters,
        class_labels=class_labels,
        device=device,
    )

    print(f"\n########## Completed {dataset_name} ##########\n")
