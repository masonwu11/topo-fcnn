import itertools
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
import pandas as pd
import os
import random
from sklearn.model_selection import KFold, StratifiedKFold
import top_clustering as tc
import loading_pipeline as lp
import models as model_architectures


# Set seed for reproducibility
def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def generate_hyperparam_sets(hyperparam):
    """Generate hyperparameter sets from cartesion products of hyperparameter entries

    Args:
        hyperparam: dictionary of hyperparameters with key as the hyperparameter name and value as a list of choices

    Returns:
        hyperparam_sets: list of dictionaries, each dictionary contains a set of hyperparameters
    """
    keys = [key for key in hyperparam if isinstance(hyperparam[key], list)]
    values = [hyperparam[key] for key in keys]
    combinations = list(itertools.product(*values))
    hyperparam_sets = [
        {
            **{k: hyperparam[k] for k in hyperparam if k not in keys},
            **{k: v for k, v in zip(keys, combination)},
        }
        for combination in combinations
    ]
    return hyperparam_sets


def train_loops(dataloader, model, loss_fn, optimizer, epoch, device="cpu"):
    """Train the model for a specified number of epochs

    Args:
        dataloader: data loader for the training dataset.
        model: model to be trained.
        loss_fn: loss function.
        optimizer: optimizer.
        epoch: number of epochs to train the model.
        device: device to run the model on. Defaults to "cpu".

    Returns:
        epoch_losses: list of average losses for each epoch.
    """
    model.train()
    epoch_losses = []
    for i in range(epoch):
        total_loss = 0
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        average_loss = total_loss / len(dataloader)
        epoch_losses.append(average_loss)
        # print(f"Epoch {i+1}, Average Loss: {average_loss}")
    return epoch_losses


def test_loop(dataloader, model, loss_fn, device="cpu"):
    """Test the model on the validation dataset

    Args:
        dataloader: data loader for the validation dataset.
        model: model to be tested.
        loss_fn: loss function.
        device: device to run the model on. Defaults to "cpu". Defaults to "cpu".

    Returns:
        correct: accuracy of the model on the validation dataset.
        test_loss: average loss of the model on the validation dataset.
    """
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Ensures that no gradients are computed during test mode
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct, test_loss


def k_fold_CV(
    train_dataset,
    hyperparams,
    k,
    dataset_name,
    task,
    stratification=False,
    batch_size=50,
    device="cpu",
    seed=42,
):
    """k-fold cross-validation for hyperparameter tuning

    Args:
        train_dataset: training dataset.
        hyperparams: dictionary of hyperparameters with key as the hyperparameter name and value as a list of choices.
        k: number of folds.
        dataset_name: name of the dataset. 
        task: task name. Used for logging.
        stratification: whether to use stratified k-fold. Defaults to False.
        batch_size: size of a batch. Defaults to 50.
        device: device to run the model on. Defaults to "cpu".
        seed: seed value for reproducibility. Defaults to 42.

    Returns:
        best_hyperparam: a dictionary of the best hyperparameters.
        best_accuracy: average accuracy of the model with the best hyperparameters.
    """
    kf = None
    splits = None
    if stratification:
        labels = train_dataset.tensors[1]
        if labels.is_cuda:
            labels = labels.cpu()
        labels = labels.numpy()
        kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        splits = list(kf.split(np.zeros(len(train_dataset)), labels))
    else:
        kf = KFold(n_splits=k, shuffle=True, random_state=seed)
        splits = list(kf.split(train_dataset))

    model_architecture = None
    if dataset_name == "mnist":
        model_architecture = model_architectures.MNIST_NN
    elif dataset_name == "fashion-mnist":
        model_architecture = model_architectures.Fashion_MNIST_NN
    elif dataset_name == "cifar10":
        model_architecture = model_architectures.VGGNet

    # Initialize variables
    best_accuracy = 0
    best_test_loss = 0
    best_hyperparam = {}
    accuracies = {}
    test_losses = {}

    # Generate hyperparameter sets
    hyperparam_sets = generate_hyperparam_sets(hyperparams)

    # Perform k-fold cross-validation
    for hyperparam in hyperparam_sets:
        fold_accuracies = []
        fold_test_losses = []
        print(f"Hyperparameters: {hyperparam}")
        for fold, (train_idx, val_idx) in enumerate(splits):
            print(f"Fold {fold+1}\n-------------------------------")
            # Creating data samplers and loaders:
            train_subsampler = Subset(train_dataset, train_idx)
            val_subsampler = Subset(train_dataset, val_idx)

            train_loader = DataLoader(
                train_subsampler, batch_size=batch_size, shuffle=True
            )
            val_loader = DataLoader(
                val_subsampler, batch_size=batch_size, shuffle=False
            )

            # Initialize model
            hyperparam_model = hyperparam.copy()
            epoch = hyperparam_model.pop("epochs")
            lamda = hyperparam_model.get("l2_lambda", None)
            model = model_architecture(**hyperparam_model).to(device)

            loss = nn.CrossEntropyLoss()
            # Initialize optimizer
            if lamda is None:
                optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
            else:
                optimizer = torch.optim.SGD(
                    model.parameters(), lr=0.01, momentum=0.9, weight_decay=lamda
                )
            # Train the model
            train_loops(train_loader, model, loss, optimizer, epoch, device)

            # Test the model
            accuracy, test_loss = test_loop(val_loader, model, loss, device)

            fold_accuracies.append(accuracy)
            fold_test_losses.append(test_loss)

        # Calculate average accuracy for the current hyperparameter set across all folds
        avg_accuracy = np.mean(fold_accuracies)
        accuracies[str(hyperparam)] = avg_accuracy
        # Calculate average test loss for the current hyperparameter set across all folds
        avg_test_loss = np.mean(fold_test_losses)
        test_losses[str(hyperparam)] = avg_test_loss

        # Update best model if the current set has higher average accuracy
        if avg_accuracy > best_accuracy:
            best_accuracy = avg_accuracy
            best_hyperparam = hyperparam
            best_test_loss = avg_test_loss

        print()

    log_cv(dataset_name, best_hyperparam, best_accuracy, best_test_loss, notes=task)

    return best_hyperparam, best_accuracy

def train_model(
    full_loader,
    best_hyperparam,
    index,
    dataset_name,
    task,
    regularization_type,
    class_label=None,
    device="cpu",
    save_model=True,
):
    """Train the model with the best hyperparameters on the full dataset (training and testing data combined)

    Args:
        full_loader: data loader for the full dataset.
        best_hyperparam: dictionary of the best hyperparameters.
        index: index of the model.
        dataset_name: name of the dataset. Used for logging.
        task: task name. Used for logging.
        regularization_type: type of regularization. Used for logging.
        class_label: class label for the dataset. Used for logging. Defaults to None.
        device: device to run the model on. Defaults to "cpu".
        save_model: whether to save the model. Defaults to True.

    Returns:
        model: trained model.
    """

    print(f"Training model #{index+1}")

    # Model architecture selection based on dataset
    model_architecture = None
    if dataset_name == "mnist":
        model_architecture = model_architectures.MNIST_NN
    elif dataset_name == "fashion-mnist":
        model_architecture = model_architectures.Fashion_MNIST_NN
    elif dataset_name == "cifar10":
        model_architecture = model_architectures.VGGNet

    # Initialize the model with the best hyperparameters
    model_param = best_hyperparam.copy()
    epoch = model_param.pop("epochs")
    lamda = model_param.get("l2_lambda", None)
    regularization_type = model_param.get("regularization_type", "unknown")
    model = model_architecture(**model_param).to(device)

    # Loss and optimizer setup
    loss = nn.CrossEntropyLoss()
    lamda = best_hyperparam.get("l2_lambda", None)
    if lamda is not None:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=0.01, momentum=0.9, weight_decay=lamda
        )
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Training loop
    train_loops(full_loader, model, loss, optimizer, epoch, device)

    # Model saving
    if save_model:
        if class_label is None:
            file_path = f"models/{task}/{dataset_name}/{regularization_type}/"
        else:
            file_path = (
                f"models/{task}/{dataset_name}/{regularization_type}/{class_label}/"
            )

        # Ensure the save directory exists
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        # Save the model
        model_filename = f"{file_path}{index}.pth"
        torch.save(
            (
                model.module.state_dict()
                if isinstance(model, nn.DataParallel)
                else model.state_dict()
            ),
            model_filename,
        )
        print(f"Model saved to {model_filename}")

    return model


def extract_neuron_values(dataset_name, model, graph_data, device="cpu"):
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
    if torch.isnan(graph_data).any():
        raise ValueError("Input data contains NaN values")
    activations = []
    graph_data = graph_data.to(device)
    model = model.to(device)

    def get_activation():
        def hook(model, input, output):
            activations.append(output.detach())

        return hook

    if dataset_name == "mnist" or dataset_name == "fashion-mnist":
        model.lrelu1.register_forward_hook(get_activation())
        model.lrelu2.register_forward_hook(get_activation())
    elif dataset_name == "cifar10":
        # names modified for CIFAR10
        model.lrelu_fc1.register_forward_hook(get_activation())
        model.lrelu_fc2.register_forward_hook(get_activation())
    else:
        print("Invalid dataset name for extracting neuron values")

    # Pass the graph data through the model
    model.eval()
    with torch.no_grad():
        model(graph_data)

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


def train_n_networks(
    n_networks,
    full_loader,
    best_hyperparam,
    graph_data,
    graph_labels,
    dataset_name,
    task,
    regularization_type,
    class_label=None,
    device="cpu",
    save_model=True,
    save_cm=True,
):
    """Train n networks and save correlation matrices

    Args:
        n_networks: number of networks to train.
        full_loader: data loader for the full dataset.
        best_hyperparam: dictionary of the best hyperparameters.
        graph_data: the graph data to pass through the model before extraction.
        graph_labels: the graph labels.
        dataset_name: the name of the dataset. Used for logging.
        task: the task name. Used for logging.
        regularization_type: the type of regularization. Used for logging.
        class_label: the class label to filter the dataset by. Defaults to None.
        device: the device to run the model on. Defaults to "cpu".
        save_model: whether to save the model. Defaults to True.
        save_cm: whether to save the correlation matrices. Defaults to True.
    """

    print(f"\nTraining {n_networks} networks on hyperparameters: {best_hyperparam} \n")

    networks = []
    neurons = []
    correlation_matrices = []
    for i in range(n_networks):
        network = train_model(
            full_loader,
            best_hyperparam,
            i,
            dataset_name,
            task,
            regularization_type,
            class_label=class_label,
            device=device,
            save_model=save_model,
        )
        networks.append(network)

    if save_cm:
        if class_label is None:
            cm_file_path = (
                f"correlation_matrices/{task}/{dataset_name}/{regularization_type}/"
            )
        else:
            cm_file_path = f"correlation_matrices/{task}/{dataset_name}/{regularization_type}/{class_label}/"
        # Filter the dataset by class if class_label is not None
        if class_label is not None:
            filtered_graph_data = lp.filter_data_by_class(
                graph_data, graph_labels, class_label
            )
        else:
            filtered_graph_data = graph_data
        for network in networks:
            neuron = extract_neuron_values(
                dataset_name, network, filtered_graph_data, device=device
            )
            neurons.append(neuron)
            correlation_matrices.append(correlation_graph(neuron))

        if not os.path.exists(cm_file_path):
            os.makedirs(cm_file_path)
            print(f"Directory created at {cm_file_path}")
        for i, cm in enumerate(correlation_matrices):
            cm_filename = f"{cm_file_path}{i}.npy"
            np.save(cm_filename, cm)
            print(f"Correlation matrix saved to {cm_filename}")
            del cm

    del networks, neurons, correlation_matrices


def log_cv(
    dataset_name,
    best_hyperparam,
    accuracy,
    loss,
    notes="",
    filename="logs/cv_log.csv",
):
    """Log cross-validation results to a csv file (append if file exists)

    Args:
        dataset_name: name of the dataset.
        best_hyperparam: dictionary of the best hyperparameters.
        accuracy: accuracy of the model.
        loss: loss of the model.
        notes: additional notes. Defaults to "".
        filename: name of the log file. Defaults to "logs/cv_log.csv".

    Returns:
        log: the log dataframe containing the results.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    log = pd.DataFrame(
        {
            "Timestamp": pd.Timestamp.now(),
            "Dataset": dataset_name,
            "Hyperparameters": str(best_hyperparam),
            "Accuracy": accuracy,
            "Loss": loss,
            "Notes": notes,
        },
        index=[0],
    )

    header = not os.path.exists(filename)

    log.to_csv(filename, mode="a", header=header, index=False)

    return log
