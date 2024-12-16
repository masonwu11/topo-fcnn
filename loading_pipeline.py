import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Subset, ConcatDataset
import os
import random
from sklearn.model_selection import StratifiedShuffleSplit
from torchvision import datasets, transforms


# Set seed for reproducibility
def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def load_mnist(
    graph_data_proportion=1 / 6, batch_size=50, stratification=False, device="cpu"
):
    """Import MNIST dataset

    Args:
        graph_data_proportion: the proportion of the training dataset to be used for generating functional connectomes. Defaults to 1/6.
        batch_size: the number of samples in each batch. Defaults to 50.
        stratification: if True, the loaded data will be stratified. Defaults to False.
        device: the device to move the data to. Defaults to "cpu".

    Returns:
        train_dataset: the training dataset.
        test_dataset: the testing dataset.
        training_loader: the training data loader.
        testing_loader: the testing data loader.
        graph_data: the data used for generating functional connectomes.
        graph_labels: the labels of the data used for generating functional connectomes.
    """
    # Download the MNIST dataset if it is not already downloaded
    if not os.path.exists("./data/MNIST/raw"):
        download = True
    else:
        download = False

    trainset = datasets.MNIST(
        root="./data", train=True, download=download, transform=transforms.ToTensor()
    )
    testset = datasets.MNIST(
        root="./data", train=False, download=download, transform=transforms.ToTensor()
    )

    # Convert dataset to PyTorch tensors along rows
    train_data = torch.cat([data[0].view(1, 28 * 28) for data in trainset], dim=0)
    if stratification == True:
        train_labels = trainset.targets.numpy()
    else:
        train_labels = trainset.targets
    test_data = torch.cat([data[0].view(1, 28 * 28) for data in testset], dim=0)
    test_labels = testset.targets

    if stratification == True:
        # Apply stratified sampling to the training data
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=graph_data_proportion, random_state=42
        )
        train_idx, graph_idx = next(sss.split(train_data, train_labels))
        graph_data = train_data[graph_idx]
        graph_labels = torch.tensor(train_labels[graph_idx])
        train_data = train_data[train_idx]
        train_labels = torch.tensor(train_labels[train_idx])
    else:
        # Sample data for generating connectomes
        indices = torch.randperm(len(train_data))[
            : int(graph_data_proportion * len(train_data))
        ]
        graph_data = train_data[indices]
        graph_labels = train_labels[indices]
        # Create a mask for all data points
        mask = torch.ones(len(train_data), dtype=torch.bool)
        mask[indices] = False
        # Use the mask to select the remaining data
        train_data = train_data[mask]
        train_labels = train_labels[mask]

    # Move data to device
    train_data = train_data.to(device)
    train_labels = train_labels.to(device)
    test_data = test_data.to(device)
    test_labels = test_labels.to(device)
    graph_data = graph_data.to(device)
    graph_labels = graph_labels.to(device)

    # Wrap tensors in TensorDataset
    train_dataset = TensorDataset(train_data, train_labels)
    test_dataset = TensorDataset(test_data, test_labels)

    # Create data loaders for our datasets; shuffle for training, not for validation
    training_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    testing_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    return (
        train_dataset,
        test_dataset,
        training_loader,
        testing_loader,
        graph_data,
        graph_labels,
    )


def load_fashion_mnist(
    graph_data_proportion=1 / 6, batch_size=50, stratification=False, device="cpu"
):
    """Import Fashion-MNIST dataset

    Args:
        graph_data_proportion: the proportion of the training dataset to be used for generating functional connectomes. Defaults to 1/6.
        batch_size: the number of samples in each batch. Defaults to 50.
        stratification: if True, the loaded data will be stratified. Defaults to False.
        device: the device to move the data to. Defaults to "cpu".

    Returns:
        train_dataset: the training dataset.
        test_dataset: the testing dataset.
        training_loader: the training data loader.
        testing_loader: the testing data loader.
        graph_data: the data used for generating functional connectomes.
        graph_labels: the labels of the data used for generating functional connectomes.
    """
    # Download the Fashion MNIST dataset if it is not already downloaded
    if not os.path.exists("./data/FashionMNIST/raw"):
        download = True
    else:
        download = False

    trainset = datasets.FashionMNIST(
        root="./data", train=True, download=download, transform=transforms.ToTensor()
    )
    testset = datasets.FashionMNIST(
        root="./data", train=False, download=download, transform=transforms.ToTensor()
    )

    # Convert dataset to PyTorch tensors along rows
    train_data = torch.cat([data[0].view(1, 28 * 28) for data in trainset], dim=0)
    if stratification == True:
        train_labels = trainset.targets.numpy()
    else:
        train_labels = trainset.targets
    test_data = torch.cat([data[0].view(1, 28 * 28) for data in testset], dim=0)
    test_labels = testset.targets

    if stratification == True:
        # Apply stratified sampling to the training data
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=graph_data_proportion, random_state=42
        )
        train_idx, graph_idx = next(sss.split(train_data, train_labels))
        graph_data = train_data[graph_idx]
        graph_labels = torch.tensor(train_labels[graph_idx])
        train_data = train_data[train_idx]
        train_labels = torch.tensor(train_labels[train_idx])
    else:
        # Sample data for generating connectomes
        indices = torch.randperm(len(train_data))[
            : int(graph_data_proportion * len(train_data))
        ]
        graph_data = train_data[indices]
        graph_labels = train_labels[indices]
        # Create a mask for all data points
        mask = torch.ones(len(train_data), dtype=torch.bool)
        mask[indices] = False
        # Use the mask to select the remaining data
        train_data = train_data[mask]
        train_labels = train_labels[mask]

    # Move data to device
    train_data = train_data.to(device)
    train_labels = train_labels.to(device)
    test_data = test_data.to(device)
    test_labels = test_labels.to(device)
    graph_data = graph_data.to(device)
    graph_labels = graph_labels.to(device)

    # Wrap tensors in TensorDataset
    train_dataset = TensorDataset(train_data, train_labels)
    test_dataset = TensorDataset(test_data, test_labels)

    # Create data loaders for our datasets; shuffle for training, not for validation
    training_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    testing_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    return (
        train_dataset,
        test_dataset,
        training_loader,
        testing_loader,
        graph_data,
        graph_labels,
    )


def load_cifar10(
    graph_data_proportion=1 / 6, batch_size=50, stratification=False, device="cpu"
):
    """Import CIFAR-10 dataset

    Args:
        graph_data_proportion: the proportion of the training dataset to be used for generating functional connectomes. Defaults to 1/6.
        batch_size: the number of samples in each batch. Defaults to 50.
        stratification: if True, the loaded data will be stratified. Defaults to False.
        device: the device to move the data to. Defaults to "cpu".

    Returns:
        train_dataset: the training dataset.
        test_dataset: the testing dataset.
        training_loader: the training data loader.
        testing_loader: the testing data loader.
        graph_data: the data used for generating functional connectomes.
        graph_labels: the labels of the data used for generating functional connectomes.
    """

    if not os.path.exists("./data/CIFAR10/raw"):
        download = True
    else:
        download = False

    # Load CIFAR-10 dataset
    trainset = datasets.CIFAR10(
        root="./data", train=True, download=download, transform=transforms.ToTensor()
    )
    testset = datasets.CIFAR10(
        root="./data", train=False, download=download, transform=transforms.ToTensor()
    )

    # Convert dataset to PyTorch tensors
    train_data = torch.stack([data[0] for data in trainset])
    if stratification:
        train_labels = torch.tensor(trainset.targets)
    else:
        train_labels = torch.tensor(trainset.targets)
    test_data = torch.stack([data[0] for data in testset])
    test_labels = torch.tensor(testset.targets)

    if stratification:
        # Apply stratified sampling to the training data
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=graph_data_proportion, random_state=42
        )
        train_idx, graph_idx = next(sss.split(train_data.numpy(), train_labels.numpy()))
        graph_data = train_data[graph_idx]
        graph_labels = train_labels[graph_idx]
        train_data = train_data[train_idx]
        train_labels = train_labels[train_idx]
    else:
        # Sample data for generating graphs:
        indices = torch.randperm(len(train_data))[
            : int(graph_data_proportion * len(train_data))
        ]
        graph_data = train_data[indices]
        graph_labels = train_labels[indices]
        # Create a mask for all data points
        mask = torch.ones(len(train_data), dtype=torch.bool)
        mask[indices] = False
        train_data = train_data[mask]
        train_labels = train_labels[mask]

    # Move data to device
    train_data = train_data.to(device)
    train_labels = train_labels.to(device)
    test_data = test_data.to(device)
    test_labels = test_labels.to(device)
    graph_data = graph_data.to(device)
    graph_labels = graph_labels.to(device)

    # Wrap tensors in TensorDataset
    train_dataset = TensorDataset(train_data, train_labels)
    test_dataset = TensorDataset(test_data, test_labels)

    # Create data loaders for our datasets; shuffle for training, not for validation
    training_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testing_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return (
        train_dataset,
        test_dataset,
        training_loader,
        testing_loader,
        graph_data,
        graph_labels,
    )


def load_dataset(
    dataset_name,
    graph_data_proportion=1 / 6,
    batch_size=50,
    stratification=False,
    device="cpu",
):
    """Load dataset based on the dataset name

    Args:
        dataset_name: the name of the dataset to be loaded. (mnist, fashion-mnist, cifar10)
        graph_data_proportion: the proportion of the training dataset to be used for generating functional connectomes. Defaults to 1/6.
        batch_size: the number of samples in each batch. Defaults to 50.
        stratification: if True, the loaded data will be stratified. Defaults to False.
        device: the device to move the data to. Defaults to "cpu".

    Raises:
        ValueError: if the dataset name is not found.

    Returns:
        train_dataset: the training dataset.
        test_dataset: the testing dataset.
        training_loader: the training data loader.
        testing_loader: the testing data loader.
        graph_data: the data used for generating functional connectomes.
        graph_labels: the labels of the data used for generating functional connectomes.
    """
    if dataset_name == "mnist":
        return load_mnist(
            graph_data_proportion=graph_data_proportion,
            batch_size=batch_size,
            stratification=stratification,
            device=device,
        )
    elif dataset_name == "fashion-mnist":
        return load_fashion_mnist(
            graph_data_proportion=graph_data_proportion,
            batch_size=batch_size,
            stratification=stratification,
            device=device,
        )
    elif dataset_name == "cifar10":
        return load_cifar10(
            graph_data_proportion=graph_data_proportion,
            batch_size=batch_size,
            stratification=stratification,
            device=device,
        )
    else:
        raise ValueError("Dataset not found")


def combine_data(train_dataset, test_dataset, batch_size=50):
    """Combine training and testing datasets into a single dataset and data loader

    Args:
        train_dataset: the training dataset.
        test_dataset: the testing dataset.
        batch_size: the number of samples in each batch. Defaults to 50.

    Returns:
        full_dataset: the combined dataset.
        full_loader: the data loader for the combined dataset.
    """
    full_dataset = ConcatDataset([train_dataset, test_dataset])
    full_loader = DataLoader(full_dataset, batch_size=batch_size)
    return full_dataset, full_loader


def filter_dataset_by_class(dataset, class_label, batch_size=50):
    """Filter a dataset by class label

    Args:
        dataset: the dataset to be filtered.
        class_label: the class label to filter the dataset.
        batch_size: the number of samples in each batch. Defaults to 50.

    Returns:
        filtered_dataset: the filtered dataset.
        filtered_loader: the data loader for the filtered dataset
    """
    indices = []
    cumulative_size = 0

    for sub_dataset in dataset.datasets:
        sub_labels = torch.tensor([label.item() for _, label in sub_dataset])
        sub_indices = (sub_labels == class_label).nonzero(as_tuple=True)[0]
        indices.append(sub_indices + cumulative_size)
        cumulative_size += len(sub_dataset)

    indices = torch.cat(indices)
    print("Number of data points in class", class_label, ":", len(indices))

    filtered_dataset = Subset(dataset, indices)
    filtered_loader = DataLoader(filtered_dataset, batch_size=batch_size, shuffle=True)

    return filtered_dataset, filtered_loader


def filter_data_by_class(data, labels, class_label):
    """Filter data by class label

    Args:
        data: the data to be filtered.
        labels: the labels of the data.
        class_label: the class label to filter the data.

    Returns:
        filtered_data: the filtered data
    """
    indices = (labels == class_label).nonzero(as_tuple=True)[0]
    print("Number of data points in class", class_label, ":", len(indices))
    filtered_data = data[indices]
    return filtered_data
