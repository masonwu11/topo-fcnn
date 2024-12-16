import torch.nn as nn
import torch.nn.functional as F


# Define the neural network architecture for Neural Network used in the MNIST dataset
# Input layer: 28 * 28 = 784 neurons
# Fully connected layer 1: 128 neurons, Leaky ReLU activation
# Fully connected layer 2: 64 neurons, Leaky ReLU activation
# Output layer: 10 neurons, log softmax activation
class MNIST_NN(nn.Module):
    def __init__(
        self,
        regularization_type="vanilla",
        l2_lambda=0,
        dropout_rate=0.0,
        alpha=0.01,
    ):
        super(MNIST_NN, self).__init__()
        self.regularization_type = regularization_type
        self.alpha = alpha

        self.fc1 = nn.Linear(784, 128)
        self.lrelu1 = nn.LeakyReLU(negative_slope=self.alpha)
        self.fc2 = nn.Linear(128, 64)
        self.lrelu2 = nn.LeakyReLU(negative_slope=self.alpha)
        self.output = nn.Linear(64, 10)

        if self.regularization_type == "batch_norm":
            self.bn1 = nn.BatchNorm1d(128)
            self.bn2 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        if self.regularization_type == "batch_norm":
            x = self.bn1(self.lrelu1(self.fc1(x)))
        else:
            x = self.lrelu1(self.fc1(x))

        x = self.dropout(x)

        if self.regularization_type == "batch_norm":
            x = self.bn2(self.lrelu2(self.fc2(x)))
        else:
            x = self.lrelu2(self.fc2(x))

        x = self.output(x)
        return F.log_softmax(x, dim=1)


# Define the neural network architecture for Neural Network used in the Fashion MNIST dataset
# Input layer: 28 * 28 = 784 neurons
# Fully connected layer 1: 256 neurons, Leaky ReLU activation
# Fully connected layer 2: 128 neurons, Leaky ReLU activation
# Output layer: 10 neurons, log softmax activation
class Fashion_MNIST_NN(nn.Module):
    def __init__(
        self,
        regularization_type="vanilla",
        l2_lambda=0,
        dropout_rate=0.0,
        alpha=0.01,
    ):
        super(Fashion_MNIST_NN, self).__init__()
        self.regularization_type = regularization_type
        self.alpha = alpha

        self.fc1 = nn.Linear(784, 256)
        self.lrelu1 = nn.LeakyReLU(negative_slope=self.alpha)
        self.fc2 = nn.Linear(256, 128)
        self.lrelu2 = nn.LeakyReLU(negative_slope=self.alpha)
        self.output = nn.Linear(128, 10)

        if self.regularization_type == "batch_norm":
            self.bn1 = nn.BatchNorm1d(256)
            self.bn2 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        if self.regularization_type == "batch_norm":
            x = self.bn1(self.lrelu1(self.fc1(x)))
        else:
            x = self.lrelu1(self.fc1(x))

        x = self.dropout(x)

        if self.regularization_type == "batch_norm":
            x = self.bn2(self.lrelu2(self.fc2(x)))
        else:
            x = self.lrelu2(self.fc2(x))

        x = self.output(x)
        return F.log_softmax(x, dim=1)


# Define the neural network architecture for Neural Network used in the CIFAR-10 dataset
# Input layer: 32 * 32 * 3
# 3 VGG blocks
#   Convolutional layer 1: 32 filters, kernel size 3x3, padding 1, Leaky ReLU activation
#   Convolutional layer 2: 32 filters, kernel size 3x3, padding 1, Leaky ReLU activation
#   Max pooling layer 1: kernel size 2x2, stride 2
# Fully connected layer 1: 256 neurons, Leaky ReLU activation
# Fully connected layer 2: 128 neurons, Leaky ReLU activation
# Output layer: 10 neurons, log softmax activation
class VGGNet(nn.Module):
    def __init__(
        self, regularization_type="vanilla", l2_lambda=0, dropout_rate=0.0, alpha=0.01
    ):
        super(VGGNet, self).__init__()
        self.regularization_type = regularization_type

        # VGG blocks
        self.conv1_1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.lrelu1_1 = nn.LeakyReLU(negative_slope=alpha)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.lrelu1_2 = nn.LeakyReLU(negative_slope=alpha)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.lrelu2_1 = nn.LeakyReLU(negative_slope=alpha)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.lrelu2_2 = nn.LeakyReLU(negative_slope=alpha)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.lrelu3_1 = nn.LeakyReLU(negative_slope=alpha)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.lrelu3_2 = nn.LeakyReLU(negative_slope=alpha)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(dropout_rate)

        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.lrelu_fc1 = nn.LeakyReLU(negative_slope=alpha)
        self.dropout_fc1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.lrelu_fc2 = nn.LeakyReLU(negative_slope=alpha)
        self.dropout_fc2 = nn.Dropout(dropout_rate)
        self.output = nn.Linear(128, 10)

        # Batch norm layers
        if self.regularization_type == "batch_norm":
            self.bn1_1 = nn.BatchNorm2d(32)
            self.bn1_2 = nn.BatchNorm2d(32)
            self.bn2_1 = nn.BatchNorm2d(64)
            self.bn2_2 = nn.BatchNorm2d(64)
            self.bn3_1 = nn.BatchNorm2d(128)
            self.bn3_2 = nn.BatchNorm2d(128)

    def forward(self, x):
        if self.regularization_type == "batch_norm":
            x = self.bn1_1(self.lrelu1_1(self.conv1_1(x)))
            x = self.bn1_2(self.lrelu1_2(self.conv1_2(x)))
        else:
            x = self.lrelu1_1(self.conv1_1(x))
            x = self.lrelu1_2(self.conv1_2(x))

        x = self.pool1(x)
        x = self.dropout1(x)

        if self.regularization_type == "batch_norm":
            x = self.bn2_1(self.lrelu2_1(self.conv2_1(x)))
            x = self.bn2_2(self.lrelu2_2(self.conv2_2(x)))
        else:
            x = self.lrelu2_1(self.conv2_1(x))
            x = self.lrelu2_2(self.conv2_2(x))

        x = self.pool2(x)
        x = self.dropout2(x)

        if self.regularization_type == "batch_norm":
            x = self.bn3_1(self.lrelu3_1(self.conv3_1(x)))
            x = self.bn3_2(self.lrelu3_2(self.conv3_2(x)))
        else:
            x = self.lrelu3_1(self.conv3_1(x))
            x = self.lrelu3_2(self.conv3_2(x))

        x = self.pool3(x)
        x = self.dropout3(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.lrelu_fc1(x)
        if self.regularization_type == "batch_norm":
            x = self.bn_fc1(x)
        x = self.dropout_fc1(x)
        x = self.fc2(x)
        x = self.lrelu_fc2(x)
        if self.regularization_type == "batch_norm":
            x = self.bn_fc2(x)
        x = self.dropout_fc2(x)
        x = self.output(x)
        return F.log_softmax(x, dim=1)
