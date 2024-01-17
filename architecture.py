import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import r2_score
import argparse

# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument("--epoch", nargs="?", default=100, type=int,
                    help="Epoch to train (Default: 100)")
parser.add_argument("--dataset", nargs="?", required=True,
                    help="Dataset for training")
parser.add_argument("--k", nargs="?", type=int, default=3,
                    help="K to use for k fold (Default: 3)")
args = parser.parse_args()

# set Hyperparameters
n_epochs = args.epoch
k_folds = args.k
dataset = np.loadtxt(args.dataset, delimiter=",")

models = []

models.append(nn.Sequential(
    nn.Linear(65, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 1)))

models.append(nn.Sequential(
    nn.Linear(65, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 1)))

models.append(nn.Sequential(
    nn.Linear(65, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 1)))

models.append(nn.Sequential(
    nn.Linear(65, 32),
    nn.ReLU(),
    nn.Linear(32, 24),
    nn.ReLU(),
    nn.Linear(24, 8),
    nn.ReLU(),
    nn.Linear(8, 1)))

models.append(nn.Sequential(
    nn.Linear(65, 16),
    nn.ReLU(),
    nn.Linear(16, 1)))

X = dataset[:, :65]
y = dataset[:, 65:]
# Convert to tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)


def k_fold(dataset, k):
    """
    Convert the dataset into k folds for cross-validation.

    :param1 dataset: The dataset
    :param2 k: Number of folds

    :return: array of dataset split into k-fold
    """
    fold_size = len(dataset) // k
    indices = np.arange(len(dataset))
    folds = []
    for i in range(k):
        test_indices = indices[i * fold_size: (i + 1) * fold_size]
        train_indices = np.concatenate(
            [indices[:i * fold_size], indices[(i + 1) * fold_size:]])
        folds.append((train_indices, test_indices))
    return folds


def mse(value, expected):
    """
    Calculates the mean square error of the value and the expected values

    :param1 value: the output value of the model
    :param2 expected: the expected value

    :return: the MSE
    """
    return (np.square(value - expected)).mean(axis=0)[0]


def mae(value, expected):
    """
    Calculates the mean absolute error of the value and the expected values

    :param1 value: the output value of the model
    :param2 expected: the expected value

    :return: the MAE
    """
    return np.abs(value - expected).mean(axis=0)[0]


# K-fold cross-validation
kl = k_fold(X, 3)


for learning_rate in [0.001, 0.001, 0.1]:
    for batch_size in [1000, 100, 10]:
        for model in models:
            mse_list, mae_list, r2_list = [], [], []
            loss_fn = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            for i in range(len(kl)):
                train_index = kl[i][0]
                val_index = kl[i][1]
                print(f"Fold {i + 1}/{k_folds}")
                X_train, X_val = X[train_index], X[val_index]
                y_train, y_val = y[train_index], y[val_index]

                # Convert to tensors
                X_train = X_train.detach().clone().float()
                y_train = y_train.detach().clone().float()
                X_val = X_val.detach().clone().float()
                y_val = y_val.detach().clone().float()

                loss_fn = nn.MSELoss()
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)

                dataset_train = TensorDataset(X_train, y_train)
                dataloader_train = DataLoader(
                    dataset_train, batch_size=batch_size, shuffle=True)

                # Training loop
                for epoch in range(n_epochs):
                    model.train()
                    for Xbatch, ybatch in dataloader_train:
                        y_pred = model(Xbatch)
                        loss = loss_fn(y_pred, ybatch)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # Validation
                model.eval()
                with torch.no_grad():
                    y_val_pred = model(X_val)

                    # Regression metrics
                    mse_fold = mse(np.array(y_val), np.array(y_val_pred))
                    mae_fold = mae(np.array(y_val), np.array(y_val_pred))
                    r2 = r2_score(y_val, y_val_pred)

                    mse_list.append(mse_fold)
                    mae_list.append(mae_fold)
                    r2_list.append(r2)
                    print(
                        f"Fold {i + 1} Metrics - MSE: {mse_fold:.4f}, MAE: {mae_fold:.4f}, R2: {r2:.4f}")

            # Print average metrics across folds
            print(f"\nAverage Metrics Across {k_folds} Folds:")
            print(f"Average Mean Squared Error (MSE): {np.mean(mse_list):.4f}")
            print(
                f"Average Mean Absolute Error (MAE): {np.mean(mae_list):.4f}")
            print(
                f"{learning_rate} {batch_size} Average R-squared (R2): {np.mean(r2_list):.4f}")
