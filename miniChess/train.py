import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import argparse
import sys

# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=100,
                    help="Number of epochs to train (Default: 100)")
parser.add_argument("--batch", type=int, default=10,
                    help="Batch size (Default: 10)")
parser.add_argument("--model", nargs="?",
                    help="Model name on which to train")
parser.add_argument("--name", default="minichess_two.pt",
                    help="Name to save model (Default: minichess.pt)")
parser.add_argument("--dataset", required=True,
                    help="Dataset for training")
parser.add_argument("--lr", type=float, default=0.001,
                    help="Learning rate for the model (Default: 0.001)")
args = parser.parse_args()

# Load dataset
dataset = np.loadtxt(f"{args.dataset}", delimiter=',')

# Set hyperparameters
n_epochs = args.epoch
lr = args.lr
batch_size = args.batch
name = f"{args.name}"

# Load or initialize model
if args.model:
    model = torch.load(f"{args.model}")
else:
    # Define model architecture
    model = nn.Sequential(
        nn.Linear(65, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 1))

# Convert to tensors
X = torch.tensor(dataset[:, :65], dtype=torch.float32)
y = torch.tensor(dataset[:, 65:], dtype=torch.float32)

# Loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Create DataLoader
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(n_epochs):
    avg_loss = 0
    for Xbatch, ybatch in dataloader:
        y_pred = model(Xbatch)
        loss = loss_fn(y_pred, ybatch)
        avg_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print loss after each epoch
    avg_loss /= len(dataloader)
    print(f'Finished epoch {epoch}, latest loss: {avg_loss:.4f}')

# Save the trained model
torch.save(model, name)
