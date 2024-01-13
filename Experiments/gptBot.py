import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from stockfish import Stockfish

# Load the dataset, split into input (X) and output (y) variables
dataset = np.loadtxt('positions_new_2.csv', delimiter=',')
X = dataset[:, :66]
y = dataset[:, 66]

# Set a random seed for reproducibility
np.random.seed(42)

# Shuffle the indices to randomize the data
indices = np.arange(len(X))
np.random.shuffle(indices)

# Define the split ratio (e.g., 80% for training, 20% for testing)
split_ratio = 0.8
split_idx = int(split_ratio * len(X))

# Split the dataset
X_train, X_test = X[indices[:split_idx]], X[indices[split_idx:]]
y_train, y_test = y[indices[:split_idx]], y[indices[split_idx:]]

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

# Define the model
model = nn.Sequential(
    nn.Linear(66, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
    nn.Sigmoid())

loss_fn = nn.BCELoss()  # binary cross entropy
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer

# Train the model
n_epochs = 10000
batch_size = 100
for epoch in range(n_epochs):
    for i in range(0, len(X_train), batch_size):
        Xbatch = X_train[i:i+batch_size]
        y_pred = model(Xbatch)
        ybatch = y_train[i:i+batch_size]
        loss = loss_fn(y_pred, ybatch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Finished epoch {epoch}, latest loss {loss}')

# Save the trained model
torch.save(model, "model.pt")

# Load the trained model
model = torch.load("model.pt")
model.eval()

# Evaluate the model on the test set
with torch.no_grad():
    y_pred_test = model(X_test)
    y_pred_test_binary = (y_pred_test >= 0.5).float()
    accuracy = (y_pred_test_binary == y_test).float().mean()
    print(f'Test Accuracy: {accuracy.item()}')
