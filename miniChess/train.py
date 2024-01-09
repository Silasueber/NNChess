import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

dataset = np.loadtxt('data/minichess.csv', delimiter=',')
X = dataset[:, :65]
y = dataset[:, 65:]
# Convert to tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# define model
model = nn.Sequential(
    nn.Linear(65, 512),
    nn.ReLU(),
    nn.Linear(512, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 1))

# load model:
model = torch.load("models/minichess.pt")

loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

batch_size = 10
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

n_epochs = 200
for epoch in range(n_epochs):
    avg_loss = 0
    amount = 0
    for Xbatch, ybatch in dataloader:
        y_pred = model(Xbatch)
        loss = loss_fn(y_pred, ybatch)
        avg_loss += loss.item()
        amount += 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Print loss after each epoch
    print(f'Finished epoch {epoch}, latest loss {str(avg_loss/amount)}')

torch.save(model, "models/minichess.pt")
model = torch.load("models/minichess.pt")
model.eval()
