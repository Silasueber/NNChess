import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import argparse
import sys

# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument("--epoch", nargs="?",
                    help="Epoch to train (Default: 100)")
parser.add_argument("--batch", nargs="?",
                    help="Batch size (Default: 10)")
parser.add_argument("--model", nargs="?",
                    help="Model name on which to train")
parser.add_argument("--name", nargs="?",
                    help="Name to save model (Default: minichess.pt)")
parser.add_argument("--dataset", nargs="?",
                    help="Dataset for training")
parser.add_argument("--lr", nargs="?",
                    help="Learning rate for the model (Default: 0.001)")
args = parser.parse_args()

if args.dataset != None:
    dataset = np.loadtxt(args.dataset, delimiter=',')
else:
    print("Please provide a dataset (--dataset)")
    sys.exit()

if args.epoch != None:
    n_epochs = int(args.epoch)
else:
    n_epochs = 100

if args.lr != None:
    lr = float(args.lr)
else:
    lr = 0.001

if args.batch != None:
    batch_size = int(args.batch)
else:
    batch_size = 10

if args.name != None:
    name = "models/"+args.name
else:
    name = "models/minichess.pt"

if args.model != None:
    model = torch.load(args.model)
else:
    # define model
    model = nn.Sequential(
        nn.Linear(65, 512),
        nn.ReLU(),
        nn.Linear(512, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 1))


X = dataset[:, :65]
y = dataset[:, 65:]
# Convert to tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)


loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)


dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


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

torch.save(model, name)
model = torch.load(name)
model.eval()
