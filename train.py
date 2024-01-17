import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import argparse
import matplotlib.pyplot as plt
import numpy as np


# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=100,
                    help="Number of epochs to train (Default: 100)")
parser.add_argument("--batch", type=int, default=10,
                    help="Batch size (Default: 10)")
parser.add_argument("--model", nargs="?",
                    help="Model name on which to train")
parser.add_argument("--name", default="models/minichess/minichess.pt",
                    help="Name to save model (Default: models/minichess/minichess.pt)")
parser.add_argument("--dataset", required=True,
                    help="Dataset for training")
parser.add_argument("--lr", type=float, default=0.001,
                    help="Learning rate for the model (Default: 0.001)")
parser.add_argument("--layers", default="256,128,64",
                    help="The structure of the model (Default: 256,128,64)")
args = parser.parse_args()

# Load dataset
dataset = np.loadtxt(args.dataset, delimiter=',')

# Set hyperparameters
n_epochs = args.epoch
model_structure = args.layers
lr = args.lr
batch_size = args.batch
name = args.name

# Load or initialize model
if args.model:
    model = torch.load(f"{args.model}")
else:
    # Define model architecture
    model = nn.Sequential()
    last_layer = 65
    for layer in model_structure.split(","):
        model.append(nn.Linear(last_layer, int(layer)))
        last_layer = int(layer)
        model.append(nn.ReLU())
    model.append(nn.Linear(last_layer, 1))
    print(model)

# Split dataset into train and eval
X_train, X_eval, y_train, y_eval = train_test_split(
    dataset[:, :65], dataset[:, 65:], test_size=0.2, random_state=42)

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

X_eval_tensor = torch.tensor(X_eval, dtype=torch.float32)
y_eval_tensor = torch.tensor(y_eval, dtype=torch.float32)

# Loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)


# Create DataLoaders for train and eval
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)

eval_dataset = TensorDataset(X_eval_tensor, y_eval_tensor)
eval_dataloader = DataLoader(
    eval_dataset, batch_size=batch_size, shuffle=False)

# Matplot values
train_loss_values = []
eval_loss_values = []

# Checkpoint values
checkpoint_epoch = 0

# Training loop
for epoch in range(n_epochs):
    # Train
    model.train()
    avg_train_loss = 0
    for Xbatch, ybatch in train_dataloader:
        y_pred = model(Xbatch)
        loss = loss_fn(y_pred, ybatch)
        avg_train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_train_loss /= len(train_dataloader)
    train_loss_values.append(avg_train_loss)

    # Evaluate
    model.eval()
    avg_eval_loss = 0
    with torch.no_grad():
        for Xbatch, ybatch in eval_dataloader:
            y_pred = model(Xbatch)
            loss = loss_fn(y_pred, ybatch)
            avg_eval_loss += loss.item()

    avg_eval_loss /= len(eval_dataloader)
    if len(eval_loss_values) > 0 and avg_eval_loss < min(eval_loss_values):
        # Save checkpoint of the model
        torch.save(model, name.split(".pt")[0]+"_checkpoint.pt")
        checkpoint_epoch = epoch
    eval_loss_values.append(avg_eval_loss)

    # Print loss after each epoch
    print(
        f'Epoch {epoch + 1}/{n_epochs}, Train Loss: {avg_train_loss:.4f}, Eval Loss: {avg_eval_loss:.4f}')

# Save the trained model
torch.save(model, name)

print(f'Final eval loss {eval_loss_values[-1]}')
print(
    f'Checkpoint made at epoch {checkpoint_epoch} with a loss of {min(eval_loss_values)} modelname: {name.split(".pt")[0]+"_checkpoint.pt"}')

# Plot loss curves
plt.plot(train_loss_values, label='Train Loss')
plt.plot(eval_loss_values, label='Eval Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
