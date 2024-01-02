import numpy as np
import torch
import torch.nn as nn
from os import listdir
from os.path import isfile, join


dataset = np.loadtxt('data/p3_2.csv', delimiter=',')
X = dataset[:, :65]
y = dataset[:, 65:]
# Convert to tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

loss_fn = nn.MSELoss()

models = [f for f in listdir("models") if isfile(join("models", f))]
best_model = None
best_value = 1
for m in models:
    try:
        # load model:
        model = torch.load("models/"+m)

        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        if loss.item() < best_value:
            best_model = m
            best_value = loss.item()
        print(m, loss.item())
    except Exception as e:
        print(m, "Error")
print("----------")
print(f'Best Model: {best_model} with a loss of {best_value}')
