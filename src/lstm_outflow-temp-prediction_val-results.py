# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 11:46:20 2025

@author: zpb4
"""
import sys
import os
sys.path.insert(0, os.path.abspath('./src'))
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import util
from util import water_day
import calendar
import matplotlib.pyplot as plt
import pickle

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#----------------------------------------------------------------------------
# 1. Read in data
#---------------------------------------------------------------------------

sd = '10-01-1997'
ed = '12-31-2015'

dtg = np.load('./data/date-time_hist.npz')['arr']

#grid search parameters and scaling values
val_res = pickle.load(open('./out/val-grid-results.pkl','rb'),encoding='latin1')
bsize = val_res['opt batch size']
hdsize = val_res['opt hidden layer']
#scaling
yscale = pickle.load(open('./data/y-scale-minmax.pkl','rb'),encoding='latin1')
y_minmax = yscale['y minmax']

#read in predictors and predictands
predictors_unscale = np.load('./data/w2-predictors.npz')['arr']
predictors = util.minmax_scale(predictors_unscale)[0]

predictands_unscale = np.load('./data/w2-predictands.npz')['arr']
predictands = util.minmax_scale(predictands_unscale)[0]

# -----------------------------------------
# 2. Chunk the sequences and targets 
# ----------------------------------------
# Create sequences from full dataset
seq_length = 30  # You can change this as needed
X_sequences, y_targets = util.create_sequences(predictors, predictands, seq_length)
#X_sequences, y_targets = create_sequences1(predictands, seq_length)

dates = dtg[seq_length:]

# -----------------------------------------
# 3. Create tensors for LSTM
# -----------------------------------------
# Convert to PyTorch tensors
X_tensor = torch.tensor(X_sequences, dtype=torch.float32)
y_tensor = torch.tensor(y_targets, dtype=torch.float32)

# -----------------------------------
#  4. Train LSTM 
# -----------------------------------

# Model settings
input_dim = np.shape(predictors)[1]
hidden_dim = hdsize
num_layers = 2
output_dim = 1  # predict sin(3t)

model = util.LSTMModel(input_dim, hidden_dim, num_layers, output_dim)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 50     #Hyperparameter that you can adjust
batch_size = bsize

# Dataset split
#**NOTE: you can modify this variable if you want to look at different splits
trnval_split = 'beg-end'  # 'beg-end' for trn first 80%/val last 20%, 'end-beg' for trn last 80%/val first 20%

if trnval_split == 'beg-end':
    train_size = int(0.8 * len(X_tensor))
    X_train, X_val = X_tensor[:train_size], X_tensor[train_size:]
    y_train, y_val = y_tensor[:train_size], y_tensor[train_size:]
    dates_val = dates[-len(y_val):]

if trnval_split == 'end-beg':
    train_size = int(0.2 * len(X_tensor))
    X_train, X_val = X_tensor[train_size:], X_tensor[:train_size]
    y_train, y_val = y_tensor[train_size:], y_tensor[:train_size]
    dates_val = dates[:len(y_val)]

print(f"Train size: {len(X_train)}, Validation size: {len(X_val)}")

train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()
    permutation = torch.randperm(X_train.size(0))

    epoch_loss = 0
    for i in range(0, X_train.size(0), batch_size):
        indices = permutation[i:i + batch_size]
        batch_x, batch_y = X_train[indices], y_train[indices]

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    # Validation loss
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val).item()

    train_losses.append(epoch_loss / len(X_train))
    val_losses.append(val_loss)

    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {epoch_loss/len(X_train):.4f}, Val Loss: {val_loss:.4f}")
        

# -----------------------------------
# Step 5: Plot Loss
# -----------------------------------

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.title("Training and Validation Loss over Epochs")
plt.show()


# -----------------------------------
# Step 6: Evaluation on Validation Set
# -----------------------------------

model.eval()
with torch.no_grad():
    predictions = model(X_val).numpy()
    true_vals = y_val.numpy()

unscale_preds = util.invert_minmax(predictions,y_minmax)
unscale_true = util.invert_minmax(true_vals,y_minmax)

# Plot predictions vs true values
plt.figure(figsize=(8, 6))
plt.plot(dates_val,unscale_true,color='black')
plt.plot(dates_val,unscale_preds,color='green',alpha=0.5)  # Perfect prediction line
plt.legend(['W2','LSTM'])
plt.xlabel("Val days")
plt.ylabel("Temp (C)")
plt.title("LSTM predictions vs CE-QUAL-W2 output")
plt.show()


############################################END##########################################################