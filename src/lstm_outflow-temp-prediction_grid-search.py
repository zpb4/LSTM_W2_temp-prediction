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
from util import water_day
import util
import calendar
import matplotlib.pyplot as plt
import pickle

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#-----------------------------------------
# 1. Define the inputs
#-----------------------------------------

sd = '10-01-1997'  #W2 model spinup affects data in 95-97
ed = '12-31-2015'

#read in predictors
sha_w2_store = pd.read_csv('./data/W2_hist_95-16_store.csv', index_col=0, parse_dates=True)[sd:ed] #reservoir storage ts from W2
sha_w2_met = pd.read_csv('./data/W2_hist_95-16_met.csv', index_col=0, parse_dates=True)[sd:ed] #meteorological input ts to W2 [Tair, Tdew, Wind Spd, Wind Dir, Cloud Cover]
sha_w2_rel = pd.read_csv('./data/W2_hist_95-16_gate-releases.csv', index_col=0, parse_dates=True)[sd:ed] #reservoir releases from gate 1:5 by column 
# (G1: side gate [lowest], G2: low TCD gate (2d lowest), G3: middle TCD gate, G4: upper TCD gate, G5: river outflow gate?)
sha_w2_inf = pd.read_csv('./data/W2_hist_95-16_inflow.csv', index_col=0, parse_dates=True)[sd:ed] #reservoir inflow ts to W2
sha_w2_inftemp = pd.read_csv('./data/DLT_95-16_temp_C.csv', index_col=0, parse_dates=True)[sd:ed] #reservoir inflow temp (DLT) ts to W2
dtg = sha_w2_store.index

np.savez('./data/date-time_hist.npz',arr=dtg)

#create day-of-water year as add'l predictor
dowy = np.array([water_day(d,calendar.isleap(d.year)) for d in dtg])
dwy = np.zeros((len(dowy),1))
dwy[:,0] = dowy

store = np.array(sha_w2_store)
met = np.array(sha_w2_met)
met1 = np.zeros(np.shape(dwy));met1[:,0] = np.array(sha_w2_met)[:,0] #extract only Tair as most highly correlated variable
#met1 = met[:,0:2]
rel = np.array(sha_w2_rel)
rel1 = np.zeros(np.shape(dwy));rel1[:,0] = np.array(sha_w2_rel)[:,0] #extract only release at lowest gate for now
inf = np.array(sha_w2_inf)
inftemp = np.array(sha_w2_inftemp)

preds = np.concat((store,rel1,met1,inf,inftemp,dwy),axis=1)
predictors = util.minmax_scale(preds)[0] #minmax scaling of predictors

#predictand y
sha_w2_temp = pd.read_csv('./data/W2_hist_95-16_outflow-temp.csv', index_col=0, parse_dates=True)[sd:ed] #W2 outflow temperature is predictand
pdands = np.array(sha_w2_temp)
predictands,pred_scale = util.minmax_scale(pdands) #minmax scaling of predictand

#save scaling values
y_scale = {'y minmax': pred_scale}
pickle.dump(y_scale,open('./data/y-scale-minmax.pkl','wb'))


# -----------------------------------
#  2. Train LSTM with grid search
# -----------------------------------
#define model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        out_last = out[:, -1, :]  # Take output at last time step
        out_fc = self.fc(out_last)
        return out_fc
    
#grid search across different hyperparameters
hidden = ([16,32,64,128])
bc = ([16,32,64,128])

hd = np.repeat(hidden,4)
bcs = np.concat((bc,bc,bc,bc))

val_loss_vec = np.zeros(len(hd))

for k in range(len(hd)):
    # Create sequences from full dataset
    seq_length = 30  # You can change this as needed
    X_sequences, y_targets = util.create_sequences(predictors, predictands, seq_length)
    
    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X_sequences, dtype=torch.float32)
    y_tensor = torch.tensor(y_targets, dtype=torch.float32)

    # Model settings
    input_dim = np.shape(predictors)[1]
    hidden_dim = int(hd[k])
    num_layers = 2
    output_dim = 1  # predict sin(3t)

    model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    epochs = 50     #Hyperparameter that you can adjust
    batch_size = int(bcs[k])  #Hyperparameter that you can adjust

    # Dataset split
    train_size = int(0.8 * len(X_tensor))
    X_train, X_val = X_tensor[:train_size], X_tensor[train_size:]
    y_train, y_val = y_tensor[:train_size], y_tensor[train_size:]

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
        
    val_loss_vec[k] = val_loss

#save results to use for implementation
min_idx = np.where(val_loss_vec == min(val_loss_vec))[0][0]
val_results = {'hidden layer grid': hd, 'batch size grid': bcs, 'min val loss': min(val_loss_vec),'opt hidden layer': int(hd[min_idx]),'opt batch size': int(bcs[min_idx])}
pickle.dump(val_results,open('./out/val-grid-results.pkl','wb'))


############################################END##########################################################