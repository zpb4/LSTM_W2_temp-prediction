# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 13:18:54 2025

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

jday_ref = '1950-01-01'

sd = '1990-11-16'
ed = '2024-08-29'

dtg = np.load('./data/date-time_90-24.npz')['arr']

#load trained lstm model
model = torch.load('./out/lstm_w2-output-temp_90-24.pt',weights_only=False)

#grid search parameters and scaling values
val_res = pickle.load(open('./out/val-grid-results_90-24.pkl','rb'),encoding='latin1')
bsize = val_res['opt batch size']
hdsize = val_res['opt hidden layer']

#scaling values
yscale = pickle.load(open('./data/y-scale-minmax_90-24.pkl','rb'),encoding='latin1')
y_minmax = yscale['y minmax']

#load the saved predictors
predictors_unscale = np.load('./data/w2-predictors_90-24.npz')['arr']
#***NOTE: column 1 (index 0) is the historical storage timeseries
#***NOTE: column 2 (index 1) is the historical release timeseries
# To run with a new storage and release profile, simply update these two columns with new data before scaling
predictands_unscale = np.load('./data/w2-predictands_90-24.npz')['arr']

#----------------------------------------------------------------------------
# 2. Setup LSTM model
#---------------------------------------------------------------------------
#scale predictors
X = util.minmax_scale(predictors_unscale)[0]
y = util.minmax_scale(predictands_unscale)[0]

seq_length = 30  # You can change this as needed
X_sequences, y_targets = util.create_sequences(X, y, seq_length)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X_sequences, dtype=torch.float32)

#evaluate the model against the predictors
model.eval()
with torch.no_grad():
    predictions = model(X_tensor).numpy()

temp_out = util.invert_minmax(predictions, y_minmax)


#----------------------------------------------------------------------------
# 3. Plot to verify
#---------------------------------------------------------------------------
# Plot predictions vs true values
dates = dtg[seq_length:]

plt.figure(figsize=(8, 6))
plt.plot(dtg,predictands_unscale,color='black')
plt.plot(dates,temp_out,color='green',alpha=0.5)  # Perfect prediction line
plt.legend(['W2','LSTM'])
plt.xlabel("Val days")
plt.ylabel("Temp (C)")
plt.title("LSTM predictions vs CE-QUAL-W2 output")
plt.show()