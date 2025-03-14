# LSTM_W2_temp-prediction
Code to implement an LSTM prediction model for output temperature simulations from CE-QUAL-W2 for Shasta Reservoir. Currently, this is a basic example where the CE-QUAL-W2 model has been run against the 1995-2016 Null et al. (2024) forcing data with historical inflow and outflow timeseries. Data provided for the model comes from this implementation and is all available in the 'data' sub-repository

### Dependencies
- PyTorch
- Numpy
### Workflow
1. Run 'lstm_outflow-temp-prediction_grid-search.py' to do a simple grid search through 16 combinations of hyperparameters (10-15 min runtime)
2. Run 'lstm_outflow-temp-prediction_val-results.py' to fit model using best hyperparameters from step 1 and plot results (~2-3 min runtime)
3. Run 'lstm_outflow-temp-prediction_model-out.py' to fit model to all available data and save the PyTorch LSTM model to prepare for implementation (~2-3 min runtime)
4. 'lstm_outflow-temp-prediction_model-out_example.py' shows the implementation of the fitted LSTM model to the historical data. This code is the basis of how the model could be implemented in an optimization enviroment once the model has been fitted
