"""Generated from Jupyter notebook: TSAI for Time Series Analysis and Deep Learning with Python

Magics and shell lines are commented out. Run with a normal Python interpreter."""


# --- code cell ---

# ! pip install tsai==0.3.7  # Jupyter-only


# --- code cell ---

from tsai.data import get_UCR_data
# Download and load the ArrowHead dataset
X, y, splits = get_UCR_data("ArrowHead", split_data=True)
print(X.shape, y.shape)  # (Samples, Timesteps, Features)


# --- code cell ---

from tsai.data import TSDataLoaders
# Create DataLoaders for training and validation
dls = TSDataLoaders.from_dsets(X[splits[0]], y[splits[0]], X[splits[1]], y[splits[1]])


# --- code cell ---

from tsai.models import InceptionTime
from tsai.learner import TSClassifier
# Initialize the model and learner
model = InceptionTime(dls.vars, dls.c)
learn = TSClassifier(dls, model)
# Train the model
learn.fit_one_cycle(10, 1e-3)


# --- code cell ---

# Plot training metrics
learn.plot_metrics()
# Predict on validation set
val_preds, val_targets = learn.get_preds()


# --- code cell ---

import numpy as np
from tsai.data import get_synthetic_data
# Generate synthetic regression data
X, y, splits = get_synthetic_data(n_samples=100, n_timesteps=50, task="regression")
Train a Regression Model
from tsai.learner import TSRegressor
from tsai.models import TST
# Create DataLoaders
dls = TSDataLoaders.from_dsets(X[splits[0]], y[splits[0]], X[splits[1]], y[splits[1]])
# Train a TST model (Transformer for time series)
model = TST(dls.vars, dls.c, seq_len=dls.len)
learn = TSRegressor(dls, model)
learn.fit_one_cycle(10, 1e-3)


# --- code cell ---

from tsai.data import get_synthetic_data
# Generate synthetic forecasting data
X, y, splits = get_synthetic_data(n_samples=200, n_timesteps=24, task="forecasting")
# Create DataLoaders
dls = TSDataLoaders.from_dsets(X[splits[0]], y[splits[0]], X[splits[1]], y[splits[1]])
# Train an InceptionTime model
model = InceptionTime(dls.vars, dls.c)
learn = TSRegressor(dls, model)
learn.fit_one_cycle(10, 1e-3)


# --- code cell ---

from tsai.data.transforms import RandomCrop, RandomNoise
# Apply random cropping and noise
dls.add_tfms([RandomCrop(0.9), RandomNoise(0.05)])
learn.fit_one_cycle(10, 1e-3)


# --- code cell ---

from tsai.interpretation import CAM
# Visualize feature importance using Class Activation Mapping
cam = CAM(learn)
cam.plot(X[splits[1]][0])
