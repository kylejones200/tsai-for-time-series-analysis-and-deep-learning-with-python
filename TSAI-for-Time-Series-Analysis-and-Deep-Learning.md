# TSAI for Time Series Analysis and Deep Learning with Python

**⚠️ Requires**: `tsai` library (`pip install tsai`)

## Introduction

`tsai` is a deep learning library specifically designed for time series tasks. Built on top of PyTorch and fastai, it provides state-of-the-art models and tools for time series classification, regression, and forecasting. With its user-friendly API, `tsai` makes it easy to train, evaluate, and deploy deep learning models for time series.

## Why Use `tsai`?

`tsai` stands out because it offers:

1. **State-of-the-Art Models**: Access to advanced architectures like InceptionTime, ResNet, TST, and XCM
2. **Ease of Use**: Simplified APIs for loading data, training models, and evaluating results
3. **Performance**: Built on PyTorch, ensuring efficiency and scalability
4. **Flexibility**: Supports both univariate and multivariate time series data

## Installation

```bash
pip install tsai
```

## Key Features of `tsai`

1. **Deep Learning Models**: Prebuilt architectures for classification, regression, and forecasting
2. **Data Augmentation**: Built-in support for augmenting time series data
3. **Interpretability**: Tools to visualize model predictions and feature importance

## Loading and Preparing Data

`tsai` provides utilities to load and preprocess time series data.

### Example Dataset: ArrowHead

The **ArrowHead** dataset is commonly used for time series classification.

```python
from tsai.data import get_UCR_data

# Download and load the ArrowHead dataset
X, y, splits = get_UCR_data("ArrowHead", split_data=True)
print(X.shape, y.shape)  # (Samples, Timesteps, Features)
```

**Output**:
```
(211, 251, 1) (211,)
```

Here:
- `X` contains the time series data (211 samples, each with 251 timesteps and 1 feature)
- `y` contains the labels for classification
- `splits` divides the data into training and testing sets

## Time Series Classification

### Step 1: Create a DataLoader

```python
from tsai.data import TSDataLoaders

# Create DataLoaders for training and validation
dls = TSDataLoaders.from_dsets(X[splits[0]], y[splits[0]], X[splits[1]], y[splits[1]])
```

### Step 2: Train a Model

```python
from tsai.models import InceptionTime
from tsai.learner import TSClassifier

# Initialize the model and learner
model = InceptionTime(dls.vars, dls.c)
learn = TSClassifier(dls, model)

# Train the model
learn.fit_one_cycle(10, 1e-3)
```

### Step 3: Evaluate the Model

```python
# Plot training metrics
learn.plot_metrics()

# Predict on validation set
val_preds, val_targets = learn.get_preds()
```

## Time Series Regression

`tsai` also supports regression tasks, where the goal is to predict a continuous value.

### Example: Synthetic Data

```python
import numpy as np
from tsai.data import get_synthetic_data

# Generate synthetic regression data
X, y, splits = get_synthetic_data(n_samples=100, n_timesteps=50, task="regression")
```

### Train a Regression Model

```python
from tsai.learner import TSRegressor
from tsai.models import TST

# Create DataLoaders
dls = TSDataLoaders.from_dsets(X[splits[0]], y[splits[0]], X[splits[1]], y[splits[1]])

# Train a TST model (Transformer for time series)
model = TST(dls.vars, dls.c, seq_len=dls.len)
learn = TSRegressor(dls, model)

learn.fit_one_cycle(10, 1e-3)
```

## Time Series Forecasting

While `tsai` primarily focuses on classification and regression, it can handle forecasting tasks using its flexible APIs.

### Example: Forecasting with Synthetic Data

```python
from tsai.data import get_synthetic_data

# Generate synthetic forecasting data
X, y, splits = get_synthetic_data(n_samples=200, n_timesteps=24, task="forecasting")

# Create DataLoaders
dls = TSDataLoaders.from_dsets(X[splits[0]], y[splits[0]], X[splits[1]], y[splits[1]])

# Train an InceptionTime model
model = InceptionTime(dls.vars, dls.c)
learn = TSRegressor(dls, model)
learn.fit_one_cycle(10, 1e-3)
```

## Data Augmentation

`tsai` supports data augmentation to improve model robustness.

```python
from tsai.data.transforms import RandomCrop, RandomNoise

# Apply random cropping and noise
dls.add_tfms([RandomCrop(0.9), RandomNoise(0.05)])
learn.fit_one_cycle(10, 1e-3)
```

## Model Interpretability

Use `tsai` to interpret model predictions and understand feature importance.

### Example: Grad-CAM for Time Series

```python
from tsai.interpretation import CAM

# Visualize feature importance using Class Activation Mapping
cam = CAM(learn)
cam.plot(X[splits[1]][0])
```

## Key Models in `tsai`

| Model | Use Case |
|-------|----------|
| **InceptionTime** | Time series classification and regression |
| **ResNet** | Robust for time series classification |
| **TST (Transformer)** | Large, multivariate time series |
| **XCM** | Multivariate time series with categorical and continuous inputs |

## Key Benefits of `tsai`

1. **Ease of Use**: High-level APIs simplify complex tasks
2. **Advanced Models**: Access to cutting-edge architectures
3. **Flexibility**: Handles univariate, multivariate, and hierarchical time series data
4. **Interpretability**: Tools for visualizing and explaining model predictions

## Key Takeaways

`tsai` is a powerful library that democratizes deep learning for time series. Whether you're working on classification, regression, or forecasting, `tsai` provides a streamlined approach to building and deploying state-of-the-art models. Its rich set of features, combined with the power of PyTorch and fastai, makes it a must-have tool for time series practitioners.

Give `tsai` a try in your next time series project and experience the power of deep learning at your fingertips!

## Resources

- **tsai Documentation**: https://timeseriesai.github.io/tsai/
- **tsai GitHub**: https://github.com/timeseriesAI/tsai
- **fastai**: https://docs.fast.ai/
- **PyTorch**: https://pytorch.org/

---

**Note**: Requires understanding of deep learning concepts. Start with simpler models if new to neural networks.
