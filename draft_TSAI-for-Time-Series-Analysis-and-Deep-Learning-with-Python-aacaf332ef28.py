# Description: Short example for TSAI for Time Series Analysis and Deep Learning with Python.


import logging

from tsai.data import TSDataLoaders, get_synthetic_data, get_UCR_data
from tsai.data.transforms import RandomCrop, RandomNoise
from tsai.interpretation import CAM
from tsai.learner import TSClassifier, TSRegressor
from tsai.models import TST, InceptionTime


def main():
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


    # Download and load the ArrowHead dataset
    X, y, splits = get_UCR_data("ArrowHead", split_data=True)
    logger.info(X.shape, y.shape)  # (Samples, Timesteps, Features)

    (211, 251, 1)(
        211,
    )


    # Create DataLoaders for training and validation
    dls = TSDataLoaders.from_dsets(X[splits[0]], y[splits[0]], X[splits[1]], y[splits[1]])


    # Initialize the model and learner
    model = InceptionTime(dls.vars, dls.c)
    learn = TSClassifier(dls, model)

    # Train the model
    learn.fit_one_cycle(10, 1e-3)

    # Plot training metrics
    learn.plot_metrics()

    # Predict on validation set
    val_preds, val_targets = learn.get_preds()


    # Generate synthetic regression data
    X, y, splits = get_synthetic_data(n_samples=100, n_timesteps=50, task="regression")


    # Create DataLoaders
    dls = TSDataLoaders.from_dsets(X[splits[0]], y[splits[0]], X[splits[1]], y[splits[1]])

    # Train a TST model (Transformer for time series)
    model = TST(dls.vars, dls.c, seq_len=dls.len)
    learn = TSRegressor(dls, model)

    learn.fit_one_cycle(10, 1e-3)


    # Generate synthetic forecasting data
    X, y, splits = get_synthetic_data(n_samples=200, n_timesteps=24, task="forecasting")

    # Create DataLoaders
    dls = TSDataLoaders.from_dsets(X[splits[0]], y[splits[0]], X[splits[1]], y[splits[1]])

    # Train an InceptionTime model
    model = InceptionTime(dls.vars, dls.c)
    learn = TSRegressor(dls, model)
    learn.fit_one_cycle(10, 1e-3)


    # Apply random cropping and noise
    dls.add_tfms([RandomCrop(0.9), RandomNoise(0.05)])
    learn.fit_one_cycle(10, 1e-3)


    # Visualize feature importance using Class Activation Mapping
    cam = CAM(learn)
    cam.plot(X[splits[1]][0])


if __name__ == "__main__":
    main()
