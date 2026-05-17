from tsai.data import TSDataLoaders
from tsai.data import get_UCR_data
from tsai.data import get_synthetic_data
from tsai.data.transforms import RandomCrop, RandomNoise
from tsai.interpretation import CAM
from tsai.learner import TSClassifier
from tsai.models import InceptionTime


def main() -> None:
    X, y, splits = get_UCR_data('ArrowHead', split_data=True)

    print(X.shape, y.shape)

    dls = TSDataLoaders.from_dsets(X[splits[0]], y[splits[0]], X[splits[1]], y[splits[1]])

    model = InceptionTime(dls.vars, dls.c)

    learn = TSClassifier(dls, model)

    learn.fit_one_cycle(10, 0.001)

    learn.plot_metrics()

    val_preds, val_targets = learn.get_preds()

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

    X, y, splits = get_synthetic_data(n_samples=200, n_timesteps=24, task='forecasting')

    dls = TSDataLoaders.from_dsets(X[splits[0]], y[splits[0]], X[splits[1]], y[splits[1]])

    model = InceptionTime(dls.vars, dls.c)

    learn = TSRegressor(dls, model)

    learn.fit_one_cycle(10, 0.001)

    dls.add_tfms([RandomCrop(0.9), RandomNoise(0.05)])

    learn.fit_one_cycle(10, 0.001)

    cam = CAM(learn)

    cam.plot(X[splits[1]][0])

if __name__ == "__main__":
    main()
