"""Generated from Jupyter notebook: TSAI for Time Series Analysis and Deep Learning with Python.

Magics and shell lines are commented out. Run with a normal Python interpreter."""

from tsai.data import TSDataLoaders, get_UCR_data, get_synthetic_data
from tsai.data.transforms import RandomCrop, RandomNoise
from tsai.interpretation import CAM
from tsai.learner import TSClassifier, TSRegressor
from tsai.models import InceptionTime, TST


def download_and_load_the_arrowhead_dataset() -> None:
    global X, y, splits
    X, y, splits = get_UCR_data("ArrowHead", split_data=True)
    print(X.shape, y.shape)


def create_dataloaders_for_training_and_validation() -> None:
    global dls
    dls = TSDataLoaders.from_dsets(
        X[splits[0]], y[splits[0]], X[splits[1]], y[splits[1]]
    )


def initialize_the_model_and_learner() -> None:
    global learn
    model = InceptionTime(dls.vars, dls.c)
    learn = TSClassifier(dls, model)
    learn.fit_one_cycle(10, 0.001)


def plot_training_metrics() -> None:
    learn.plot_metrics()
    learn.get_preds()


def generate_synthetic_regression_data() -> None:
    global X, y, splits, dls, learn
    X, y, splits = get_synthetic_data(
        n_samples=100, n_timesteps=50, task="regression"
    )
    dls = TSDataLoaders.from_dsets(
        X[splits[0]], y[splits[0]], X[splits[1]], y[splits[1]]
    )
    model = TST(dls.vars, dls.c, seq_len=dls.len)
    learn = TSRegressor(dls, model)
    learn.fit_one_cycle(10, 1e-3)


def generate_synthetic_forecasting_data() -> None:
    global X, y, splits, dls, learn
    X, y, splits = get_synthetic_data(
        n_samples=200, n_timesteps=24, task="forecasting"
    )
    dls = TSDataLoaders.from_dsets(
        X[splits[0]], y[splits[0]], X[splits[1]], y[splits[1]]
    )
    model = InceptionTime(dls.vars, dls.c)
    learn = TSRegressor(dls, model)
    learn.fit_one_cycle(10, 0.001)


def apply_random_cropping_and_noise() -> None:
    dls.add_tfms([RandomCrop(0.9), RandomNoise(0.05)])
    learn.fit_one_cycle(10, 0.001)


def visualize_feature_importance_using_class_activations() -> None:
    cam = CAM(learn)
    cam.plot(X[splits[1]][0])


def main() -> None:
    download_and_load_the_arrowhead_dataset()
    create_dataloaders_for_training_and_validation()
    initialize_the_model_and_learner()
    plot_training_metrics()
    generate_synthetic_regression_data()
    generate_synthetic_forecasting_data()
    apply_random_cropping_and_noise()
    visualize_feature_importance_using_class_activations()


if __name__ == "__main__":
    main()
