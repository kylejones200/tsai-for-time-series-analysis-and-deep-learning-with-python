import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, GlobalAveragePooling1D, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
np.random.seed(42)
tf.random.set_seed(42)

plt.rcParams.update({
    'font.family': 'serif',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
})

def save_fig(path: str):
    plt.tight_layout(); plt.savefig(path, bbox_inches='tight'); plt.close()

# Synthetic, reproducible dataset (no external data)
# Purpose: compare architectures with identical windows/horizon and time-aware CV

def make_sine_dataset(n=1500, freq=0.03, noise=0.3, trend=0.001):
    t = np.arange(n)
    y = 2.0 * np.sin(2 * np.pi * freq * t) + trend * t + noise * np.random.randn(n)
    return pd.DataFrame({'t': t, 'y': y})

def make_windows(y: np.ndarray, window: int, horizon: int):
    X, Y = [], []
    for i in range(len(y) - window - horizon + 1):
        X.append(y[i:i+window])
        Y.append(y[i+window:i+window+horizon])
    X = np.array(X)[:, :, None]  # add channel dim
    Y = np.array(Y)
    return X, Y

def lstm_model(window: int, horizon: int):
    m = Sequential([
        Input(shape=(window, 1)),
        LSTM(64),
        Dense(horizon),
    ])
    m.compile(optimizer='adam', loss='mae')
    return m

def cnn_model(window: int, horizon: int):
    m = Sequential([
        Input(shape=(window, 1)),
        Conv1D(64, kernel_size=5, activation='relu'),
        GlobalAveragePooling1D(),
        Dense(horizon),
    ])
    m.compile(optimizer='adam', loss='mae')
    return m

def chrono_cv_eval(X, Y, builders, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=0),
    ]
    results = {name: [] for name in builders}
    last_fold_artifacts = {}
    for fold, (tr, te) in enumerate(tscv.split(X), start=1):
        Xtr, Xte = X[tr], X[te]
        Ytr, Yte = Y[tr], Y[te]
        for name, builder in builders.items():
            model = builder(X.shape[1], Y.shape[1])
            model.fit(Xtr, Ytr, epochs=50, batch_size=32, shuffle=False, validation_data=(Xte, Yte), callbacks=callbacks, verbose=0)
            Yhat = model.predict(Xte, verbose=0)
            # Evaluate one-step MAE (first horizon step) and full-horizon MAE
            mae1 = mean_absolute_error(Yte[:, 0], Yhat[:, 0])
            maeH = mean_absolute_error(Yte.ravel(), Yhat.ravel())
            results[name].append({'fold': fold, 'mae@1': mae1, 'mae@H': maeH})
            last_fold_artifacts[name] = (model, Xte, Yte, Yhat)
    return results, last_fold_artifacts

def plot_last_fold(name, artifacts, title_suffix=""):
    model, Xte, Yte, Yhat = artifacts
    # Plot first test window forecast vs truth for visualization
    y_true = Yte[0]
    y_pred = Yhat[0]
    plt.figure(figsize=(8, 4))
    plt.plot(range(len(y_true)), y_true, label='Truth')
    plt.plot(range(len(y_pred)), y_pred, label=f'{name} forecast')
    plt.title(f'{name} horizon forecast {title_suffix}')
    plt.xlabel('Steps ahead'); plt.ylabel('Value'); plt.legend()
    save_fig(f'{name.lower()}_forecast.png')


def main():
    df = make_sine_dataset(n=1800)
    window, horizon = 48, 12
    X, Y = make_windows(df['y'].values, window=window, horizon=horizon)

    builders = {
        'LSTM': lstm_model,
        'CNN': cnn_model,
    }
    results, artifacts = chrono_cv_eval(X, Y, builders, n_splits=5)

    # Print mean metrics
    for name, rows in results.items():
        mae1 = np.mean([r['mae@1'] for r in rows]); maeH = np.mean([r['mae@H'] for r in rows])
        logger.info(f'{name}: MAE@1={mae1:.4f}, MAE@H={maeH:.4f}')

    # Plots from last fold
    for name, art in artifacts.items():
        plot_last_fold(name, art, title_suffix=f'(window={window}, horizon={horizon})')

    logger.info('Images written: lstm_forecast.png, cnn_forecast.png')

if __name__ == "__main__":
    main()
