"""Generated from Jupyter notebook: TSAI

Magics and shell lines are commented out. Run with a normal Python interpreter."""


# --- code cell ---

# !pip install tsai  # Jupyter-only


# --- code cell ---

from tsai.data.external import get_UCR_data


def main():
    # Download and load the ArrowHead dataset
    X, y, splits = get_UCR_data("ArrowHead", split_data=True)
    print(X.shape, y.shape)  # (Samples, Timesteps, Features)


    # --- code cell ---

    dsid = "OliveOil"
    X_train, y_train, X_valid, y_valid = get_UCR_data(
        dsid, on_disk=True, force_download=True
    )
    X_on_disk, y_on_disk, splits = get_UCR_data(
        dsid, on_disk=True, return_split=False, force_download=True
    )
    X_in_memory, y_in_memory, splits = get_UCR_data(
        dsid, on_disk=False, return_split=False, force_download=True
    )
    y_tensor = cat2int(y_on_disk)
    y_array = y_tensor.numpy()


    # --- code cell ---

    import tsai.basics as ts

    dsid = "NATOPS"
    bs = 16
    X, y, splits = ts.get_UCR_data(dsid, return_split=False)
    tfms = [None, [TSCategorize()]]
    dsets = ts.TSDatasets(X, y, tfms=tfms, splits=splits)
    dls = ts.TSDataLoaders.from_dsets(
        dsets.train, dsets.valid, bs=bs, num_workers=0, shuffle=False
    )
    model = ts.RNN(
        c_in,
        c_out,
        hidden_size=100,
        n_layers=2,
        bidirectional=True,
        rnn_dropout=0.5,
        fc_dropout=0.5,
    )
    learn = ts.Learner(dls, model, metrics=accuracy)
    ts.learn.fit_one_cycle(1, 3e-3)


if __name__ == "__main__":
    main()
