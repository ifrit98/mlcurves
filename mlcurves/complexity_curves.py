import os
import numpy as np

from .curve_utils import process_history, plot_metrics, split_Xy, get_param_count, fargs


def complexity_curves_npy(model_fn, 
                          X, y, 
                          configs,
                          epochs=10,
                          batch_size=16,
                          input_shape=None,
                          num_classes=None,
                          outpath='./plot'):

    (X_train, y_train), (X_test, y_test) = split_Xy(X, y, split=0.1)
    (X_test, y_test), (X_val, y_val) = split_Xy(X_test, y_test, split=0.4)
    del X, y

    if input_shape is None or num_classes is None:
        input_shape = X_train.shape[1:]
        num_classes = len(np.unique(y))

    histories = {}
    results = {}
    model_sizes = []

    config_by_name = 'model_name' in fargs(model_fn)

    for i, (nm, cfg) in enumerate(configs.items()):
        if config_by_name:
            model = model_fn(
                input_shape=input_shape, num_classes=num_classes, model_name=nm
            )
        else:
            model = model_fn(
                input_shape=input_shape, num_classes=num_classes, **cfg
            )

        h = model.fit(
            x=X_train, y=y_train, validation_data=(X_val, y_val), 
            epochs=epochs,
            batch_size=batch_size
        )
        histories[i] = h
        plot_metrics(h, show=False, outpath=os.path.join(
            outpath, 'training_curves_{}'.format(i))
        )

        r = model.evaluate(X_test, y_test)
        results[i] = r

        model_sizes.append(get_param_count(model))

    total_history = process_history(results, histories)

    plot_metrics(
        total_history, show=False, xlab="Model complexity (# params)", 
        xticks=range(len(model_sizes)), xtick_labels=model_sizes,
        outpath=os.path.join(outpath, 'final_complexity_curves.png')
    )

    return total_history


# TODO: update this so we can work directly with tensorflow dataset objects
def complexity_curves_tf(model_fn, 
                        train_ds, 
                        val_ds, 
                        test_ds, 
                        configs,
                        input_shape=None,
                        num_classes=None,
                        outpath='./plot'):
    raise NotImplementedError

    if input_shape is None or num_classes is None:
        for x in train_ds: break
        input_shape = x[0].shape[1:]
        num_classes = x[1].shape[-1] if len(x[1].shape) > 1 else len(np.unique(x[1]))
        del x

    # Must fully prepare data beforehand for model ingestion (e.g. batch, repeat, prefetch)
    histories = []
    results = []
    model_sizes = []

    for cfg in configs:
        model = model_fn(
            input_shape=input_shape, num_classes=num_classes, **cfg
        ); print(model.summary())

        histories.append(model.fit(train_ds, validation_data=val_ds))

        results.append(model.evaluate(test_ds))

        model_sizes.append(get_param_count(model))

    total_history = process_history(results, histories)

    plot_metrics(
        total_history, show=False, xlab="Model complexity (# params)", 
        xticks=range(len(model_sizes)), xtick_labels=(model_sizes).astype(int),
        outpath='plot/final_run_curves.png'
    )

    return total_history
