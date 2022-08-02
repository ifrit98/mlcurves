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
                          
    if not os.path.exists(outpath): os.mkdir(outpath)

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

    model_basename = nm.split("_")[0]
    total_history = process_history(results, histories)

    plot_metrics(
        total_history, show=False, xlab="Model complexity (# params)", 
        ylab="Crossentropy Loss",
        xticks=range(len(model_sizes)), xtick_labels=list(model_sizes.values()),
        outpath=os.path.join(outpath, '{}_complexity_curves.png'.format(model_basename))
    )

    return total_history



def complexity_curves_tf(model_fn, 
                         input_shape,
                         num_classes,
                         train_ds, 
                         val_ds, 
                         test_ds, 
                         configs,
                         epochs=10,
                         batch_size=16,
                         outpath='./plot'):

    if not os.path.exists(outpath): os.mkdir(outpath)

    # Must fully prepare data beforehand for model ingestion (e.g. batch, repeat, prefetch)
    train_ds = train_ds.batch(batch_size=batch_size)
    val_ds = val_ds.batch(batch_size=batch_size)
    test_ds = test_ds.batch(batch_size=batch_size)

    histories = {}
    results = {}
    model_sizes = {}

    config_by_name = 'model_nm' in fargs(model_fn)

    for i, (nm, cfg) in enumerate(configs.items()):
        if config_by_name:
            model = model_fn(
                input_shape=input_shape, num_classes=num_classes, model_nm=nm
            )
        else:
            model = model_fn(
                input_shape=input_shape, num_classes=num_classes, **cfg
            )

        h = model.fit(
            train_ds, 
            validation_data=val_ds, 
            epochs=epochs,
            batch_size=batch_size
        )
        histories[i] = h
        plot_metrics(h, show=False, outpath=os.path.join(
            outpath, 'training_curves_{}'.format(i))
        )

        r = model.evaluate(test_ds)
        results[i] = r

        model_sizes[i] = get_param_count(model)

    model_basename = nm.split("_")[0]
    total_history = process_history(results, histories)

    plot_metrics(
        total_history, show=False, xlab="Model complexity (# params)", 
        ylab="Crossentropy Loss",
        xticks=range(len(model_sizes)), xtick_labels=list(model_sizes.values()),
        outpath=os.path.join(outpath, '{}_complexity_curves.png'.format(model_basename))
    )

    return total_history




def test(conv1D=True, subset=False):
    set_based_gpu()

    import tensorflow as tf
    import tensorflow_datasets as tfds
    from mlcurves import models, complexity_curves_tf, complexity_curves_npy
    from mlcurves.curve_utils import fargs, plot_metrics, get_param_count

    conv1D=True
    model_fn = models.antirectifier.build_antirectifier_cnn_1D \
        if conv1D else models.antirectifier.build_antirectifier_cnn_1D
    configs = models.antirectifier.cnn_configs

    # For tensorflow datasets
    ds, ts = tfds.load('mnist', split=['train', 'test'], shuffle_files=True)
    if conv1D:
        train_ds = ds.map(lambda x: (tf.reshape(x['image'], [-1, 1]), x['label']))
        test_ds = ts.map(lambda x: (tf.reshape(x['image'], [-1, 1]), x['label']))
    else:
        train_ds = ds.map(lambda x: (x['image'], x['label']))
        test_ds =  ts.map(lambda x: (x['image'], x['label']))

    for x in train_ds: break
    input_shape = x[0].shape
    num_classes = 10
    del x

    batch_size = 32
    epochs = 10

    val_size = 1500
    val_ds = train_ds.take(val_size)
    train_ds = train_ds.skip(val_size)

    subset=True
    if subset:
        train_ds = train_ds.take(100).shuffle(100)
        val_ds = val_ds.take(5)
        test_ds = test_ds.take(10)

    complexity_curves_tf(
        model_fn=model_fn,
        train_ds=train_ds, val_ds=val_ds, test_ds=test_ds, num_classes=num_classes,
        batch_size=batch_size, epochs=epochs, input_shape=input_shape, configs=configs, 
        outpath='./plots'
    )


    # For numpy arrays
    trainXy = np.asarray(list(ds.as_numpy_iterator()))
    testXy = np.asarray(list(ts.as_numpy_iterator()))
    X = np.concatenate([trainXy, testXy])
    y = X[:, 1]
    X = X[:, 0]


    complexity_curves_npy(
        model_fn=model_fn, X=X, y=y,
        epochs=epochs, batch_size=batch_size, num_classes=num_classes,
        input_shape=input_shape, configs=configs, outpath='./plots'
    )