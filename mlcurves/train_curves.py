import os
import numpy as np
import tensorflow as tf

from .curve_utils import process_history, plot_metrics, batch_generator
from .curve_utils import  shuffle_sk, shufflej, sample_pd


def train_set_size_curves_tf(model_fn, trainset, valset, 
                             testset, num_classes,
                             batch_size=16,
                             epochs=10, n_runs=11, 
                             shuffle_init=True, 
                             buffer_size=None,
                             outpath='./plot'):
    # Takes model and tensorflow Models and data.Dataset objects ONLY
    # Assume model is compiled properly before being passed as an argument

    # Get monotonically increasing range based on percentages
    train_len = len(list(trainset.as_numpy_iterator()))

    if not os.path.exists(outpath): os.mkdir(outpath)

    if buffer_size is None:
        buffer_size = train_len

    if shuffle_init:
        trainset = trainset.shuffle(buffer_size)
        valset = valset.shuffle(buffer_size)
        testset = testset.shuffle(buffer_size)

    u = 1 / n_runs
    rng = np.arange(u, 1+u, u)

    # Split datasets into random subsets of increasing size, e.g. [10%, 20%,..., 100%]
    train_sizes = [int(train_len * p) for p in rng]
    tr_prev = 0

    for x in trainset: break
    input_shape = x[0].shape

    histories = {}
    results = {}
    for i, tr in enumerate(train_sizes):
        print("Starting dataset size: (train) {}", tr)
        print("Percentage of full trainset {}%".format((tr/train_len)*100))

        ds_sub = trainset.skip(tr_prev).take(tr).shuffle(buffer_size).batch(batch_size)
        tr_prev = tr

        model = model_fn(input_shape, num_classes=num_classes)
        # TODO: STANDARDIZE THIS! (MUST KNOW if CATEGORICAL CROSSENTROPY OR BINARY)
        history = model.fit(ds_sub, validation_data=valset.batch(batch_size), epochs=epochs)
        histories[i] = history

        res = model.evaluate(testset.batch(batch_size))
        results[i] = res
        
        plot_metrics(history, show=False, outpath=os.path.join(
            outpath, 'training_curves_{}'.format(i))
        )

        del model

    total_history = process_history(results, histories)

    plot_metrics(
        total_history, show=False, xlab="Train Set Size (#)", 
        xticks=range(len(train_sizes)), xtick_labels=train_sizes,
        outpath=os.path.join(outpath, 'final_train_size_curves.png')
    )
    return total_history


def train_set_size_curves_npy(model_fn, X, y, num_classes, 
                              prepared_npy_datasets=None, 
                              epochs=10, labels=None, n_runs=10, batch_size=16, 
                              preserve_dist=True, outpath='./plot'):
    # Takes model and tensorflow Models and data.Dataset objects ONLY
    # Assume model is compiled properly before being passed as an argument
    
    if prepared_npy_datasets is not None:
        train_X, train_y, val_X, val_y, test_X, test_y = prepared_npy_datasets

    if preserve_dist:
        (train_X, train_y), (val_X, val_y), (test_X, test_y) = shufflej(X, y, labels)
    else:
        (train_X, train_y), (val_X, val_y), (test_X, test_y) = shuffle_sk(X, y, labels)

    input_shape = train_X.shape[1:]

    # Get monotonically increasing range based on percentages
    u = 1 / n_runs
    train_sizes = np.arange(u, 1+u, u)

    val_X  = tf.convert_to_tensor(val_X, dtype='int32')
    val_y  = tf.convert_to_tensor(val_y, dtype='int32')
    test_X = tf.convert_to_tensor(test_X, dtype='int32')
    test_y = tf.convert_to_tensor(test_y, dtype='int32')

    histories = {}
    results = {}
    # Split datasets into random subsets of increasing size, e.g. [10%, 20%,..., 100%]
    for i, tr in enumerate(train_sizes):
        print("Percentage of full trainset {}%".format(tr*100))

        # Get data subset and shuffle
        train_X_sub, train_y_sub = sample_pd(train_X, train_y, sample_size=tr)

        # Batch it
        traingen = batch_generator(train_X_sub, train_y_sub, batch_size)

        # Reinstantiate model
        model = model_fn(input_shape, num_classes=num_classes)

        # Train and record
        history = model.fit(
            traingen, validation_data=(val_X, val_y), epochs=epochs, 
            batch_size=batch_size, steps_per_epoch=len(train_X_sub) // batch_size
        )
        res = model.evaluate(test_X, test_y)

        histories[i] = history
        results[i] = res
        
        plot_metrics(history, show=False, outpath=os.path.join(
            outpath, 'training_curves_{}'.format(i))
        )

        del model

    total_history = process_history(results, histories)

    plot_metrics(
        total_history, show=False, xlab="Train Set Size (# examples)", 
        xticks=range(len(train_sizes)), xtick_labels=(train_sizes*len(train_X)).astype(int),
        outpath=os.path.join(outpath, 'final_train_size_curves.png')
    )
    return total_history


def test():
    import tensorflow_datasets as tfds
    from .models.antirectifier import build_antirectifier_cnn_1D
    
    # For tensorflow datasets
    ds, ts = tfds.load('mnist', split=['train', 'test'], shuffle_files=True)

    ds = ds.map(lambda x: (tf.reshape(x['image'], [-1]), x['label']))
    ts = ts.map(lambda x: (tf.reshape(x['image'], [-1]), x['label']))

    # For numpy arrays
    trainXy = np.asarray(list(ds.as_numpy_iterator()))
    testXy = np.asarray(list(ts.as_numpy_iterator()))
    X = np.concatenate([trainXy, testXy])
    y = X[:, 1]
    X = X[:, 0]


    batch_size=32
    n_runs=11
    epochs=10

    val_size = 1500
    trainset = ds.skip(val_size)
    valset = ds.take(val_size)
    testset = ts

    train_set_size_curves_tf(
        model_fn=build_antirectifier_cnn_1D, num_classes=10,
        trainset=trainset, valset=valset, testset=testset, 
        batch_size=batch_size, n_runs=n_runs, epochs=epochs
    )

    train_set_size_curves_npy(
        build_antirectifier_cnn_1D, X, y, num_classes=10,
        epochs=epochs, batch_size=batch_size, n_runs=n_runs
    )