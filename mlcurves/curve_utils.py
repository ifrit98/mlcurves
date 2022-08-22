import time
from warnings import warn
from functools import reduce
from inspect import signature

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import tensorflow as tf

fargs = lambda f: list(signature(f).parameters.keys())
timestamp = lambda: time.strftime("%m_%d_%y_%H-%M-%S", time.strptime(time.asctime()))
product = lambda x: reduce(lambda a,b: a*b, x)
is_onehot = lambda y: len(y.shape) > 1


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    def __iter__(self):
        return iter(self.__dict__.items())
    def add_to_namespace(self, **kwargs):
        self.__dict__.update(kwargs)

def env(**kwargs):
    return Namespace(**kwargs)
namespace = environment = env


# Padding sequences
def get_max_shape(xs):
    unlist = lambda x: x[0] if hasattr(x, "__len__") else x
    shape = lambda x: np.asarray(x).shape
    lmap = lambda f, x: list(map(f, x))
    return unlist(max(lmap(shape, xs)))

pad_sequences = lambda x: tf.keras.preprocessing.sequence.pad_sequences(
    x, maxlen=get_max_shape(x)
)

def permutation(x):
    """Return the indices of random permutation of `x`"""
    return np.random.permutation(len(x) if hasattr(x, '__len__') else int(x))


def random_permutation(x):
    """Return a random permutation of `x` across its zero-axis"""
    return x[permutation(x)]


def split_Xy(X, y, split=0.1):
    perm = np.random.permutation(len(X))
    return (
        X[perm[int(len(X)*split):]], y[perm[int(len(X)*split):]]), (
            X[perm[:int(len(X)*split)]], y[perm[:int(len(X)*split)]]
        )


def get_param_count(model):
    trainableParams = np.sum(
        [np.prod(v.get_shape()) for v in model.trainable_weights]
    )
    nonTrainableParams = np.sum(
        [np.prod(v.get_shape()) for v in model.non_trainable_weights]
    )
    return trainableParams + nonTrainableParams


def import_history(path='history/model_history'):
    import pickle
    with open(path, 'rb') as f:
      history = pickle.load(f)
    return history


def batch_generator(X, y, batch_size, shuffle=False):
    indices = np.arange(len(X)) 
    batch=[]
    i=0
    while True:
            # it might be a good idea to shuffle your data before each epoch
            if shuffle:
                np.random.shuffle(indices) 
            for i in indices:
                batch.append(i)
                if len(batch)==batch_size:
                    yield np.stack(X[batch]), np.stack(y[batch])
                    batch=[]


def sample_pd(xs, ys, sample_size, shuffle_out=True):
    """
    Sample `sample_size` (float %) from numpy arrays while preserving original distribution.
    """
    argmax = np.argmax(ys, axis=1) if len(ys.shape) > 1 else ys
    classes = np.unique(argmax)
    percentages = dict.fromkeys([c for c in classes])
    for c in classes:
        idx = np.where(argmax == c)[0]
        percentages[c] = len(idx) / len(ys)

    labels = list(map(lambda x: str(x), np.unique(argmax)))
    _p = {labels[i]: v for i,v in enumerate(percentages.values())}
    p = percentages

    idx2class = dict(zip(list(p), list(_p)))
    class2idx = {v: k for k,v in idx2class.items()}

    out_idx = dict.fromkeys(labels)

    for cl in idx2class.values():
        ci = np.where(argmax == class2idx[cl])[0]
        idx = int(len(ci)*sample_size)
        out_idx[cl] = ci[:idx]

    sample_idxs = np.concatenate(list(out_idx.values()))

    if shuffle_out:
        np.random.shuffle(sample_idxs) # shuffle indices before separating into xs,ys

    return np.stack(xs[sample_idxs]), np.stack(ys[sample_idxs])


# https://likegeeks.com/numpy-shuffle/
def create_random_indices(xs, ys, 
                          labels, 
                          test_size=0.1, 
                          val_size=0.02, 
                          shuff=True, verbose=0):
    """
    Generate random indices for training, valdation, and testing datasets.

    Usage:
    X = np.random.normal([784,])
    y = np.random.integers
    labels = 
    train_idx, val_idx, test_idx = create_random_indices(xs, ys, labels=labels)
    """
    argmax = np.argmax(ys, axis=1) if len(ys.shape) > 1 else ys
    classes = np.unique(argmax)
    percentages = dict.fromkeys([c for c in classes])
    for c in classes:
        idx = np.where(argmax == c)[0]
        percentages[c] = len(idx) / len(ys)

    test_size = test_size - val_size # default: 0.08
    assert test_size + val_size <= 0.1

    _p = {labels[i]: v for i,v in enumerate(percentages.values())}
    p = percentages

    idx2class = dict(zip(list(p), list(_p)))
    class2idx = {v: k for k,v in idx2class.items()}

    tr, va, te = list(
        map(
            lambda x: dict.fromkeys(labels), 
            range(3)
        )
    )
    for cl in idx2class.values():
        ci = class_idx = np.where(argmax == class2idx[cl])[0]
        test_idx = int(len(ci)*test_size)
        val_idx = test_idx + int(len(ci)*val_size)
        if verbose:
            print("\nclass index:", len(ci))
            print("test_len:", len(ci[:test_idx]))
        te[cl] = ci[:test_idx]
        ci = ci[test_idx:]
        val_idx -= test_idx
        va[cl] = ci[:val_idx]
        tr[cl] = ci[val_idx:]
        if verbose:
            print("val_len:", len(ci[:val_idx]))
            print("train_len:", len(ci[val_idx:]))
            print("total:", len(va[cl]) + len(tr[cl]) + len(te[cl]))

    train, val, test = list(
        map(lambda x: np.concatenate(list(x.values())), [tr, va, te]))

    if verbose:
        print(len(train) / len(xs))
        print(len(val) / len(xs))
        print(len(test) / len(xs))
        print(len(test) / len(xs) + len(train) / len(xs) + len(val) / len(xs))
        print(len(test) + len(train) + len(val))

    assert len(test) + len(train) + len(val) == len(xs)
    assert not any(set(train).intersection(set(val)))
    assert not any(set(train).intersection(set(test)))

    if shuff: 
        np.random.shuffle(train); np.random.shuffle(val)
    return train, val, test

train_val_test_indices = create_random_indices


def shuffle_sk(X, y, val_size=0.05, test_size=0.1):
    _X, test_X, _y, test_y = train_test_split(X, y, test_size=test_size)
    train_X, val_X, train_y, val_y = train_test_split(_X, _y, test_size=val_size)
    print("Returning: {}% train, {}% val, {}% test".format(
        (train_X.shape[0] / len(X) * 100),
        (val_X.shape[0] / len(X) * 100),
        (test_X.shape[0] / len(X) * 100))
    )
    return (train_X, train_y), (val_X, val_y), (test_X, test_y)

train_val_test_split = shuffle_sk



def shufflej(X, y, labels=None, stack=True):
    if labels is None:
        labels = list(map(
            lambda x: str(x), 
            np.unique(y if len(y.shape) == 1 else np.argmax(y)))
        )
    train, val, test = create_random_indices(X, y, labels)
    if stack:
        return (
            np.stack(X[train]), y[train]), \
                (np.stack(X[val]), y[val]), (np.stack(X[test]), y[test]
        )
    return (X[train], y[train]), (X[val], y[val]), (X[test], y[test])


def vec_matrix(X):
    """
    Vectorize matrix `X` along axis=0.
    
    Equivalent to (and more efficent than) calling .ravel() on each element in X along axis 0.
    e.g
    >>> for i in range(len(X)):
    >>>     X[i] = X[i].ravel()

    Usage:
    >>> X = np.random.normal([100, 28, 28])
    >>> X.shape
    >>> (100, 28, 28)
    >>> v = vec_matrix(X)
    >>> v.shape
    >>> (100, 784)
    """
    if len(X.shape) > 2:
        X = np.reshape(
            X, [X.shape[0]] + [product(X.shape[1:])]
        )
    return X
unravel = vec_matrix


def take_subset(X, p):
    """ 
    Efficiently take a p percentage (%) subset of X.
    """
    if p > 1:
        p = p / 100
    if len(X.shape) > 2:
        X = vec_matrix(X)
    perm = np.random.permutation(X.shape[0])
    return X[perm[:int(len(X)*p)]]
subset = take_subset



def process_history(results, histories):
    train_errs, val_errs, test_errs = [], [], []
    train_losses, train_accs = [], []
    val_losses, val_accs, test_losses, test_accs = [], [], [], []

    for (test_loss, test_acc), history in zip(results.values(), histories.values()):
        val_acc = history.history['val_accuracy'][-1]
        train_acc = history.history['accuracy'][-1]
        train_loss = history.history['loss'][-1]
        val_loss = history.history['val_loss'][-1]

        train_errs.append(1 - train_acc)
        val_errs.append(1 - val_acc)
        test_errs.append(1 - test_acc)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

    total_hist = env()
    total_hist.history = {
        'train_loss': train_losses, 'train_accuracy': train_accs,
        'val_loss': val_losses, 'val_accuracy': val_accs,
        'test_loss': test_losses, 'test_accuracy': test_accs
        }
    return total_hist



# Plotting routine for history objects
def plot_metrics(history,
                 show=False,
                 save_png=True,
                 xlab=None, ylab=None,
                 xticks=None, xtick_labels=None,
                 outpath='training_curves_' + timestamp()):
    sns.set()
    plt.clf()
    plt.cla()

    keys = list(history.history)
    epochs = range(
        min(list(map(lambda x: len(x[1]), history.history.items())))
    )  

    ax = plt.subplot(211)
    if 'acc' in keys:
        acc  = history.history['acc']
    elif 'accuracy' in keys:
        acc  = history.history['accuracy']
    elif 'train_acc' in keys:
        acc  = history.history['train_acc']
    elif 'train_accuracy' in keys:
        acc  = history.history['train_accuracy']
    else:
        raise ValueError("Training accuracy not found")


    plt.plot(epochs, acc, color='green', 
        marker='+', linestyle='dashed', label='Training accuracy'
    )

    if 'val_acc' in keys:
        val_acc = history.history['val_acc']
    elif 'val_accuracy' in keys:
        val_acc = history.history['val_accuracy']
    else:
        raise ValueError("Validation accuracy not found")

    plt.plot(epochs, val_acc, color='blue', 
        marker='o', linestyle='dashed', label='Validation accuracy'
    )

    if 'test_acc' in keys:
        test_acc = history.history['test_acc']
        plt.plot(epochs, test_acc, color='red', 
            marker='x', linestyle='dashed', label='Test Accuracy'
        )
    elif 'test_accuracy' in keys:
        test_acc = history.history['test_accuracy']
        plt.plot(epochs, test_acc, color='red', 
            marker='x', linestyle='dashed', label='Test Accuracy'
        )
    else:
        warn("Test accuracy not found... skipping")


    plt.title('Training, validation and test accuracy')
    plt.legend()

    ax2 = plt.subplot(212)
    if 'loss' in keys:
        loss = history.history['loss']
    elif 'train_loss' in keys:
        loss = history.history['train_loss']
    else:
        raise ValueError("Training loss not found")

    plt.plot(epochs, loss, color='green', 
        marker='+', linestyle='dashed', label='Training Loss'
    )

    if 'val_loss' in keys:
        val_loss = history.history['val_loss']
    elif 'validation_loss' in keys:
        val_loss = history.history['validation_loss']
    else:
        raise ValueError("Validation loss not found")

    plt.plot(epochs, val_loss, color='blue', 
        marker='o', linestyle='dashed', label='Validation Loss'
    )

    if 'test_loss' in keys:
        test_loss = history.history['test_loss']
        plt.plot(epochs, test_loss, color='red', marker='x', label='Test Loss')

    plt.title('Training, validation, and test loss')
    plt.legend()

    if xlab:
        plt.xlabel(xlab)
    if ylab:
        plt.ylabel(ylab)
    if xticks is not None:
        ax.set_xticks(xticks)
        ax2.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels)
        ax2.set_xticklabels(xtick_labels)

    plt.tight_layout()

    if save_png:
        plt.savefig(outpath)
    if show:
        plt.show()


def plot_metrics2(history, acc='accuracy', loss='loss', 
                 val_acc='val_accuracy', val_loss='val_loss'):
    acc      = history.history[acc]
    val_acc  = history.history[val_acc]
    loss     = history.history[loss]
    val_loss = history.history[val_loss]
    epochs   = range(len(acc))
    sns.set()
    plt.plot(epochs, acc, 'bo', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


def mnist_model(lr=0.001):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(64,activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(lr),
        metrics=['accuracy'],
    )
    return model


def mnist_tfds(shuffle=True):
    import tensorflow_datasets as tfds
    
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=shuffle,
        as_supervised=True,
        with_info=True,
    )
    normalize_img = lambda img, lbl: (tf.cast(img, tf.float32) / 255., lbl)

    # Train
    ds_train = ds_train.map(
        normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    # Test
    ds_test = ds_test.map(
        normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.batch(128)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    return (ds_train, ds_test)


def mnist_npy(subsample=False, 
              take_n=3000, 
              shuffle=True, 
              vectorize=True, 
              norm=True, 
              return_test=True):
    from tensorflow.keras.datasets import mnist
    (x_train, y_train), (x_test , y_test) = mnist.load_data()

    if shuffle:
        train_idx = permutation(len(x_train))
        test_idx  = permutation(len(x_test))

        x_train = x_train[train_idx]
        y_train = y_train[train_idx]
        x_test  = x_test[test_idx]
        y_test  = y_test[test_idx]

    if subsample:
        x_train = x_train[:take_n]
        y_train = y_train[:take_n]

    if vectorize:
        x_train = np.reshape(
            x_train, [x_train.shape[0], x_train.shape[1]*x_train.shape[2]]
        )
        x_test  = np.reshape(
            x_test, [x_test.shape[0], x_test.shape[1]*x_test.shape[2]]
        )

    # Normalize images
    if norm:
        x_train = x_train.astype(float) / 255.0
        x_test  = x_test.astype(float) / 255.0

    if return_test:
        print("Loading MNIST with shape:\ntrain: {}\ntest:  {}".format(
            x_train.shape, x_test.shape)
        )
        return (x_train, y_train), (x_test, y_test)
    else:
        print("Loading MNIST with shape:\ntrain: {}".format(x_train.shape))
        return x_train, y_train