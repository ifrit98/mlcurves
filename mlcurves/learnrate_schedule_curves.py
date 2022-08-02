import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import backend as K

from .curve_utils import mnist_tfds, mnist_model, timestamp, plot_metrics

add_time = lambda s: s + '_' + timestamp()



def ploty(y, x=None, xlab='obs', ylab='value', 
          save=True, title='', filepath='plot'):
    sns.set()
    if x is None: x = np.linspace(0, len(y), len(y))
    filepath = add_time(filepath)
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set(xlabel=xlab, ylabel=ylab, title=title)
    ax.grid()
    best_lr = x[np.argmin(y)]
    plt.axvline(x=best_lr, color='r', label='Best LR {:.4f}'.format(best_lr))
    plt.legend()
    if save:
       fig.savefig(filepath)
    plt.show()
    return filepath


def plot_lr_range_test_from_hist(history,
                                filename="lr_range_test",
                                max_loss=5,
                                max_lr=1):
    loss = np.asarray(history.history['loss'])
    lr   = np.asarray(history.history['lr'])
    cut_index = np.argmax(loss > max_loss)
    if cut_index == 0:
        print("\nLoss did not exceed `MAX_LOSS`.")
        print("Increase `epochs` and `MAX_LR`, or decrease `MAX_LOSS`.")
        print("\nPlotting with full history. May be scaled incorrectly...\n\n")
    else:
        loss[cut_index] = max_loss
        loss = loss[:cut_index]
        lr = lr[:cut_index]
    
    lr_cut_index = np.argmax(lr > max_lr)
    if lr_cut_index != 0:
        lr[lr_cut_index] = max_lr
        lr = lr[:lr_cut_index]
        loss = loss[:lr_cut_index]

    ploty(
        loss, lr, 
        xlab='Learning Rate', ylab='Loss', 
        filepath=filename
    )


def infer_best_lr_params(history, factor=3): 
    idx = tf.argmin(history.history['loss'])
    best_run_lr = history.history['lr'][idx]
    min_lr = best_run_lr / factor
    return [min_lr, best_run_lr, idx]



class AddLearningRateToHistory(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if 'lr' not in self.model.history.history.keys():
            self.model.history.history['lr'] = list()
        optimizer = self.model.optimizer
        lr = K.eval(optimizer.lr)
        self.model.history.history['lr'].append(lr)


def fold_histories(histories):
    """
    Go from list of history objects to history object with [-1] of each as entry per epoch
    e.g.
    histories = [ 
        hist1.history = {
            'acc':  [0, ..., 0.81],
            'loss': [inf, ..., 0.4],
            'lr':   [1e-4, ..., 1e-4]
        },
        hist2.history = {
            'acc':  [0, ..., 0.99],
            'loss': [inf, ..., 0.1],
            'lr':   [1e-4, ..., 1e-4]

        } 
    ]
    --> histories.history = {
        'acc':  [0.81, 0.99],
        'loss': [ 0.4,  0.1],
        'lr':   [1e-4, 1e-5]
    }

    """
    history = {k: [] for k in histories[0].history.keys()}
    
    for hist in histories.values():
        {history[k].append(v[-1]) for k,v in hist.history.items()}

    # prepare final hist object
    hist.history = history
    hist.epoch = list(range(len(histories)))
    hist.params['epochs'] = len(histories)

    if 'model' in hist.__dict__.keys():
        del hist.model

    return hist



def learn_rate_range_test2(model_fn, ds,
                           init_lr=1e-4,
                           factor=3,
                           plot=True,
                           steps_per_epoch=None,
                           max_lr=2,
                           max_loss=2, 
                           epochs=10,
                           n_runs=20,
                           save_hist=True, 
                           verbose=1):
    """
    Perform a learn rate range test using multiple epochs per learn_rate.
    """

    schedule = lambda epoch: init_lr * tf.pow(
        tf.pow(max_lr / init_lr, 1 / (n_runs - 1)), epoch
    )

    learn_rates = list(
        map(lambda x: schedule(x), list(range(n_runs)))
    )

    histories = {}

    for i, lr in enumerate(learn_rates):
        print("=============================================")
        print("learn rate: {}".format(lr))
        model = model_fn(lr=lr)

        if steps_per_epoch is not None:
            hist = model.fit(
                ds,
                epochs=epochs,
                steps_per_epoch=int(steps_per_epoch),
                callbacks=[AddLearningRateToHistory()],
                verbose=verbose)
        else:
            hist = model.fit(
                ds,
                epochs=epochs,
                callbacks=[AddLearningRateToHistory()],
                verbose=verbose)

        histories[i] = hist

        del model

    # Accumulate history objects into single history
    lr_hist = fold_histories(histories)

    if save_hist:
        from pickle import dump
        f = open("lr-range-test-history", 'wb')
        dump(lr_hist.history, f)
        f.close()

    min_lr, best_lr = infer_best_lr_params(lr_hist, factor)

    if plot:
        plot_lr_range_test_from_hist(
            lr_hist, max_lr=max_lr, max_loss=max_loss
        )

    return (min_lr, best_lr), lr_hist



# Reference: https://arxiv.org/pdf/1708.07120.pdf%22
def learn_rate_range_test(model, ds, init_lr=1e-4, factor=3, plot=True, steps_per_epoch=None,
                          max_lr=3, max_loss=2, epochs=25, save_hist=True, verbose=1):
    """
    Perform a learn rate range test using a single epoch per learn_rate. (paper version)
    """
    lr_range_callback = tf.keras.callbacks.LearningRateScheduler(
        schedule = lambda epoch: init_lr * tf.pow(
            tf.pow(max_lr / init_lr, 1 / (epochs - 1)), epoch))

    if steps_per_epoch is not None:
        hist = model.fit(
            ds,
            epochs=epochs,
            steps_per_epoch=int(steps_per_epoch),
            callbacks=[lr_range_callback],
            verbose=verbose)
    else:
        hist = model.fit(
            ds,
            epochs=epochs,
            callbacks=[lr_range_callback],
            verbose=verbose)

    if save_hist:
        from pickle import dump
        f = open("lr-range-test-history", 'wb')
        dump(hist.history, f)
        f.close()

    min_lr, best_lr, best_lr_idx = infer_best_lr_params(hist, factor)

    if plot:
        plot_lr_range_test_from_hist(
            hist, 
            max_lr=max_lr, max_loss=max_loss, best_lr_idx=best_lr_idx
        )

    return (min_lr, best_lr), hist



def demo(epochs=25, max_lr=3):

    init_lr=1e-4; factor=3; plot=True; steps_per_epoch=None
    max_lr=3; max_loss=2; epochs=25; save_hist=True; verbose=1


    print("\n\nLOADING AND PROCESSING MNIST DATA...")
    (ds, ts) = mnist_tfds()

    print("LOADING SMALL MNIST MODEL...")
    model = mnist_model()

    print("CONDUCTING LEARNING_RATE RANGE TEST2 (multiple epochs per LR)...")
    (min_lr, max_lr), lr_hist = learn_rate_range_test2(
        mnist_model, ds, max_lr=max_lr, epochs=epochs)
    print("Minimum learning rate:", min_lr)
    print("Maximum learning rate:", max_lr)


    print("CONDUCTING LEARNING_RATE RANGE TEST...")
    (min_lr, max_lr), lr_hist = learn_rate_range_test(
        model, ds, max_lr=max_lr, epochs=epochs)
    print("Minimum learning rate:", min_lr)
    print("Maximum learning rate:", max_lr)

    print("\nRECOMPILING MODEL WITH NEW LEARNING RATE PARAM...")
    model = mnist_model(lr=max_lr)

    print("FITTING NEW MODEL TO DATA WITH NEW LR {}...".format(max_lr))
    history = model.fit(
        ds,
        epochs=epochs,
        validation_data=ts,
        verbose=1
    )
    res = model.evaluate(ts); print("Final results:\n", res)

    print("PLOTTING METRICS WITH NEW LR...")
    plot_metrics(history)

    # TODO Show improvement/differences between baseline and new runs?

    return {
        'lr_history': lr_hist,
        'history': history, 
        'min_max_lr': (min_lr, max_lr)
    }