import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import backend as K

timestamp = lambda: time.strftime("%m_%d_%y_%H-%M-%S", time.strptime(time.asctime()))
add_time = lambda s: s + '_' + timestamp()



def mnist_data():
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
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


def plot_metrics(history, acc='accuracy', loss='loss', 
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




def learn_rate_range_test2(model_fn, ds, init_lr=1e-4, factor=3, plot=True, steps_per_epoch=None,
                           max_lr=3, max_loss=2, epochs=25, save_hist=True, verbose=1):

    lr_range_callback = tf.keras.callbacks.LearningRateScheduler(
        schedule = lambda epoch: init_lr * tf.pow(
            tf.pow(max_lr / init_lr, 1 / (epochs - 1)), epoch))

    schedule = lambda epoch: init_lr * tf.pow(
        tf.pow(max_lr / init_lr, 1 / (epochs - 1)), epoch
    )

    sched = list(map(lambda x: schedule(x), list(range(epochs))))

    for lr in sched:

        model = model_fn()

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





def learn_rate_range_test(model, ds, init_lr=1e-4, factor=3, plot=True, steps_per_epoch=None,
                          max_lr=3, max_loss=2, epochs=25, save_hist=True, verbose=1):

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
    (ds, ts) = mnist_data()

    print("LOADING SMALL MNIST MODEL...")
    model = mnist_model()

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