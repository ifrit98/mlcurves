from .cluster import cluster

from .complexity_curves import complexity_curves_npy, complexity_curves_tf
from .train_curves import train_set_size_curves_npy, train_set_size_curves_tf
from .learnrate_schedule_curves import learn_rate_range_test, AddLearningRateToHistory

from .models import antirectifier
from .models import convnext
from .models import vit
from .models import bayes
from .models import lstm

from .curve_utils import env, permutation, random_permutation, sample_pd
from .curve_utils import create_random_indices, shuffle_sk, shufflej, vec_matrix
from .curve_utils import take_subset, plot_metrics, mnist_model, mnist_tfds, mnist_npy
from .curve_utils import fargs, timestamp, product, get_param_count, batch_generator

from .model_paces import model_paces


def demo(n_val=2000):
    n_val=2000
    import numpy as np
    import tensorflow as tf
    ds, ts = mnist_tfds(shuffle=True)

    ds = ds.map(lambda x,y: (
        tf.reshape(x, tf.constant([-1, np.product(x.shape[1:])])), y)
    )
    ts = ts.map(lambda x,y: (
        tf.reshape(x, tf.constant([-1, np.product(x.shape[1:])])), y)
    )

    vs = ts.take(n_val)
    ts = ts.skip(n_val)

    for x in ds: break
    input_shape = x[0].shape[1:]
    num_classes = len(np.unique(x[1]))

    from mlcurves.models.antirectifier import build_antirectifier_dense, dense_configs
    from mlcurves import model_paces

    paces = model_paces(
        build_antirectifier_dense, 
        input_shape,
        num_classes=num_classes,
        train_ds=ds, test_ds=ts, val_ds=vs,
        cfg_dict=dense_configs,
    )
