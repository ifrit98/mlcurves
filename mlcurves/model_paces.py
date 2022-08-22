import os

from .learnrate_schedule_curves import learn_rate_range_test
from .complexity_curves import complexity_curves_tf
from .train_curves import train_set_size_curves_tf


def model_paces(model_fn, input_shape, num_classes, train_ds, test_ds, cfg_dict, val_ds=None, outpath="."):
    """
    Put a model through its paces.

    (1) Runs a learning rate scheduler to find best learning rate parameters for this model, 
    given a particular dataset.

    (2) Runs a routine to train and test model on increasing data set sizes (takes subsets)

    (3) Runs a rounine to infer average best model size (measured in # parameters)

    (4) Returns the results as `dict` and saves plots to `outpath/*.png`
    """
    
    model = model_fn(input_shape, num_classes)

    (min_lr, init_lr), h = learn_rate_range_test(model, train_ds, outpath=os.path.join(
        outpath, "lr_range_test")
    )


    train_size_history = train_set_size_curves_tf(
        model_fn, train_ds, val_ds, test_ds, epochs=25,
        outpath=os.path.join(outpath, "train_size_test")
    )


    complexity_history = complexity_curves_tf(
        model_fn, configs=cfg_dict, train_ds=train_ds, val_ds=val_ds, test_ds=test_ds,
        outpath=os.path.join(outpath, "complexity_test")
    )

    return {
        'min_lr': min_lr,
        'init_lr': init_lr,
        'train_size_history': train_size_history,
        'complexity_history': complexity_history,
        'lr_range_history': h
    }
    # TODO: Train using pipeline that writes eval out with new `init_lr`
