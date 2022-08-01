# MLCurves
A lightweight python module to generate training curves for analysis during model development

## Useful tools and submodules
### Clustering
Clustering can be done using the cluster submodule, accessed through mlcurves.cluster.

Clutsering algorithms available:
- tsne
- pca
- pca_then_tsne # compute PCA reduction then visualize using T-SNE

{python}
```
import mlcurves as mc

x_train, y_train = data[...], data[...]

transform = mc.cluster.tsne(
    x_train, # can be n-dimensional floating point data
    y_train, # must be list or array of integers
    perplexity=30, # default=50 
    n_components=2,
    n_iter=300, # default=2000
    scale=True, # default=False
    scale_type='standard'
)

```


### Training Size Curves
{python}
```
import mlcurves as mc
import tensorflow_datasets as tfds

# For tensorflow datasets
ds, ts = tfds.load('mnist', split=['train', 'test'], shuffle_files=True)

ds = ds.map(lambda x: (tf.reshape(x['image'], [-1]), x['label']))
ts = ts.map(lambda x: (tf.reshape(x['image'], [-1]), x['label']))

batch_size=32
n_runs=11
epochs=10

val_size = 1500
trainset = ds.skip(val_size)
valset = ds.take(val_size)
testset = ts

generate_train_set_size_curves_tf(
    model_fn=build_antirectifier_model,
    trainset=trainset, valset=valset, testset=testset, 
    batch_size=batch_size, n_runs=n_runs, epochs=epochs
)



# For numpy arrays
trainXy = np.asarray(list(ds.as_numpy_iterator()))
testXy = np.asarray(list(ts.as_numpy_iterator()))
X = np.concatenate([trainXy, testXy])
y = X[:, 1]
X = X[:, 0]

generate_train_set_size_curves_npy(
    model_fn=build_antirectifier_model, # model function (callable)
    X=X, y=y, # data sources
    epochs=epochs, 
    batch_size=batch_size, 
    n_runs=n_runs # number of runs to perform (slices data int `n_run` chunks of increasing size)
)
```


### Complexity Curves

{python}

```
...

history = complexity_curves_npy(model_fn, # model function (callable)
                                X=X, y=y, # data sources 
                                configs,
                                epochs=10,
                                batch_size=16,
                                input_shape=None,
                                num_classes=None,
                                outpath='./plot')
```

