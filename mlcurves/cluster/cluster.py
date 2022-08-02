import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from functools import reduce

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer


# REFERENCES #
# https://arxiv.org/abs/1712.09005
# https://distill.pub/2016/misread-tsne/
# https://stats.stackexchange.com/questions/263539/clustering-on-the-output-of-t-sne

product = lambda x: reduce(lambda a,b: a*b, x)


def correlation_heatmap(df, cols=None, mask_upper=True, show=True, light_cmap=False, lw=0.5):
    #
    # Correlation between different variables
    #
    df = df[cols] if cols is not None else df
    corr = df.corr()
    #
    # Set up the matplotlib plot configuration
    #
    fig, ax = plt.subplots(figsize=(12, 10))
    #
    # Generate a mask for upper traingle
    #
    mask = np.triu(np.ones_like(corr, dtype=bool)) if mask_upper else None
    #
    # Configure a custom diverging colormap
    #
    cmap = "YlGnBu" if light_cmap else sns.diverging_palette(230, 20, as_cmap=True) 
    #
    # Draw the heatmap
    #
    sns.heatmap(corr, annot=True, mask = mask, cmap=cmap, linewidths=lw)
    #
    # Show and return fig,ax
    #
    if show:
        plt.show()

    return fig, ax 


def pca(data, n_components, whiten=False, random_state=None):
    # 
    # Create PCA transform
    # 
    pca = PCA(n_components=n_components, whiten=whiten, random_state=random_state)
    pca_transform = pca.fit_transform(data)
    
    #
    # Show cumulative explained variation (cumsum of component variances)
    #
    print('Cumulative explained variation for {} principal components: {}'.format(
        n_components, np.sum(pca.explained_variance_ratio_))
    )    
    pcaratio = pca.explained_variance_ratio_
    sns.scatterplot(
        x=np.arange(len(pcaratio)), y=np.cumsum(pcaratio), 
        title="PCA Explained Variance"
    )
    plt.show()
    
    return pca_transform


def tsne(x_train, y_train=None, 
         random_state=123, 
         scale=False,
         scale_type='standard',
         perplexity=50,
         n_components=2, 
         verbose=1, 
         n_iter=2000, 
         early_exaggeration=12, 
         n_iter_without_progress=1000,
         title="T-SNE projection"):

    # Ensure data is in form (n_obs, samples)
    # e.g. if x_train.shape == (n_obs, samples_x, samples_y), 
    # Will be reshaped -> (n_obs, samples_x*sampels_y)
    if len(x_train.shape) > 2:
        x_train = np.reshape(
            x_train, [x_train.shape[0]] + [product(x_train.shape[1:])]
        )

    # Use a scaler to standardize data
    if scale:
        scaler_fn = {
            'normal': Normalizer(),
            'minmax': MinMaxScaler(),
            'standard': StandardScaler()
        }[scale_type]
        x_train = scaler_fn.fit_transform(x_train)

    # Create and Run T-SNE with given hparams
    tsne = TSNE(
        n_components=n_components, verbose=verbose, random_state=random_state,
        perplexity=perplexity, early_exaggeration=early_exaggeration, n_iter=n_iter,
        n_iter_without_progress=n_iter_without_progress
    )
    tsne_transform = z = tsne.fit_transform(x_train)

    # Use pandas for plotting convenience
    df = pd.DataFrame()
    if y_train is not None:
        df["y"] = y_train
    df["tsne-1"] = z[:,0]
    df["tsne-2"] = z[:,1]

    # Create scatterplot and show
    sns.scatterplot(x="tsne-1", y="tsne-2", hue=df.y.tolist(),
                    palette=sns.color_palette("hls", 10),
                    data=df).set(title=title); plt.show();
    del df
    return tsne_transform



def pca_then_tsne(x_train, y_train, 
                  n_pca_components=50, 
                  n_tsne_components=2, 
                  whiten=False, 
                  perplexity=50, # 40
                  verbose=1,
                  early_exag=12,
                  n_iter=2000, # 300
                  n_iter_without_progress=1000):

    # Compute PCA and get transform
    pca_transform = pca(x_train, n_components=n_pca_components, whiten=whiten)

    # Compute TSNE and get transform
    tsne_transform = tsne(
        x_train=pca_transform, y_train=y_train,
        n_components=n_tsne_components, verbose=verbose, perplexity=perplexity, 
        n_iter=n_iter, early_exaggeration=early_exag, 
        n_iter_without_progress=n_iter_without_progress
    )

    # Final visualization
    df = pd.DataFrame()
    df['pca-one'] = pca_transform[:,0]
    df['pca-two'] = pca_transform[:,1]
    df['tsne-pca50-one'] = tsne_transform[:,0]
    df['tsne-pca50-two'] = tsne_transform[:,1]

    plt.figure(figsize=(16,4))
    ax1 = plt.subplot(1, 3, 1)
    sns.scatterplot(
        x="pca-one", y="pca-two",
        hue="y",
        palette=sns.color_palette("hls", 10),
        data=df,
        legend="full",
        alpha=0.3,
        ax=ax1
    )
    ax2 = plt.subplot(1, 3, 2)
    sns.scatterplot(
        x="tsne-pca50-one", y="tsne-pca50-two",
        hue="y",
        palette=sns.color_palette("hls", 10),
        data=df,
        legend="full",
        alpha=0.3,
        ax=ax2
    )
    plt.show()

    return tsne_transform



def pca_3D(X, y=None, n_components=3):
    """
    Compute PCA with 3 components and visualize in 3Space
    """
    #
    # Compute PCA transform for 3D axes
    #
    transform = pca(X, n_components=n_components)

    # 
    # Create dataframe for plotular convenience 
    # 
    cols = list(map(lambda x: 'pca-' + str(x), range(1, transform.shape[-1] + 1)))
    df = pd.DataFrame(transform, columns=cols)
    if y is not None:
        df['y'] = y
    
    #
    # Do the plotting
    #
    ax = plt.figure(figsize=(16,10)).gca(projection='3d')
    ax.scatter(
        xs=df["pca-1"], 
        ys=df["pca-2"], 
        zs=df["pca-3"], 
        c=df["y"], cmap='tab10'
    )
    ax.set_xlabel('pca-one')
    ax.set_ylabel('pca-two')
    ax.set_zlabel('pca-three')
    plt.show()

    return transform


def dbscan():
    raise NotImplementedError


def demo():
    from mlcurves.curve_utils import mnist_npy
    X, y = mnist_npy(return_test=False)

    from mlcurves import cluster
    transform_3D = cluster.pca_3D(X, y)
    cluster.tsne()
    cluster.pca_then_tsne()
    cluster.correlation_heatmap()