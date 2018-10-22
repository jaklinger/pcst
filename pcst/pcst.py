"""
Principal Component Segmentation Tree
=====================================

Performant segmentation of high-dimensional
spaces, with a specific applicability to
word-embedding spaces. Typical run-time on
the pretrained FastText Wikipedia word embedding
is 40 minutes, using 2 cores and 5 GB RAM.

The algorithm applies PCA (specifically, the fast
'randomised' SVD implementation) to the full
dataset, along which a single segmentation is
imposed at the point which minimises the ratio
of the sum of variances (on either side of the
segmentation) with respect to the variance before
segmentation. The optimisation is solved
using Gaussian processes. If no segmentation is favourable
then the algorithm is stopped.

After segmentation, the algorithm is repeated iteratively
to each subsequent pair of branches.
"""

from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from GPyOpt.methods import BayesianOptimization as bopt
import GPy


def load_minimal_wv(gensim_loader, *gensim_args, **gensim_kwargs):
    '''Helper method for loading gensim models and stripping out all of
    the unnecessary overheads.

    Args:
        gensim_loader: one of either
                       :obj:`gensim.models.FastText.load_fasttext_format`
                       or :obj:`gensim.models.Word2Vec.load_word2vec_format`.
        gensim_args / gensim_kwargs: Arguments for :obj:`gensim_loader`.

    Returns:
        :obj:`vectors`, :obj:`words`: Ordered arrays of word-embedding space
                                      vectors and words, respectively.
    '''
    model = gensim_loader(*gensim_args, **gensim_kwargs)
    model.wv.init_sims(replace=True)
    vectors = model.wv.vectors
    words = np.array(model.wv.index2word)
    del model
    return vectors, words


class PrincipalComponentSegmentation:
    """Segmentation of data along it's principal component,
    optimised using Gaussian Processes. For word embedding segmentation,
    the only parameters you should consider tuning are :obj:`num_cores`,
    which should be self-explanatory, and :obj:`iterated_power` which you
    can either increase for marginal improvements in PCA performance, or
    decrease for small but reasonable improvements in running time.

    Args:
        data (:obj:`np.array`): The data on which to apply segmentation.
        kernel (:obj:`GPy.kern`): If :obj:`None` is provided then
                                  :obj:`GPy.kern.Matern32` is used.
        initial_design_numdata (int): See :obj:`GPyOpt` documentation.
        exact_feval (int): See :obj:`GPyOpt` documentation.
        optimize_restarts (int): See :obj:`GPyOpt` documentation.
        num_cores (int): See :obj:`GPyOpt` documentation.
        iterated_power (int): See :obj:`sklearn.decomposition.PCA`.
    """
    def __init__(self, data, kernel=None,
                 initial_design_numdata=10,
                 exact_feval=True,
                 optimize_restarts=10, num_cores=1,
                 iterated_power=3):

        self.pca = PCA(n_components=1, copy=False,
                       iterated_power=iterated_power)
        self._data = data
        self.var = np.var(data)
        self.data = None
        self.opt = None

        # Generate the optimisation framework parameters
        if kernel is None:
            kernel = GPy.kern.Matern32(1)
        self.opt_kwargs = dict(f=self.evaluate_score,
                               kernel=kernel,
                               initial_design_numdata=initial_design_numdata,
                               exact_feval=exact_feval,
                               initial_design_type='latin',
                               optimize_restarts=optimize_restarts,
                               num_cores=num_cores)

    def evaluate_score(self, *args):
        """The objective function, defined as the
        the ratio of the sum of variances (on either side of the
        segmentation) with respect to the variance before segmentation)
        """
        cut = args[0][0][0]
        below = self.data <= cut
        above = self.data > cut
        if below.sum() == 0 or above.sum() == 0:
            return 1
        var1 = np.var(self.data[below])
        var2 = np.var(self.data[above])
        return (var1 + var2)/self.var

    def apply_segmentation(self, **kwargs):
        """Apply PCA to the data, then find the optimal segmentation point."""
        self.data = self.pca.fit_transform(self._data)
        self.opt = bopt(domain=[{'name': 'segmentation_cut',
                                 'type': 'continuous',
                                 'domain': (min(self.data), max(self.data))}],
                        **self.opt_kwargs)
        self.opt.run_optimization(**kwargs)
        below = self.data <= self.opt.x_opt[0]
        above = self.data > self.opt.x_opt[0]
        left, right = self._data[below.flatten()], self._data[above.flatten()]
        del self._data
        return left, right

    def plot(self):
        """Plot the segmentation point on top of the histogrammed data."""
        fig, ax = plt.subplots(figsize=(6, 3))
        n, _, _ = ax.hist(self.data, bins=500)
        ax.vlines(self.opt.x_opt[0], ymin=0, ymax=max(n))


class PrincipalComponentSegmentationTree:
    """A self-assembling tree structure based on the Principal Component
    Segmentation algorithm.

    Args:
        max_iter (int): Maximum number of iterations of the Bayesian
                        optimisation. Note that this the practical number of
                        of iterations is typically very small, since
                        convergence occurs very quickly.
        eps (int): Minimum distance between success sampling points in the
                   optimisation.
        min_leaf_size (int): The minimum cluster size that will be considered.
                             The default leaf size implies a maximum
                             of 1000 clusters.
        verbose (bool): Print progress statements.
        plot (bool): Plot segmentation histograms (note that for large trees
                     this can be memory intensive.
        level (int): Don't touch this (used for pretty printing).
        **pcst_kwargs: See :obj:`PrincipalComponentSegmentation`.
                       Note that the :obj:`data` parameter is mandatory.
    """

    def __init__(self, max_iter=10, eps=1e-3,
                 min_leaf_size=None, verbose=False,
                 plot=False, level=0, **pcst_kwargs):

        if 'data' not in pcst_kwargs:
            raise ValueError("Expected to find 'data' in 'pcst_kwargs'.")

        # Default min leaf size implies maximum of 1000 clusters
        if min_leaf_size is None:
            min_leaf_size = int(np.ceil(len(pcst_kwargs['data'])/1000))

        # Perform the splitting
        pcc = PrincipalComponentSegmentation(**pcst_kwargs)
        left, right = pcc.apply_segmentation(max_iter=max_iter, eps=eps)
        if plot:
            pcc.plot()

        # Assign the cut to the node
        self.cut = (pcc.data <= pcc.opt.x_opt[0]).flatten()
        if verbose:
            print(f"{level*'> > '} left, right = "
                  f"{self.cut.sum()} | {(~self.cut).sum()}")
        del pcc

        # If the cut is at the extremes of the distribution, stop
        # (This can imply that the distribution is approximately Gaussian)
        if self.cut.sum() == 0 or (~self.cut).sum() == 0:
            return

        # If the min leaf size has been reached
        if self.cut.sum() < min_leaf_size or (~self.cut).sum() < min_leaf_size:
            return

        # Otherwise, keep on splitting
        pcst_kwargs.pop("data")
        kwargs = dict(max_iter=max_iter, eps=eps,
                      min_leaf_size=min_leaf_size,
                      verbose=verbose, plot=plot,
                      level=level+1, **pcst_kwargs)
        self.left = PrincipalComponentSegmentationTree(data=left, **kwargs)
        self.right = PrincipalComponentSegmentationTree(data=right, **kwargs)

    def generate_clusters(self, labels):
        '''Recursively extract the clusters from the tree.

        Args:
            labels (:obj:`np.array`): Labels of the data which have been
                                      segmented. This could be indexes
                                      or (in the case of
                                      word embeddings) a list of words
                                      corresponding to each data point.
        Returns:
            A :obj:`list` of :obj:`set` objects.
        '''
        # Check whether children exist
        has_right = hasattr(self, 'right')
        has_left = hasattr(self, 'left')
        if not (has_right and has_left):
            return [set(labels)]
        # If so, recurse with the subset given by self.cut
        right = self.right.generate_clusters(labels[~self.cut])
        left = self.left.generate_clusters(labels[self.cut])
        return right + left


if __name__ == "__main__":
    from gensim.models import FastText as ft
    vectors, words = load_minimal_wv(ft.load_fasttext_format,
                                     "/Users/jklinger/Downloads/wiki.en.bin")
    pcst = PrincipalComponentSegmentationTree(data=vectors, verbose=True,
                                              min_leaf_size=500, num_cores=2)
    clusters = pcst.generate_clusters(words)

    for i, cluster in enumerate(clusters):
        with open(f'data/syn-{i}.csv', "w") as f:
            for word in cluster:
                f.write(f"{word}\n")
