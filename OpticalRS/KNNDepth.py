# -*- coding: utf-8 -*-
"""
KNN Depth
=========

Code for the empirical regression of depth from multispectral imagery using the
k nearest neighbors technique (Kibele and Shears, In Review). This method should
typically be accessed throught the `OpticalRS.DepthEstimator` module.

References
----------

Kibele, J., Shears, N.T., In Review. Non-parametric empirical depth regression
for bathymetric mapping in coastal waters. IEEE Journal of Selected Topics in
Applied Earth Observations and Remote Sensing.

"""

from sklearn.neighbors import KNeighborsRegressor
from ArrayUtils import mask3D_with_2D

def train_model(pixels, depths, **kwargs):
    """
    Return a KNN Regression model trained on the given pixels for the given
    depths.

    Notes
    -----
    This is just a thin wrapper over the KNeighborsRegressor model in scikit-learn.
    For more info: http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html#sklearn.neighbors.KNeighborsRegressor

    Parameters
    ----------
    pixels : array-like
        These are the spectral values of the pixels being used to train the
        model. The shape of the array should be (n_pixels,n_bands). The number
        of pixels (n_pixels) needs to be the same as the number of `depths` and
        they must correspond.
    depths : array-like
        The measured depths that correspond to each pixel in the `pixels` array.
        The units of measurement shouldn't make any difference. Predictions made
        will, of course, be in the same units as the training depths.

    Keyword Arguments
    -----------------
    k : int, optional
        The number of neighbors used by the KNN algorithm.
    weights : str or callable, optional
        Weight function used in prediction. Possible values:
        * 'uniform' : All points in each 'hood are weighted equally (default)
        * 'distance': Weight points by inverse distance so closer neighbors have
            more influence.
        * [callable]: A user-defined function wich accepts an array of distances
            and returns an array of the same shape containing the weights.
    See http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html
        for additional keyword arguments.

    Returns
    -------
    sklearn.neighbors.KNeighborsRegressor
        A trained KNNRegression model. See the link in the Notes section for
        more information.
    """
    k = kwargs.pop('k',5)
    knn = KNeighborsRegressor(n_neighbors=k, **kwargs)
    return knn.fit(pixels,depths)

## This bit isn't really required anymore and it's incomplete anyway. The best
## way to access this depth estimation method is through the DepthEstimator
## module.

# def model_from_imarr(imarr,depths,k=5,weights='uniform'):
#      """
#      Build a trained KNN depth model from an image array and a depth array. The
#      shape of the depth array is assumed to be the same as the shape of a single
#      band of the image array. The `depths` are assumed to be a subset of the
#      pixels in `imarr` for which the depths are known.
#
#      Notes
#      -----
#      This is just a wrapper over the KNeighborsRegressor model in scikit-learn.
#      For more info (particularly on the `k` and `weights` params) see:
#      http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html#sklearn.neighbors.KNeighborsRegressor
#
#      Parameters
#      ----------
#      imarr : np.ma.MaskedArray
#          The image array of shape (Rows,Cols,nBands).
#      depths : np.ma.MaskedArray
#          The measured depths that correspond to each pixel with known depth in
#          the `pixels` array. The shape should be (Rows,Cols) (with the same Rows
#          and Cols numbers as in `imarr`. This should be a subset of the unmasked
#          pixels in `imarr`. The units of measurement shouldn't make any
#          difference. Predictions made will, of course, be in the same units as
#          the training depths.
#      k : int, optional
#          The number of neighbors used by the KNN algorithm.
#      weights : str or callable, optional
#          Weight function used in prediction. Possible values:
#          * 'uniform' : All points in each 'hood are weighted equally (default)
#          * 'distance': Weight points by inverse distance so closer neighbors have
#              more influence.
#          * [callable]: A user-defined function wich accepts an array of distances
#              and returns an array of the same shape containing the weights.
#
#      Returns
#      -------
#      sklearn.neighbors.KNeighborsRegressor
#          A trained KNNRegression model. See the link in the Notes section for
#          more information.
#      """
#      pix = mask3D_with_2D(imarr,depths.mask)
