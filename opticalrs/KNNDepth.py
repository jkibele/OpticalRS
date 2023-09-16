# -*- coding: utf-8 -*-
"""
KNN Depth
=========

Code for the empirical regression of depth from multispectral imagery using the
k nearest neighbors technique (Kibele and Shears, In Review). This method should
typically be accessed throught the `OpticalRS.DepthEstimator` module.

References
----------

Kibele, J., & Shears, N. T. (2016). Nonparametric Empirical Depth Regression for
Bathymetric Mapping in Coastal Waters. IEEE Journal of Selected Topics in
Applied Earth Observations and Remote Sensing, 1–9.
https://doi.org/10.1109/JSTARS.2016.2598152

"""

from sklearn.neighbors import KNeighborsRegressor

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
