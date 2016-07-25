# -*- coding: utf-8 -*-
"""
Lyzenga2006
===========

This module implements methods described in Lyzenga et al. 2006. The methods
implemented so far are mostly the image preprocessing steps.

This implementation is the work of the author of this code (Jared Kibele), not
the authors of the original paper. I tried to get it  right but I'm not making
any promises. Please check your results and let me know if you find any
problems.

References
----------
Lyzenga, D.R., Malinas, N.P., Tanis, F.J., 2006. Multispectral bathymetry using
a simple physically based algorithm. Geoscience and Remote Sensing, IEEE
Transactions on 44, 2251 –2259. doi:10.1109/TGRS.2006.872909

Armstrong, R.A., 1993. Remote sensing of submerged vegetation canopies for
biomass estimation. International Journal of Remote Sensing 14, 621–627.
doi:10.1080/01431169308904363

Ji, W., Civco, D., Kennard, W., 1992. Satellite remote bathymetry: a new
mechanism for modeling. Photogrammetric Engineering and Remote Sensing 58,
545–549.


Notes
-----
I developed this in ClassificationDev/Lyzenga/Lyzenga2006/DeepWaterMasking.ipynb

"""

import numpy as np
from skimage.filters import rank
from skimage import morphology
from sklearn.linear_model import LinearRegression
import itertools
from collections import OrderedDict

def get_fit( ind, x_train, y_train, n_jobs=4 ):
    """
    Get a linear regression fit object from Scikit-learn for eq. 9 from Lyzenga
    et al. 2006.

    Parameters
    ----------
    ind : list of integers
        The zero-indexed band numbers to use in the linear regression.
    x_train : np.ma.MaskedArray
        The image array. If you're following Lyzenga et al. 2006, these will be
        radiance values transformed according to Lyzenga 1978 eq. 7 (see Lyzenga
        et al. 2006 eq. 8).
    y_train : np.array or MaskedArray
        The measured depths that correspond to the pixels in `x_train`. This
        array must have the same dimensions as a single band of `x_train`.

    Returns
    -------
    fit object
        For information on this object see: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
    """
    if x_train.ndim > 2:
        nbands = x_train.shape[-1]
        x_train = x_train.compressed().reshape(-1,nbands)
    skols = LinearRegression(n_jobs=n_jobs)
    skolsfit = skols.fit(x_train[...,ind],y_train.compressed())
    return skolsfit

def get_selfscore( ind, x_train, y_train, n_jobs=4 ):
    """
    Get the r^2 value from linear regression fit object from Scikit-learn for
    eq. 9 from Lyzenga et al. 2006.

    Parameters
    ----------
    ind : list of integers
        The zero-indexed band numbers to use in the linear regression.
    x_train : np.ma.MaskedArray
        The image array. If you're following Lyzenga et al. 2006, these will be
        radiance values transformed according to Lyzenga 1978 eq. 7 (see Lyzenga
        et al. 2006 eq. 8).
    y_train : np.array or MaskedArray
        The measured depths that correspond to the pixels in `x_train`. This
        array must have the same dimensions as a single band of `x_train`.

    Returns
    -------
    float
        The r^2 value for the linear regression.
    """
    if x_train.ndim > 2:
        nbands = x_train.shape[-1]
        x_train = x_train.compressed().reshape(-1,nbands)
    fit = get_fit( ind, x_train, y_train, n_jobs=n_jobs )
    return fit.score( x_train[...,ind], y_train.compressed() )

def ranked_combos(x_img,y_depths,n=2):
    """
    Rank all possible combinations of `n` bands from `x_img` based on r^2 values
    from eq. 9 (Lyzenga et al. 2006).

    Parameters
    ----------
    x_img : np.array or MaskedArray
        The image array. If you're following Lyzenga et al. 2006, these will be
        radiance values transformed according to Lyzenga 1978 eq. 7 (see Lyzenga
        et al. 2006 eq. 8).
    y_depths : np.array or MaskedArray
        The measured depths that correspond to the pixels in `x_img`. This array
        must have the same dimensions as a single band of `x_img`.
    n : int, optional
        The number of bands to use in combos. This must not exceed the number of
        bands in `x_img`.

    Returns
    -------
    OrderedDict
        The dict keys are the r^2 values and the dict values are the zero
        indexed band combinations in list format. The best band combo (the one
        with the highest r^2) will be the first item in the dict.
    """
    od = OrderedDict()
    nbands = x_img.shape[-1]
    for comb in itertools.combinations( range(nbands), n ):
        od[ get_selfscore(comb,x_img,y_depths) ] = comb
    od_sort = sorted( od.items(), key=lambda t: t[0], reverse=True )
    return OrderedDict(od_sort)

def best_combo(x_img,y_depths,n=2):
    """
    Evaluate all combinations of `n` bands from `x_img` for regression against
    `y_depths` and return the best band combo as a tuple of zero-indexed band
    numbers. "Best" is determined by choosing the band combo with the greatest
    r^2 value.

    Parameters
    ----------
    x_img : np.array or MaskedArray
        The image array. If you're following Lyzenga et al. 2006, these will be
        radiance values transformed according to Lyzenga 1978 eq. 7 (see Lyzenga
        et al. 2006 eq. 8).
    y_depths : np.array or MaskedArray
        The measured depths that correspond to the pixels in `x_img`. This array
        must have the same dimensions as a single band of `x_img`.
    n : int, optional
        The number of bands to use in combos. This must not exceed the number of
        bands in `x_img`.

    Returns
    -------
    tuple of ints
        This will be the best band combo to use for depth regression. The length
        of this tuple will be `n`.
    """
    return ranked_combos(x_img,y_depths,n).values()[0]

def tuned_linear_model(x_img,y_depths,n=2,n_jobs=4):
    """
    Find the best combo of `n` bands from `x_img` and return a model tuned to
    the training data (`x_img` and `y_depths`).

    Parameters
    ----------
    x_img : np.array or MaskedArray
        The image array. If you're following Lyzenga et al. 2006, these will be
        radiance values transformed according to Lyzenga 1978 eq. 7 (see Lyzenga
        et al. 2006 eq. 8).
    y_depths : np.array or MaskedArray
        The measured depths that correspond to the pixels in `x_img`. This array
        must have the same dimensions as a single band of `x_img`.
    n : int, optional
        The number of bands to use in the estimation model. Must not exceed the
        number of bands in `x_img`.

    Returns
    -------
    fit object
        For information on this object see:
        http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
    """
    ind = best_combo(x_img,y_depths,n)
    return get_fit(ind,x_img,y_depths,n_jobs=n_jobs)

def fit_and_predict(x_train,y_train,x_img,n=2,n_jobs=4):
    """
    Build Lyzenga depth estimation model trained on `x_train` (transformed
    imagery) and `y_train` (depths) and use it to estimate depths for `x_img`.

    Parameters
    ----------
    x_train : np.ma.MaskedArray
        The image array subset to be used for training the model. If you're
        following Lyzenga et al. 2006, these will be radiance values transformed
        according to Lyzenga 1978 eq. 7 (see Lyzenga et al. 2006 eq. 8).
    y_train : np.array or MaskedArray
        The measured depths that correspond to the pixels in `x_train`. This
        array must have the same dimensions as a single band of `x_train`.
    x_img : np.array or MaskedArray
        The image array for which you want depth estimates. Must have the same
        dimensions and preprocessing as `x_train`.
    n : int, optional
        The number of bands to use in the estimation model. Must not exceed the
        number of bands in `x_img`.

    Returns
    -------
    np.array
        Predicted depths for `x_img`.
    """
    outarr = x_img[...,0].copy()
    ind = best_combo(x_train,y_train,n)
    modl = get_fit(ind,x_train,y_train,n_jobs=n_jobs)
    pred = modl.predict(x_img[...,ind].compressed().reshape(-1,n))
    outarr[~outarr.mask] = pred
    return outarr

## Deep water masking --------------------------------------------------------
#   These methods are implementations of the preprocessing steps in section V
#   of Lyzenga et al. 2006.

def dark_pixels( imarr, p=10 ):
    """
    Return a single band boolean array with all pixels <= the `p` percentile of
    brightness (across all bands) marked as `True`. All other pixels marked as
    `False`. This method was developed to carry out the following step from
    section V of Lyzenga et al. 2006:

    "A first estimate of the mean deep-water radiance is calculated by first
    identifying the tenth-percentile brightness magnitude (using all four
    bands) within the water area of interest."

    Parameters
    ----------
    imarr : numpy array (RxCxBands shape)
        The multispectral image array. See `OpticalRS.RasterDS` for more info.
    p : int or float (Default value = 10)
        The percentile of brightness to use as the threshold for declaring a
        pixel 'dark'. Lyzenga et al. 2006 used the 10th percnetile so that's
        the default.

    Returns
    -------
    boolean array
        True where pixels are <= `p` percentile of brightness and False for all
        other pixels.

    """
    # average DNs to get (RxCx1) brightness
    brt = imarr.mean(axis=2)
    dim_thresh = np.percentile( brt.compressed(), p )
    dark_pix = ( brt <= dim_thresh )
    if np.ma.is_masked( dark_pix ):
        dark_pix.set_fill_value( False )
        dark_pix = dark_pix.filled()
    return dark_pix

def moving_window( dark_arr, win_size=3 ):
    """
    Find average value of pixels in a square window. Used on a boolean array,
    this can be used to find the percentage of pixels marked as `True`.

    Parameters
    ----------
    dark_arr : boolean array
        This is an (RxCx1) array of pixels that are considered dark. The
        `dark_pixels` method in this module can be used to create this array.
    win_size : int (Default value = 3)
        The size of the moving window to be used. Lyzenga et al. 2006 uses a
        3x3 window so the default is 3.

    Returns
    -------
    array
        An array the same shape as `dark_arr` with values representing the
        proportion of pixels in the surrounding window that are `True` in
        `dark_arr`.
    """
    win = morphology.square( win_size )
    npix = win.size
    if np.ma.is_masked( dark_arr ):
        outarr = rank.sum( dark_arr.filled().astype('uint8'), win ) / float( npix )
        outarr = np.ma.MaskedArray( outarr, mask=dark_arr.mask, fill_value=dark_arr.fill_value )
    else:
        outarr = rank.sum( dark_arr.astype('uint8'), win ) / float( npix )
    return outarr

def dark_kernels( imarr, p=10, win_size=3, win_percentage=50 ):
    """
    Return a single band boolean image array where pixels are `True` if at
    least 50% of the surrounding pixels are at or below the `p` percentile of
    image brightness. This is an implementation of the following sentence from
    Lyzenga et al. 2006:

    "A moving window is then passed through the image to identify kernels that
    contain more than 50% of pixels at or below the 10% brightness threshold."

    Parameters
    ----------
    imarr : numpy array (RxCxBands shape)
        The multispectral image array. See `OpticalRS.RasterDS` for more info.
    p : int or float (Default value = 10)
        The percentile of brightness to use as the threshold for declaring a
        pixel 'dark'. Lyzenga et al. 2006 used the 10th percnetile so that's
        the default.
    win_size : int (Default value = 3)
        The size of the moving window to be used. Lyzenga et al. 2006 uses a
        3x3 window so the default is 3.
    win_percentage : int or float (Default value = 50)
        The percentage of the moving window that must be at or below the
        threshold. Lyzenga et al. 2006 used 50% so that's the default.

    Returns
    -------
    numpy boolean array
        An (RxC) shaped boolean array. `True` values are "dark kernels".
    """
    dps = dark_pixels( imarr, p=p )
    if win_size:
        dmeans = moving_window( dps, win_size=win_size )
        dps = dmeans >= (win_percentage/100.0)
    return dps.astype('bool')

def dark_pixel_array( imarr, p=10, win_size=3, win_percentage=50 ):
    """
    Return a masked version of imarr where only the "dark kernels" described in
    section V of Lyzenga et al. 2006 are left unmasked.

    Parameters
    ----------
    imarr : numpy array (RxCxBands shape)
        The multispectral image array. See `OpticalRS.RasterDS` for more info.
    p : int or float (Default value = 10)
        The percentile of brightness to use as the threshold for declaring a
        pixel 'dark'. Lyzenga et al. 2006 used the 10th percnetile so that's
        the default.
    win_size : int (Default value = 3)
        The size of the moving window to be used. Lyzenga et al. 2006 uses a
        3x3 window so the default is 3.
    win_percentage : int or float (Default value = 50)
        The percentage of the moving window that must be at or below the
        threshold. Lyzenga et al. 2006 used 50% so that's the default.

    Returns
    -------
    numpy MaskedArray
        A masked version of imarr where only the "dark kernels" described in
        section V of Lyzenga et al. 2006 are left unmasked.

    """
    nbands = imarr.shape[-1]
    dp = dark_kernels( imarr, p, win_size, win_percentage )
    dparr = imarr.copy()
    dparr.mask = ~np.repeat( np.atleast_3d(dp), nbands, 2 )
    return dparr

def deep_water_means(imarr, n_std=0, p=10, win_size=3, win_percentage=50):
    """
    Return deep water mean values for each band of `imarr`.

    Parameters
    ----------
    imarr : numpy array (RxCxBands shape)
        The multispectral image array. See `OpticalRS.RasterDS` for more info.
    n_std : int or float
        The number of standard deviations to subtract from the deep water means
        before returning them. This can get you around the 'over deduction'
        problem described by Ji et al. 1992. See Armstrong 1993 for an example
        of this (though he attributes it to sensor noise).
    p : int or float (Default value = 10)
        The percentile of brightness to use as the threshold for declaring a
        pixel 'dark'. Lyzenga et al. 2006 used the 10th percnetile so that's
        the default.
    win_size : int (Default value = 3)
        The size of the moving window to be used. Lyzenga et al. 2006 uses a
        3x3 window so the default is 3.
    win_percentage : int or float (Default value = 50)
        The percentage of the moving window that must be at or below the
        threshold. Lyzenga et al. 2006 used 50% so that's the default.
    """
    nbands = imarr.shape[-1]
    dpa = dark_pixel_array(imarr, p=p, win_size=win_size,
                           win_percentage=win_percentage)
    dpaavg = dpa.reshape(-1, nbands).mean(0).data
    dpastd = dpa.reshape(-1, nbands).std(0).data
    Rinf = dpaavg - (n_std * dpastd)
    return Rinf

def bg_thresholds( dark_arr, n_std=3 ):
    """
    Calculate band-wise mean radiance plus 3 standard deviations for pixels in
    `dark_arr`. Lyzenga et al. 2006 says:

    "...the blue and green bands are thresholded at the deep-water mean
    radiance plus three standard deviations."

    This method will calculate the mean + 3 std for all bands. You'll have to
    pick out the blue and green ones later if that's what you're after.

    Parameters
    ----------
    dark_arr : numpy Masked Array
        Typically, this will be the output of `Lyzenga2006.dark_pixels_array`.
    n_std : int (Default value = 3)
        The number of standard deviations to add to the mean. Lyzenga et al.
        2006 uses 3 so that's the default.

    Returns
    -------
    numpy array
        A 1D array with as many elements as there are bands in `dark_arr`. Each
        element corresponds to the threshold for its respective band.
    """
    nbands = dark_arr.shape[-1]
    darkmeans = dark_arr.reshape(-1,nbands).mean(0).data
    darkstds = dark_arr.reshape(-1,nbands).std(0).data
    return darkmeans + n_std * darkstds

## Glint correction ----------------------------------------------------------
#   These methods are derived from section III of Lyzenga et al. 2006.

def nir_mean(msarr,nir_band=7):
    """
    Calculate the mean of the (unmasked) values of the NIR (near infrared) band
    of an image array. The default `nir_band` value of 7 selects the NIR2 band
    in WorldView-2 imagery. If you're working with a different type of imagery,
    you will need figure out the appropriate value to use instead.

    Parameters
    ----------
    msarr : numpy array (RxCxBands shape)
        The multispectral image array. See `OpticalRS.RasterDS` for more info.
    nir_band : int (Default value = 7)
        The default `nir_band` value of 7 selects the NIR2 band in WorldView-2
        imagery. If you're working with a different type of imagery, you will
        need figure out the appropriate value to use instead. This is a zero
        indexed number (the first band is 0, not 1).

    Returns
    -------
    float
        The mean radiance in the NIR band.
    """
    return msarr[...,nir_band].mean()

def cov_ratio(msarr,band,nir_band=7):
    """
    Calculate the r_ij value according to equation 5 from Lyzenga et al. 2006.

    Parameters
    ----------
    msarr : numpy array (RxCxBands shape)
        The multispectral image array. See `OpticalRS.RasterDS` for more info.
    band : int
        The band to calculate r_ij for. Essentially, the value i in equation 5.
    nir_band : int (Default value = 7)
        The default `nir_band` value of 7 selects the NIR2 band in WorldView-2
        imagery. If you're working with a different type of imagery, you will
        need figure out the appropriate value to use instead. This is a zero
        indexed number (the first band is 0, not 1).

    Returns
    -------
    float
        The covariance ratio r_ij described in equation 5 of Lyzenga et al.
        2006.
    """
    if np.ma.is_masked( msarr ):
        b = msarr[...,band].compressed()
        nir_b = msarr[...,nir_band].compressed()
    else:
        b = msarr[...,band].flatten()
        nir_b = msarr[...,nir_band].flatten()
    cov_mat = np.cov( b, nir_b, bias=1 )
    return cov_mat[0,1] / cov_mat[1,1]

def cov_ratios(msarr,nir_band=7):
    """
    Calculate the r_ij value according to equation 5 from Lyzenga et al. 2006
    for each band of an image array.

    Parameters
    ----------
    msarr : numpy array (RxCxBands shape)
        The multispectral image array. See `OpticalRS.RasterDS` for more info.
    nir_band : int (Default value = 7)
        The default `nir_band` value of 7 selects the NIR2 band in WorldView-2
        imagery. If you're working with a different type of imagery, you will
        need figure out the appropriate value to use instead. This is a zero
        indexed number (the first band is 0, not 1).

    Returns
    -------
    numpy array
        An array of r_ij values calculated by `OpticalRS.Lyzenga2006.cov_ratio`.
        One r_ij value for each band of `msarr`.
    """
    nbands = msarr.shape[-1] #assume Rows,Cols,Bands shape
    bands = range(nbands)
    cov_rats = []
    if nir_band in bands: bands.remove(nir_band)
    for band in bands:
        cov_rat = cov_ratio(msarr,band,nir_band)
        cov_rats.append(cov_rat)
    return np.array(cov_rats)

def glint_correct_image(imarr, glintarr, nir_band=7):
    """
    Apply the sunglint removal algorithm from section III of Lyzenga et al.
    2006 to a multispectral image array.

    Parameters
    ----------
    imarr : numpy array (RxCxBands shape)
        The multispectral image array. See `OpticalRS.RasterDS` for more info.
    glintarr : numpy array
        A subset of `imarr` from an optically deep location with sun glint.
    nir_band : int (Default value = 7)
        The default `nir_band` value of 7 selects the NIR2 band in WorldView-2
        imagery. If you're working with a different type of imagery, you will
        need figure out the appropriate value to use instead. This is a zero
        indexed number (the first band is 0, not 1).

    Returns
    -------
    numpy array
        A de-glinted copy of `imarr`.

    Notes
    -----
    This deglinting method may not work well on WorldView-2 imagery because the
    bands are not captured exactly concurrently. See section II B of Eugenio et
    al. 2015 [1]_ for more information and a different sunglint correction
    algorithm that may be more appropriate.

    References
    ----------
    .. [1] Eugenio, F., Marcello, J., Martin, J., 2015. High-Resolution Maps of
       Bathymetry and Benthic Habitats in Shallow-Water Environments Using
       Multispectral Remote Sensing Imagery. IEEE Transactions on Geoscience
       and Remote Sensing 53, 3539–3549. doi:10.1109/TGRS.2014.2377300
    """
    # calculate the covariance ratios
    cov_rats = cov_ratios(glintarr,nir_band)
    # get the NIR mean
    nirm = nir_mean(glintarr,nir_band)
    # we don't want to try to apply the correction
    # to the NIR band
    nbands = imarr.shape[-1]
    bands = range(nbands)
    bands.remove(nir_band)
    outarr = imarr.copy()
    for i,band in enumerate(bands):
        outarr[:,:,band] = imarr[:,:,band] - cov_rats[i] * ( imarr[:,:,nir_band] - nirm )
    # this will leave the NIR band unchanged
    return outarr
