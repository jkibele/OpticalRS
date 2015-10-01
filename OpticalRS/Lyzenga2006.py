# -*- coding: utf-8 -*-
"""
This module is a part of the OpticalRS library. It implements methods described
in Lyzenga et al. 2006. This implementation is the work of the author of this
code, not the authors of the original paper. I tried to get it right but I'm 
not making any promises. Please check your results and let me know if you find
any problems.

Lyzenga, D.R., Malinas, N.P., Tanis, F.J., 2006. Multispectral bathymetry using
a simple physically based algorithm. Geoscience and Remote Sensing, IEEE 
Transactions on 44, 2251 –2259. doi:10.1109/TGRS.2006.872909


I developed this stuff in ClassificationDev/Lyzenga/Lyzenga2006/DeepWaterMasking.ipynb
@author: Jared Kibele
"""

import numpy as np
from skimage.filters import rank
from skimage import morphology

## Deep water masking --------------------------------------------------------

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
    dark_pix : boolean array
        True where pixels are <= `p` percentile of brightness and False for all
        other pixels.

    """
    # average DNs to get (RxCx1) brightness
    brt = imarr.mean(axis=2)
    dim_thresh = np.percentile( brt.compressed(), p )
    dark_pix = ( brt <= dim_thresh )
    if np.ma.is_masked( dark_pix ):
        dark_pix.set_fill_value( False )
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
    outarr : array
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

    Parameters
    ----------
    imarr :
        
    p :
         (Default value = 10)
    win_size :
         (Default value = 3)
    win_percentage :
         (Default value = 50)

    Returns
    -------

    """
    dps = dark_pixels( imarr, p=p )
    if win_size:
        dmeans = moving_window( dps, win_size=win_size )
        dps = dmeans >= (win_percentage/100.0)
    return dps.astype('bool')
    
def dark_pixel_array( imarr, p=10, win_size=3, win_percentage=50 ):
    """

    Parameters
    ----------
    imarr :
        
    p :
         (Default value = 10)
    win_size :
         (Default value = 3)
    win_percentage :
         (Default value = 50)

    Returns
    -------

    """
    dp = dark_kernels( imarr, p, win_size, win_percentage )
    dparr = imarr.copy()
    dparr.mask = ~np.repeat( np.expand_dims( dp.filled(), 2 ), 8, 2 )
    return dparr   
    
def bg_thresholds( dark_arr, n_std=3 ):
    """

    Parameters
    ----------
    dark_arr :
        
    n_std :
         (Default value = 3)

    Returns
    -------

    """
    nbands = dark_arr.shape[-1]
    darkmeans = dark_arr.reshape(-1,nbands).mean(0).data
    darkstds = dark_arr.reshape(-1,nbands).std(0).data
    return darkmeans + n_std * darkstds

## Glint correction ----------------------------------------------------------

def nir_mean(msarr,nir_band=7):
    """

    Parameters
    ----------
    msarr :
        
    nir_band :
         (Default value = 7)

    Returns
    -------

    """
    return msarr[...,nir_band].mean()
    
def cov_ratio(msarr,band,nir_band=7):
    """

    Parameters
    ----------
    msarr :
        
    band :
        
    nir_band :
         (Default value = 7)

    Returns
    -------

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

    Parameters
    ----------
    msarr :
        
    nir_band :
         (Default value = 7)

    Returns
    -------

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

    Parameters
    ----------
    imarr :
        
    glintarr :
        
    nir_band :
         (Default value = 7)

    Returns
    -------

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