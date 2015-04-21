# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 19:17:24 2015

@author: jkibele
"""

import numpy as np

## Deep water masking --------------------------------------------------------
def band_percentiles( imarr, p=10 ):
    """
    Return the percentile for each band of an image array of shape (RxCxN)
    where N is the number of bands.
    
    Parameters
    ----------
      imarr : numpy array or masked array
        An 3 dimensional array (RxCxN) where the third dimension (N) is the 
        number of bands. If imarr is masked, then only unmasked values will be
        considered.
        
      p : float in range [0,100] (or sequence of floats)
        Percentile to compute which must be between 0 and 100 inclusive.
        
    Returns
    -------
      percentile : ndarray
        If a single `p` value is given the results will be a 1d array of length
        N, where N is the 3rd dimension of `imarr` (RxCxN). If multiple `p`
        values are given, the results will be of shape N x length(p). If the 
        input contains integers, or floats of smaller precision than 64, then 
        the output data-type is float64. Otherwise, the output data-type is the 
        same as that of the input.
    """
    p_list = []
    n_bands = imarr.shape[-1]
    for i in range( n_bands ):
        if np.ma.is_masked:
            band = imarr[:,:,i].compressed()
        else:
            band = imarr[:,:,i].ravel()
        p_list.append( np.percentile( band, p ) )
    return np.array( p_list )
    
#def deep_pixels( imarr, p=10, bands=None ):
#    """
#    
#    """
#    if bands:
#        arr = imarr[:,:,bands]
#    else:
#        arr = imarr
#    thresh = band_percentiles( arr, p )
#    thresh_pix = ~(arr > thresh).all(axis=2)
#    return thresh_pix
    
def dark_pixels( imarr, p=10 ):
    # average DNs to get (RxCx1) brightness
    brt = imarr.mean(axis=2)
    dim_thresh = np.percentile( brt.compressed(), p )
    dark_pix = ( brt <= dim_thresh )
    if np.ma.is_masked( dark_pix ):
        dark_pix.set_fill_value( False )
    return dark_pix
    
def dark_pixel_array( imarr, p=10 ):
    dp = dark_pixels( imarr, p )
    dparr = imarr.copy()
    dparr.mask = ~np.repeat( np.expand_dims( dp.filled(), 2 ), 8, 2 )
    return dparr    

## Glint correction ----------------------------------------------------------

def nir_mean(msarr,nir_band=7):
    return msarr[:,:,nir_band].mean()
    
def cov_ratio(msarr,band,nir_band=7):
    cov_mat = np.cov(msarr[:,:,band].flatten(),msarr[:,:,nir_band].flatten())
    return cov_mat[0,1] / cov_mat[1,1]
    
def cov_ratios(msarr,nir_band=7):
    nbands = msarr.shape[-1] #assume Rows,Cols,Bands shape
    bands = range(nbands)
    cov_rats = []
    if nir_band in bands: bands.remove(nir_band)
    for band in bands:
        cov_rat = cov_ratio(msarr,band,nir_band)
        cov_rats.append(cov_rat)
    return np.array(cov_rats)
    
def glint_correct_image(imarr, glintarr, nir_band=7):
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