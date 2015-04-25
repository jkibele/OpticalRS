# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 19:17:24 2015

@author: jkibele
"""

import numpy as np
from skimage.filters import rank
from skimage import morphology

## Deep water masking --------------------------------------------------------

def dark_pixels( imarr, p=10 ):
    # average DNs to get (RxCx1) brightness
    brt = imarr.mean(axis=2)
    dim_thresh = np.percentile( brt.compressed(), p )
    dark_pix = ( brt <= dim_thresh )
    if np.ma.is_masked( dark_pix ):
        dark_pix.set_fill_value( False )
    return dark_pix
    
def moving_window( dark_arr, win_size=3 ):
    win = morphology.square( win_size )
    npix = win.size
    if np.ma.is_masked( dark_arr ):
        outarr = rank.sum( dark_arr.filled().astype('uint8'), win ) / float( npix )
        outarr = np.ma.MaskedArray( outarr, mask=dark_arr.mask, fill_value=dark_arr.fill_value )
    else:
        outarr = rank.sum( dark_arr.astype('uint8'), win ) / float( npix )
    return outarr
    
def dark_kernels( imarr, p=10, win_size=3, win_percentage=50 ):
    dps = dark_pixels( imarr, p=p )
    if win_size:
        dmeans = moving_window( dps, win_size=win_size )
        dps = dmeans >= (win_percentage/100.0)
    return dps.astype('bool')
    
def dark_pixel_array( imarr, p=10, win_size=3, win_percentage=50 ):
    dp = dark_kernels( imarr, p, win_size, win_percentage )
    dparr = imarr.copy()
    dparr.mask = ~np.repeat( np.expand_dims( dp.filled(), 2 ), 8, 2 )
    return dparr   
    
def bg_thresholds( dark_arr, n_std=3 ):
    nbands = dark_arr.shape[-1]
    darkmeans = dark_arr.reshape(-1,nbands).mean(0).data
    darkstds = dark_arr.reshape(-1,nbands).std(0).data
    return darkmeans + n_std * darkstds

## Glint correction ----------------------------------------------------------

def nir_mean(msarr,nir_band=7):
    return msarr[...,nir_band].mean()
    
def cov_ratio(msarr,band,nir_band=7):
    if np.ma.is_masked( msarr ):
        b = msarr[...,band].compressed()
        nir_b = msarr[...,nir_band].compressed()
    else:
        b = msarr[...,band].flatten()
        nir_b = msarr[...,nir_band].flatten()
    cov_mat = np.cov( b, nir_b )
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