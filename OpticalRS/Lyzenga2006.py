# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 19:17:24 2015

@author: jkibele
"""

import numpy as np

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