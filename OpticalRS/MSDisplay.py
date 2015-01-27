# -*- coding: utf-8 -*-
"""
Multispectral Display
=====================

Methods for displaying multispectral images and information about them.

Created on Tue Jan 27 15:39:55 2015
@author: jkibele
"""

import numpy as np
from math import ceil
from matplotlib.pyplot import subplots

def multiband_histogram( img, nbins=256, figwidth=14 ):
    """
    Plot histogram and cdf for each band of img array.    
    """
    imshp = img.shape
    nbands = imshp[-1]
    npix = imshp[0] * imshp[1]
    # flatten each band
    img_rs = img.reshape( [npix,nbands] )
    # want two bands per column with as many columns as needed
    ncols = 2
    nrows = int( ceil( nbands / 2.0 ) )
    fig, axarr = subplots(nrows, ncols, sharex=True, sharey=True, figsize=(figwidth, (figwidth/3.5)*nrows))
    for i in range(nbands):
        try:
            pixvalues = img_rs[:,i].compressed()
        except AttributeError:
            pixvalues = img_rs[:,i].ravel()
        ax = axarr.ravel()[i]
        ax.hist( pixvalues, bins=nbins, color='steelblue', edgecolor='none', alpha=0.8 )
        axtit = "Band %i" % (i + 1)
        ax.set_title(axtit)
        counts, bins = np.histogram( pixvalues,bins=nbins,normed=False )
        cdf = np.cumsum( counts )
        ax.twinx().plot( bins[1:], cdf, color='r', alpha=0.8 )
        
def view_bands( img, **kwargs ):
    nbands = img.shape[-1]
    if kwargs.has_key('ncols'):
        ncols = kwargs['ncols']
    else:
        ncols = 2
    if kwargs.has_key('figwidth'):
        figwidth = kwargs['figwidth']
    else:
        figwidth = 14
    nrows = int( ceil( nbands/2.0 ))
    fig, axarr = subplots(nrows, ncols, sharex=True, sharey=True, figsize=(figwidth, (figwidth/3.5)*nrows))
    for i, barr in enumerate( np.dsplit(img, nbands) ):
        ax = axarr.ravel()[i]
        axtit = "Band %i" % (i + 1)
        ax.set_title(axtit)
        ax.imshow( barr.squeeze() )
        ax.axis('off')
    