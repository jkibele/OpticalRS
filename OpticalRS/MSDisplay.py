# -*- coding: utf-8 -*-
"""
Multispectral Display
=====================

Methods for displaying multispectral images and information about them. I've
found these methods useful when exploring images.
"""

import numpy as np
from math import ceil
from matplotlib.pyplot import subplots

def multiband_histogram( img, nbins=256, figwidth=14 ):
    """
    Plot histogram and cdf for each band of img array.

    Parameters
    ----------
    img : numpy.array
        This is the image array of shape (Rows,Cols,Bands)
    nbins : int
        The number of bins to use in the histograms. (Default value = 256)
    figwidth :
        The width of figure. The height of the figure will be determined by the
        number of bands in the image. (Default value = 14)

    Returns
    -------
    Nothing
        This method dispalys a plot. Exactly how the plot is displayed is
        determined by your matplotlib settings.
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
    """
    Plot each band as a separate single band subplot.

    Parameters
    ----------
    img : numpy.array
        This is the image array of shape (Rows,Cols,Bands)
    **kwargs : keyword arguments
        This is how the keyword arguments get passed in. I'll list them below.
    ncols : int
        The number of columns in the plot. (default value = 2)
    figwidth : int or float
        The width of figure. The height of the figure will be determined by the
        number of bands in the image. (Default value = 14)
    cmap : matplotlib.colors.Colormap, optional, default: None
        The colormap to be used by ``imshow`` to dispaly the bands.
    subset : numpy.s_
        A numpy slice object used to display a subset of the image rather than
        the whole thing.


    Returns
    -------
    Nothing
        This method dispalys a plot. Exactly how the plot is displayed is
        determined by your matplotlib settings.
    """
    nbands = img.shape[-1]
    ncols = kwargs.pop('ncols',2)
    figwidth = kwargs.pop('figwidth',14)
    cmap = kwargs.pop('cmap',None)
    nrows = int( ceil( nbands/2.0 ))
    subset = kwargs.pop('subset',np.s_[:,:,:])
    fig, axarr = subplots(nrows, ncols, sharex=True, sharey=True, figsize=(figwidth, (figwidth/3.5)*nrows))
    for i, barr in enumerate( np.dsplit(img, nbands) ):
        ax = axarr.ravel()[i]
        axtit = "Band %i" % (i + 1)
        ax.set_title(axtit)
        ax.imshow( barr[subset].squeeze(), cmap=cmap )
        ax.axis('off')
