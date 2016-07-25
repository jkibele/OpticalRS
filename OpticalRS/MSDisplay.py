# -*- coding: utf-8 -*-
"""
Multispectral Display
=====================

Methods for displaying multispectral images and information about them. I've
found these methods useful when exploring images.
"""

import numpy as np
import pandas as pd
from math import ceil
from matplotlib.pyplot import subplots
from Const import wv2_bandnames, cnames
from MSExposure import equalize_adapthist

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
    fig, axarr = subplots(nrows, ncols, sharex=True, sharey=True,
                            figsize=(figwidth, (figwidth/3.5)*nrows))
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
    fig, axarr = subplots(nrows, ncols, sharex=True, sharey=True,
                            figsize=(figwidth, (figwidth/3.5)*nrows))
    for i, barr in enumerate( np.dsplit(img, nbands) ):
        ax = axarr.ravel()[i]
        axtit = "Band %i" % (i + 1)
        ax.set_title(axtit)
        ax.imshow( barr[subset].squeeze(), cmap=cmap )
        ax.axis('off')

def band_profile_display(imarr, p0, p1, displayband=None, rolling_means=False,
                            bandnames=wv2_bandnames, cnames=cnames, n=1000,
                            legend=True, ylim=None, darr=None):
    """
    Plot the image `imarr`, a line between `p0` and `p1`, and the spectral
    profile of `imarr`'s bands along that line.

    Parameters
    ----------
    imarr : numpy.array
        Image array of shape (Rows,Cols,Bands)
    p0 : array-like
        The starting point (x0,y0) for the profile line in array coordinates
        (not projected geographic coordinates).
    p1 : array-like
        The ending point (x1,y1) for the profile line in array coordinates
        (not projected geographic coordinates).
    displayband : int
        Display a single band of `imarr`. If `None` (default) display an
        adaptive histogram equalized RGB image of the first 3 bands (2,1,0) when
        there are 3 or more bands in `imarr`. If `imarr` has fewer than 3 bands
        the default behavior is to display the first band.
    rolling_means : boolean or int
        If `False` (default) rolling means will not be displayed for band values.
        If `True`, they will be displayed with a window of n/10. If an int is
        provided, that will be used as the window size.
    bandnames : array-like
        Labels for the bands of `imarr`. The default values are the names of
        WorldView-2 bands. The number of items in `bandnames` should be greater
        than or equal to the number of bands in `imarr`.
    cnames : array-like
        String matplotlib color names. The plots of band values will be these
        colors. The defaults are representative of the bands in a WorldView-2
        image. The number of items in `cnames` should be greater than or equal
        to the number of bands in `imarr`.
    n : int
        The number of points along the line from `p0` to `p1`. This will be the
        number of points along the x axis of the profile plot.

    Returns
    -------
    pyplot.figure
        A pyplot Figure instance. This can be used to save the figure to a file.
    """
    imarr = np.atleast_3d(imarr)
    nbands = imarr.shape[-1]
    if displayband==None:
        if nbands >= 3:
            disparr = equalize_adapthist(imarr[...,[2,1,0]], clip_limit=0.02)
        else:
            disparr = imarr[...,0]
    else:
        disparr = imarr[...,displayband]
    x0, y0 = p0
    x1, y1 = p1
    df = values_along_line(imarr, p0, p1, darr=darr, n=n, bandnames=bandnames)

    fig, (ax1, ax2) = subplots(ncols=2, figsize=(12,4), gridspec_kw={'width_ratios':[1,3]})
    ax1.imshow(disparr)
    ax1.plot([x0, x1], [y0, y1], 'ro-')
    ax1.set_axis_off()
    ax1.set_title("Profile Line")

    if rolling_means:
        plt_alp = 0.4
        lgnd = False
    else:
        plt_alp = 1.0
        lgnd = legend

    if nbands == 1:
        df.plot(ax=ax2, color=cnames[0], alpha=plt_alp, legend=lgnd)
    else:
        df.plot(ax=ax2, color=cnames[:nbands], alpha=plt_alp, legend=lgnd)

    if rolling_means:
        if type(rolling_means) is not bool:
            nwin = int(rolling_means)
        else:
            nwin = int( round(n/10.0) )
        rmns = pd.rolling_mean(df, nwin, center=True)
        if nbands == 1:
            rmns.plot(ax=ax2, color=cnames[0], legend=legend)
        else:
            rmns.plot(ax=ax2, color=cnames[:nbands], legend=legend)
    if darr is None:
        ax2.set_xlabel("Point Along Line")
    else:
        ax2.set_xlabel("Depth (m)")
    ax2.set_ylabel("Band Values")
    ax2.set_title("Band Values Along Profile Line")
    if ylim is not None:
        ax2.set_ylim( ylim )

    return fig

def values_along_line(imarr, p0, p1, darr=None, n=1000, bandnames=wv2_bandnames):
    """
    This will pull pixel values into a dataframe along a line from p0 to p1. It's
    used in `band_profile_display`.
    """
    imarr = np.atleast_3d(imarr)
    nbands = imarr.shape[-1]
    x0, y0 = p0
    x1, y1 = p1
    x, y = np.linspace(x0, x1, n).astype(np.int), np.linspace(y0, y1, n).astype(np.int)
    zi = imarr[y, x, :]
    if darr is not None:
        darr = darr.squeeze()
        depths = darr[y, x]
        df = pd.DataFrame(zi, columns=bandnames[:nbands], index=depths)
        # df['depth'] = depths
        df.sort_index(inplace=True)
        df.index.name = 'Depth'
    else:
        df = pd.DataFrame(zi, columns=bandnames[:nbands])
    return df
