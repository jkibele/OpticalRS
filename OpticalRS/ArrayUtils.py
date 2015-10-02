# -*- coding: utf-8 -*-
"""
This module contains functions that are applied to numpy array representations
of images. Unless stated otherwise, image arrays are expected to be of shape
(RowsxColumnsxN) where N is the number of bands.
"""

import warnings
import numpy as np
import pandas as pd

def equalize_band_masks( marr ):
    """
    For a multiband masked array, ensure that a pixel masked in one band
    is masked in all bands so that each band has an equal number of masked
    pixels.

    Parameters
    ----------
    marr : np.ma.MaskedArray
        Image array with unequal band masks.

    Returns
    -------
    np.ma.MaskedArray
        Image array with equal band masks
    """
    nbands = marr.shape[-1]
    anymask = np.repeat( np.expand_dims( marr.mask.any(2), 2 ), nbands, 2 )
    marr.mask = anymask
    return marr

def band_df( imarr, bandnames=None, equalize_masks=True ):
    """
    Return a pandas dataframe with spectral values and depths from `imarr`.

    Parameters
    ----------
    imarr : np.array or np.ma.MaskedArray
        The image array.
    bandnames : list
        A list of band names that will become column names in the returned
        pandas datafrome. The list must be the same length of third dimension
        (number of bands) as imarr. If no list of bandnames is supplied,
        'band1','band2',...,'bandN'] will be used.
    equalize_masks : bool
        If ``True``, ``ArrayUtils.equalize_band_masks`` will be called on imarr
        before the pandas dataframe is created.

    Returns
    -------
    pandas.DataFrame
        A dataframe where the columns represent bands and each row represents a
        pixel.
    """
    if equalize_masks and np.ma.isMaskedArray(imarr):
        imarr = equalize_band_masks( imarr )
    nbands = imarr.shape[-1]
    if not bandnames:
        bandnames = [ 'band'+str(i+1) for i in range( nbands ) ]
    ddict = {}
    for bn in range( nbands ):
        if np.ma.isMaskedArray(imarr):
            ddict[bandnames[bn]] = imarr[...,bn].compressed()
        else:
            ddict[bandnames[bn]] = imarr[...,bn].ravel()
    return pd.DataFrame( ddict )

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

def mask3D_with_2D( imarr, mask, keep_old_mask=True ):
    """
    Apply a 2D mask to a 2D x N-band image array.

    Parameters
    ----------
    imarr : numpy array or masked array
        An 3 dimensional array (RxCxN) where the third dimension (N) is the
        number of bands.

    mask : numpy array
        A 2 dimensional array (RxC) of the same shape as the first 2 dimensions
        of imarr. This mask will be repeated N times and applied to `imarr`.
        `mask` must be able to be converted to boolean dtype.

    keep_old_mask : bool (default: True)
        If False, any existing mask for imarr will be discarded and `mask` will
        be the only mask of the output. If True (default), the output mask
        will be the union of `mask` and any existing mask.

    Returns
    -------
    out : numpy masked array
        `imarr` with `mask` applied to every band.
    """
    nbands = imarr.shape[-1]
    rmask = np.repeat( np.expand_dims( mask, 2 ), nbands, axis=2 )
    try:
        out = np.ma.MaskedArray( imarr, mask=rmask, fill_value=imarr.fill_value, keep_mask=keep_old_mask )
    except AttributeError:
        # if `imarr` is not a masked array, it won't have `.fill_value`.
        out = np.ma.MaskedArray( imarr, mask=rmask, )
    return out

def rescale( x, rmin=0.0, rmax=1.0, clip_extremes=False, plow=1.0, phigh=99.0 ):
    """
    Scale (normalize) the values of an array to fit in the interval [rmin,rmax].

    Parameters
    ----------
    x : numpy.array or maskedarray
        The array you want to scale.
    rmin : float or int
        The minimum that you'd like to scale to. Default=0.0
    rmax : float or int
        The maximum to scale to. Default=1.0
    clip_extremes : boolean
        If True, mask values below `plow` percentile and values above `phigh`
        percentile. Default=False.
    plow : float or int
        Lower percentile for range of values to keep. Ignored if `clip_extremes`
        is False.
    phigh : float or int
        Upper percentile for range of values to keep. Ignored if `clip_extremes`
        is False.

    Returns
    -------
    array of floats
        Masked values (if there are any) are not altered unless
        `clip_extremes==True` in which case pixels outside the
        percentiles `plow` and `phigh` will be masked.
    """
    if clip_extremes:
        low_lim = np.percentile( x, plow )
        high_lim = np.percentile( x, phigh )
        x = np.ma.masked_outside( x, low_lim, high_lim )
#    print x.min(), x.max(),
    outarr = rmin + (rmax - rmin) * ( x - x.min() ) / float( x.max() - x.min() )
#    print "------",
#    print outarr.min(), outarr.max()
    return outarr

def each_band_unmasked( imarr, funct, *args, **kwargs ):
    """
    Apply a function to each band of a RxCxBands shaped unmasked image array
    as if it were a single band (RxC) array.

    Parameters
    ----------
    imarr : numpy array
        An 3 dimensional array (RxCxN) where the third dimension (N) is the
        number of bands.
        
    funct : function
        The function to apply to each band. It must take an array as input and
        return an array.
        
    *args : various
        Arguments to be supplied to to `funct`.
        
    **kwargs :
        Keyword arguments to be supplied to `funct`.
        

    Returns
    -------
    numpy array
        The results to applying `funct` to each band.
    """
    outlist = []
    for i in range( imarr.shape[-1] ):
        outlist.append( funct( imarr[...,i], *args, **kwargs ) )
    return np.dstack( outlist )

def each_band_masked( imarr, funct, *args, **kwargs ):
    """
    Apply a function to each band of a RxCxBands shaped masked image array
    as if it were a single band (RxC) array. If `funct` returns an unmasked
    array, this method will copy the mask and fill value from `imarr` and apply
    them to the returned array.

    Parameters
    ----------
    imarr : numpy masked array
        An 3 dimensional array (RxCxN) where the third dimension (N) is the
        number of bands.
        
    funct : function
        The function to apply to each band. It must take an array as input and
        return an array.
        
    *args : various
        Arguments to be supplied to to `funct`.
        
    **kwargs :
        Keyword arguments to be supplied to `funct`.
        

    Returns
    -------
    numpy masked array
        The results to applying `funct` to each band.
    """
    outlist = []
    ismalist = []
    for i in range( imarr.shape[-1] ):
        newband = funct( imarr[:,:,i], *args, **kwargs )
        outlist.append( newband )
        ismalist.append( type(newband)==np.ma.MaskedArray )
    if False in ismalist:
        warnings.warn( "A function returned an unmasked array when a masked array was expected. I'll try to copy the mask from the input array.")
        outarr = np.ma.dstack( outlist )
        outarr.mask = imarr.mask
        outarr.set_fill_value( imarr.fill_value )
        return outarr
    else:
        return np.ma.dstack( outlist )

def each_band( imarr, funct, *args, **kwargs ):
    """
    Apply a function to each band of a RxCxBands shaped image array as if it 
    were a single band (RxC) array. Masked and unmasked arrays are handled 
    slightly differently. See each_band_masked and each_band_unmasked for
    details.

    Parameters
    ----------
    imarr : numpy array
        An 3 dimensional array (RxCxN) where the third dimension (N) is the
        number of bands.
        
    funct : function
        The function to apply to each band. It must take an array as input and
        return an array.
        
    *args : various
        Arguments to be supplied to to `funct`.
        
    **kwargs :
        Keyword arguments to be supplied to `funct`.
        

    Returns
    -------
    numpy array
        The results to applying `funct` to each band.
        
    See Also
    --------
    each_band_masked, each_band_unmasked
    """
    if type(imarr)==np.ma.MaskedArray:
        return each_band_masked( imarr, funct, *args, **kwargs )
    else:
        return each_band_unmasked( imarr, funct, *args, **kwargs )