# -*- coding: utf-8 -*-
"""
ArrayUtils
==========

This module contains functions that are applied to numpy array representations
of images. Unless stated otherwise, image arrays are expected to be of shape
(RowsxColumnsxN) where N is the number of bands.
"""

import warnings
import numpy as np
import pandas as pd
from tempfile import NamedTemporaryFile
from scipy.misc import imsave
from scipy import ndimage as nd

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
    anymask = np.repeat( np.expand_dims( marr.mask.any(-1), -1 ), nbands, -1 )
    marr.mask = anymask
    return marr

def equalize_array_masks(*arrs):
    """
    Given some arrays (2 or more), return them with only pixels unmasked in all
    arrays still unmasked. Input arrays can have different numbers of bands but
    must have the same numbers of rows and columns.

    Parameters
    ----------
    arr1 : np.ma.MaskedArray
        An array of shape (RxC) or (RxCxBands).
    arr2 : np.ma.MaskedArray
        An array of shape (RxC) or (RxCxBands).
    arrN : np.ma.MaskedArray
        An array of shape (RxC) or (RxCxBands).

    Returns
    -------
    tuple of N np.ma.MaskedArray
        The original arrays with the same 2d mask for each band. If a pixel
        (RxC position) is masked in any band of any input array, it will be
        masked in every band of the output.
    """
    arrs = [np.ma.atleast_3d(a) for a in arrs]
    masks = [a.mask for a in arrs]
    maskstack = np.dstack(masks)
    combmask = maskstack.any(axis=2)
    arrsout = [mask3D_with_2D(a, combmask) for a in arrs]
    return arrsout

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
    rmask = np.repeat( np.atleast_3d(mask), nbands, axis=2 )
    try:
        out = np.ma.MaskedArray( imarr, mask=rmask, fill_value=imarr.fill_value,
                                 keep_mask=keep_old_mask )
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
        # The mask copying seems to work well, so I'll comment out the warning.
        # msg = """A function returned an unmasked array when a masked array was
        #       expected. I'll try to copy the mask from the input array."""
        # warnings.warn(msg)
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

def save_temp(imarr, rds=None):
    """
    Save the array to a temporary image file. Save to a geotiff if an `RasterDS`
    template is provided, otherwise save to a png. Don't forget to delete the
    tempfile!

    Parameters
    ----------
    imarr : ndarray, MxN or MxNx3 or MxNx4
        Array containing image values. If the shape is MxN, the  array
        represents a grey-level image. Shape MxNx3 stores  the red, green and
        blue bands along the last dimension.  An alpha layer may be included,
        specified as the last  colour band of an MxNx4 array.

    rds : OpticalRS.RasterDS instance
        A RasterDS to use as a template to create a geotiff. The RDS needs to
        have the same (R x C) dimensions as `imarr` but can have a different
        number of bands.

    Returns
    -------
    string
        The file path to the saved image. Delete this when you are done with it!

    Example
    -------

    In IPython on Unbuntu, you can put the following in a cell to  view a
    temporary .png with the Eye of Gnome image viewer and delete the temp image
    when you close the viewer:

        > tmpfp = save_temp(imarr)
        > !eog {tmpfp}
        > !rm {tmpfp}

    Similarly, this will save a temporary geotiff and open it in QGIS:

        > tmpfp = save_temp(imarr, rds=imrds)
        > !qgis {tmpfp}
        > !rm {tmpfp}
    """
    NP2GDAL_CONVERSION = {
        "bool": 1,
        "uint8": 1,
        "int8": 1,
        "uint16": 2,
        "int16": 3,
        "uint32": 4,
        "int32": 5,
        "float32": 6,
        "float64": 7,
        "complex64": 10,
        "complex128": 11,
    }
    if rds is None:
        fileTemp = NamedTemporaryFile(delete=False, suffix='.png')
        tfp = fileTemp.name
        fileTemp.close()
        imsave(tfp, imarr)
    else:
        fileTemp = NamedTemporaryFile(delete=False, suffix='.tif')
        tfp = fileTemp.name
        fileTemp.close()
        rds.new_image_from_array(np.atleast_3d(imarr),
                             tfp, dtype=NP2GDAL_CONVERSION[imarr.dtype.name])
    print "Temp image created. Don't forget to delete the file: {}".format(tfp)
    return tfp

def invalid_fill_single(data, invalid=None):
    """
    Replace the value of invalid 'data' cells (indicated by 'invalid')
    by the value of the nearest valid data cell. I got this off of stackexchange.

    Input:
        data:    numpy array of any dimension
        invalid: a binary array of same shape as 'data'.
                 data value are replaced where invalid is True
                 If None (default), use: invalid  = np.isnan(data)

    Output:
        Return a filled array.
    """
    if invalid is None: invalid = np.isnan(data)

    ind = nd.distance_transform_edt(invalid,
                                    return_distances=False,
                                    return_indices=True)
    return np.ma.masked_where(data.mask, data[tuple(ind)])

def invalid_fill(data, invalid):
    """
    Fill invalid data with values from nearest valid data. This function calls
    `invalid_fill_single` on each image band seperately to avoid filling with
    values from another band.
    """
    data = np.atleast_3d(data)
    invalid = np.atleast_3d(invalid)
    nbands = data.shape[-1]
    outarr = data.copy()
    for b in range(nbands):
        outarr[...,b] = invalid_fill_single(outarr[...,b], invalid[...,b])
    outarr = np.ma.masked_where(data.mask, outarr)
    return outarr
