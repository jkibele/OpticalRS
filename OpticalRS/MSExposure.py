# -*- coding: utf-8 -*-
"""
Multispectral Exposure
======================

This module contains **POORLY TESTED** methods for adjusting the exposure of
multispectral images. Unless otherwise stated methods will expect image arrays
in the shape of (Rows,Columns,Bands). These methods should be considered
expirimental at this point.

Most (if not all) of these methods will be based on scikit-image exposure
methods. These methods can't be used directly (for my purposes) because they do
not allow (in most cases) for use with masked arrays. My applications often
involve images in which land and deep water have been masked so I need exposure
methods that base calculations only on unmasked pixels.

I often need to apply exposure methods to many individual bands. Most of the
scikit-image methods expect single band or RGB array. The methods in this
module will apply to an arbitrary number of bands.
"""

import numpy as np
from skimage import exposure
from skimage.filters import rank as filter_rank
from skimage.morphology import disk

def apply_with_mask_as_mean( func, img, **func_kwargs ):
    """
    Convert masked values to mean of unmasked values and apply func. Return a
    masked array with the same mask as the input.

    Parameters
    ----------
    func :

    img :

    **func_kwargs :


    Returns
    -------

    """
    if type( img )==np.ma.MaskedArray:
        myimg = img.copy()
        orig_fill = myimg.get_fill_value()
        myimg.set_fill_value( myimg.mean() )
        out_img = func( myimg.filled(), **func_kwargs )
        out_img = np.ma.masked_array( out_img, mask=myimg.mask, fill_value=orig_fill )
        del myimg
    else:
        out_img = func( img, **func_kwargs )
    return out_img

def multi_apply_with_mask_as_mean( img, func, **func_kwargs ):
    """
    Apply a function to each band of a multi-band image array. Masked elements
    will be replaced with the mean of unmasked values for the calculation.

    Parameters
    ----------
    img :

    func :

    **func_kwargs :


    Returns
    -------

    """
    imshp = img.shape
    if img.ndim > 2:
        nbands = imshp[-1]
    else:
        nbands = 1
    outlist = []
    for i in range(nbands):
        outlist.append( apply_with_mask_as_mean( func, img[:,:,i], **func_kwargs ) )
    outarr = np.ma.dstack( outlist )
    outarr.fill_value = img.fill_value
    outarr.mask = img.mask
    return outarr

def multi_apply_rank_filter( img, func, **kwargs ):
    """
    The fliter.rank methods take a mask. This will just apply them to a masked
    array.

    Parameters
    ----------
    img :

    func :

    **kwargs :


    Returns
    -------

    """
    selem = kwargs.pop('selem',disk(5) )
    imshp = img.shape
    if img.ndim > 2:
        nbands = imshp[-1]
    else:
        nbands = 1
    outlist = []
    for i in range(nbands):
        barr = img[:,:,i].squeeze()
        kwargs.update( {'mask': ~barr.mask} )
        outlist.append( func( barr, selem, **kwargs ) )
    outarr = np.ma.dstack( outlist )
    outarr.fill_value = img.fill_value
    outarr.mask = img.mask
    return outarr

def multi_rescale_intensity( img, p0=0, p1=99, out_range='dtype' ):
    """
    Apply skimage.exposure.rescale_intensity to each band. Use percentiles p0
    and p1 to determine the in_range.

    Parameters
    ----------
    img :

    p0 :
         (Default value = 0)
    p1 :
         (Default value = 99)
    out_range :
         (Default value = 'dtype')

    Returns
    -------

    """
    imshp = img.shape
    if img.ndim > 2:
        nbands = imshp[-1]
    else:
        nbands = 1
    outlist = []
    for i in range(nbands):
        barr = img[:,:,i]
        inmin = np.percentile( barr.compressed(), p0 )
        inmax = np.percentile( barr.compressed(), p1 )
        outband = exposure.rescale_intensity( barr, in_range=(inmin,inmax), out_range=out_range )
        outlist.append( outband )
    outarr = np.ma.MaskedArray( np.dstack(outlist), mask=img.mask, fill_value=img.fill_value )
    return outarr

def equalize_adapthist( img, **kwargs ):
    """


    Parameters
    ----------
    img :

    **kwargs :


    Returns
    -------

    """
    return multi_apply_with_mask_as_mean( img, exposure.equalize_adapthist, **kwargs )

def equalize_hist( img ):
    """
    Given an image of shape RxCxBands, histogram equalize each band
    and reassemble into same shape and return.

    Parameters
    ----------
    img :


    Returns
    -------

    """
    nbins = 256
    imshp = img.shape
    nbands = imshp[-1]
    npix = imshp[0] * imshp[1]
    # flatten each band
    img_rs = img.reshape( [npix,nbands] )
    # handle masked arrays
    if type( img_rs )==np.ma.core.MaskedArray:
        img_rs_eh = np.ma.apply_along_axis(lambda x: exposure.equalize_hist(x.data,nbins=nbins,mask=~x.mask), 0, img_rs)
        img_rs_eh.mask = img_rs.mask
    # apply histogram equalization along each band
    else:
        img_rs_eh = np.apply_along_axis(exposure.equalize_hist, 0, img_rs)
    return img_rs_eh.reshape( imshp )
