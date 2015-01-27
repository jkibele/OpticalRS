# -*- coding: utf-8 -*-
"""
Multispectral Exposure
======================

This module contains methods for adjusting the exposure of multispectral images.
Unless otherwise stated methods will expect image arrays in the shape of 
(Rows,Columns,Bands).

Most (if not all) of these methods will be based on scikit-image exposure
methods. These methods can't be used directly (for my purposes) because they
do not allow (in most cases) for use with masked arrays. My applications often
involve images in which land and deep water have been masked so I need exposure
methods that base calculations only on unmasked pixels.

I often need to apply exposure methods to many individual bands. Most of the 
scikit-image methods expect single band or RGB array. The methods in this
module will apply to an arbitrary number of bands.

Created on Tue Jan 27 15:16:10 2015
@author: jkibele
"""

import numpy as np
from skimage import exposure
from skimage.filters import rank as filter_rank
from skimage.morphology import disk

def apply_with_mask_as_mean( func, img, **func_kwargs ):
    """
    Convert masked values to mean of unmasked values and apply func. Return a 
    masked array with the same mask as the input.
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
    
def equalize_adapthist( img, **kwargs ):
    return multi_apply_with_mask_as_mean( img, exposure.equalize_adapthist, **kwargs )