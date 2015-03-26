# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 14:09:45 2015

@author: jkibele
"""

import warnings
import numpy as np

def rescale( x, rmin=0.0, rmax=1.0 ):
    """
    Scale (normalize) the values of an array to fit in the interval [rmin,rmax].
    
    Args:
        x (numpy.array or maskedarray): The array you want to scale.
        
    Returns:
        array of floats. Masked values (if there are any) are not altered.
    """
    return rmin + (rmax - rmin) * ( x - x.min() ) / float( x.max() - x.min() )

def each_band_unmasked( imarr, funct, *args, **kwargs ):
    """
    
    """
    outlist = []
    for i in range( imarr.shape[-1] ):
        outlist.append( funct( imarr, *args, **kwargs ) )
    return np.dstack( outlist )
    
def each_band_masked( imarr, funct, *args, **kwargs ):
    outlist = []
    ismalist = []
    for i in range( imarr.shape[-1] ):
        newband = funct( imarr, *args, **kwargs )
        outlist.append( newband )
        ismalist.append( type(newband)==np.ma.MaskedArray )
    if False in ismalist:
        warnings.warn( "A function returned an unmasked array when a masked array was expected.")
    return np.ma.dstack( outlist )
    
def each_band( imarr, funct, *args, **kwargs ):
    if type(imarr)==np.ma.MaskedArray:
        return each_band_masked( imarr, funct, *args, **kwargs )
    else:
        return each_band_unmasked( imarr, funct, *args, **kwargs )