# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 16:28:57 2014
This code also shows up in the Multispectral Land Masker QGIS plugin:
https://github.com/jkibele/LandMasker.
@author: jkibele
"""
import numpy as np
from scipy.ndimage.measurements import label

def connectivity_filter(in_array,threshold=1000,structure=None):
    """
    Take a binary array (ones and zeros), find groups of ones smaller
    than the threshold and change them to zeros.
    """
    #define how raster cells touch
    if structure:
        connection_structure = structure
    else:
        connection_structure = np.array([[0,1,0],
                                         [1,1,1],
                                         [0,1,0]])
    #perform the label operation
    labelled_array, num_features = label(in_array,structure=connection_structure)
    
    #Get the bincount for all the labels
    b_count = np.bincount(labelled_array.flat)
    
    #Then use the bincount to set the output values
    out_array = np.where(b_count[labelled_array] <= threshold, 0, in_array)
    return out_array

def simple_land_mask(in_arr,threshold=50):
    """
    Return an array of ones and zeros that can be used as a land mask. Ones
    will be water and zeros will be land. This method fails to mask out
    shadows.
    
    Args:
        in_arr (numpy.array): An array of shape (Rows,Columns). Should be a NIR
            band. For WorldView-2 imagery, I use band 8.
        
        threshold (int or float): The pixel value cut-off. Pixels with a value
            lower than this will be considered water and be marked as 1 in the
            output.
            
    Returns:
        output (numpy.array): An array of 1s and 0s. 1s over water, 0s over 
            land.
            
    Note: the output from this method should be used as input for the 
        connectivity_filter method defined above. This will remove the isolated
        pixels on land that are marked as water.
    """
    band = in_arr
    # make a copy so we can modify it for output and still have the 
    # original values to test against
    output = band.copy()
    # pixels at or bellow threshold to ones
    output[np.where(band <= threshold)] = 1
    # zero out pixels above threshold
    output[np.where(band > threshold)] = 0
    # if it was zero originally, we'd still like it to be zero
    output[np.where(band == 0)] = 0
    
    return output