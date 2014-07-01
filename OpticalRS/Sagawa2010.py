# -*- coding: utf-8 -*-
r"""
Created on Tue Jul  1 10:20:07 2014

In Sagawa et al. 2010, an alternative to Lyzenga's depth invariant index is 
proposed. Sagawa's reflectance index incorporates depth measurements in an 
attempt to correct for water column attenuation. Sagawa used chart depths. I'm 
going to try to use depths estimated from the satellite imagery.

>Sagawa, T., Boisnier, E., Komatsu, T., Mustapha, K.B., Hattour, A., Kosaka, N., Miyazaki, S., 2010. Using bottom surface reflectance to map coastal marine areas: a new application method for Lyzenga’s model. International Journal of Remote Sensing 31, 3051–3064. doi:10.1080/01431160903154341

Sagawa's starting place is equation 1 from Lyzenga 1978, rewritten as (just the 
symbols have been changed):

> $L_i = L_{si} + a_i r_i e^{-K_igZ}$

$Z$ is the depth and $g$ is some kind of geometric constant that I can't find a 
good explaination of anywhere. ...but that might not end up mattering.

Sagawa's reflectance index is described as (equation 4, Sagawa et al. 2010):

> $Index_i = \frac{ L_i - L_{si} }{ e^{-K_i g Z} }$

for each band $i$.

I know how to calculate all of those terms except for $K_i$ and $g$. I know how 
to get $\frac{ K_i }{ K_j }$ (see the [Lyzenga1981](Lyzenga1981.ipynb) IPython 
Notebook) but not just $K_i$. However, Sagawa offers this somewhat cryptic 
statement:

>"Using sea truth data, we plotted satellite data against depth for sandy bottom types. The regression curve of Lyzenga’s model is then obtained. From this curve $–Kg$ can be deduced, as $g$ can be geometrically calculated from sun and satellite zenith angles."

I think that by "Lyzenga's model", Sagawa is referring to equation 1 (the first 
equation in this notebook). So, if my dark pixel subtracted image is $Y$ and a 
particular band $i$ of that image is $Y_i$, then I can rewrite equation 1 as:

> $Y_i = a_i r_i e^{-K_i g Z}$

If I say $a_i r_i = A_i$, $-K_i g = B_i$, and say the depth ($Z$) is $x$ then I 
can rewrite again as:

> $Y_i = A_i e^{B x}$

Then I can log both sides and have:

> $\ln{ Y_i } = Bx + \ln{ A_i }$

Then I can use my depth estimates as $x$ values and regress against the logged 
reflectance values. The slope will give me $B = -K_i g$ and the intercept will 
be $\ln{ A_i }$. Then I can use the $-K_i g$ for each band to calculate 
$Index_i$ according to Sagawa's equation 4 and I don't really have to worry 
about seperating $-K_i$ from $g$.

@author: jkibele
"""

from scipy.stats import linregress
import numpy as np

def band_attenuation_geometric(bandarr,deptharr):
    """
    Given a single (Rows x Columns) band array and a depth array from the same 
    region, return -K*g value for that band. See section 2.2 and 2.3 from 
    Sagawa et al., 2010. The arrays you pass in should be taken from an area 
    with uniform substrate and varying depth.
    
    Args:
        bandarr (numpy.array): Array of reflectance (or radiance or DN) values
            for a subset of pixels over a uniform substrate (often sand). The 
            values in this array replace the Li - Lsi term so, if you want 
            dark pixel subtraction or some other correction, apply it before
            you pass in the array.
            
        deptharr (numpy.array): Array of depth values for the same set of pixels
            as bandarr. Shape must match that of bandarr.
            
    Returns:
        slope (float): This is equal to -K*g from Sagawa et al., 2010. This is
            the only return from this method used in reflectance index 
            calculations.
        
        intercept (float): The y intercept from the linear regression. This is 
            not used for reflectance index calculations but may be useful for 
            plotting regression results.
            
        r_value (float): The r value from the linear regression. Not used in 
            reflectance index calculations but can give you an idea of how well
            the regression fit the data.        
    """
    lnY = np.log(bandarr).flatten()
    X = deptharr.flatten()
    slope, intercept, r_value, p_value, std_err = linregress(X,lnY)
    return slope, intercept, r_value
    
def single_band_reflectance_index(single_band_arr,depth_arr,negKG):
    r"""
    Calculate reflectance index for a single band according to equation 4 from
    Sagawa et al. 2010. The input array will represent the whole L_i - L_si
    term. In other words, if you want dark pixel subtraction, do it first and 
    use the results as input for this method.
    
    Args:
        single_band_arr (numpy.array): A single band Rows x Columns shaped 
            array from the multispectral image for which a reflectance index
            image is to be created. Sagawa et al. used radiance values but you
            may be able to use DN or reflectance values as well.
            
        depth_arr (numpy.array): An array of depths (in meters) of the same 
            dimensions as the single_band_arr.
            
        negKG (float): The -K*g value for this band. This is the slope value
            returned from the `band_attenuation_geometric` method. See the 
            docstring for that method for more information.
            
    Returns:
        RI (numpy.array): An array of the same dimensions as the input
            containing reflectance index values. This index is (or should be)
            linearly related to bottom reflectance.
    """
    RI = single_band_arr / np.exp(negKG*depth_arr)
    return RI
    
def negKg_regression_array(bandarr,deptharr,band_list=None):
    """
    Create an array of -K*g values, intercept values, and r values. One 
    set for each band of a multispectral image array. See the docstring 
    for `band_attenuation_geometric` for details.
    
    Args:
        bandarr (numpy.array): An image array of (Rows, Columns, Bands)
            shape. This should be a subset from the image you wish to create
            a reflectance index for taken from an area with constant bottom
            type and varying depth. Sand is a good choice.
            
        deptharr (numpy.array): An array of (Rows, Columns) shape with the
            same number of rows and columns as bandarr containing depths in
            meters.
            
        band_list (list of ints, optional): A subset of the bands in bandarr.
            If supplied, only the values for the bands in the list will be 
            calculated and returned. If left as `None`, all bands will be 
            calculated and returned.
            
    Returns:
        results (numpy.array): An array with 3 columns and one row for each
            band considered. The first column will contain the -K*g values
            (the slope from the regression). The second column will contain
            intercept values from the regression. The third column will 
            contain the r value from each regression.
    """
    if not band_list:
        band_list = range(bandarr.shape[-1])
    outlist = []
    for i in band_list:
        negKg = band_attenuation_geometric(bandarr[:,:,i],deptharr)
        outlist.append(negKg)
    return np.array(outlist)
    
def negKg_array(bandarr,deptharr,band_list=None):
    """
    Create an array of -K*g values. One for each band of a multispectral image 
    array. See the docstring for `band_attenuation_geometric` for details.
    
    Args:
        bandarr (numpy.array): An image array of (Rows, Columns, Bands)
            shape. This should be a subset from the image you wish to create
            a reflectance index for taken from an area with constant bottom
            type and varying depth. Sand is a good choice.
            
        deptharr (numpy.array): An array of (Rows, Columns) shape with the
            same number of rows and columns as bandarr containing depths in
            meters.
            
        band_list (list of ints, optional): A subset of the bands in bandarr.
            If supplied, only the values for the bands in the list will be 
            calculated and returned. If left as `None`, all bands will be 
            calculated and returned.
            
    Returns:
        results (numpy.array): A 1D array of -K*g values. There will be one
            for each band or, if band_list has been provided, one for each
            entry in that list.
    """
    if not band_list:
        band_list = range(bandarr.shape[-1])
    nra = negKg_regression_array(bandarr,deptharr,band_list=band_list)
    return nra[:,0]
    
def reflectance_index(bandarr,deptharr,negKgarr,band_list=None):
    """
    Produce a reflectance index image for each band of an image and return
    it as a (Row, Column, Band) shaped array. For more information see the 
    docstring for `single_band_reflectance_index`. This method simply
    applies `single_band_reflectance_index` to multiple bands.
    
    Args:
        bandarr (numpy.array): An image array of (Rows, Columns, Bands)
            shaped array of the multispectral image for which a reflectance 
            index image is to be created. Sagawa et al. used radiance values 
            but you may be able to use DN or reflectance values as well.
            
        deptharr (numpy.array): An array of (Rows, Columns) shape with the
            same number of rows and columns as bandarr. It should contains 
            depths in meters.
            
        band_list (list of ints, optional): A subset of the bands in bandarr.
            If supplied, only the values for the bands in the list will be 
            calculated and returned. If left as `None`, all bands will be 
            calculated and returned.
            
    Returns:
        RI (numpy.array): An array of the same row and column dimensions as 
            the input containing reflectance index values. Array dimensions 
            will be (Rows, Columns, Bands). The number of bands will be equal 
            to the number of bands in `bandarr` unless `band_list` has been 
            specified. In that case the number of bands will be equal to the 
            length of `band_list`. The index values are (or should be)
            linearly related to bottom reflectance.
    """
    arrlist = []
    if not band_list:
        band_list = range(bandarr.shape[-1])
    for i in band_list:
        RI = single_band_reflectance_index(bandarr[:,:,i], deptharr, negKgarr[i])
        arrlist.append(RI)
    return np.dstack(arrlist)