##OpticalRS=group
##Multispectral_image=raster
##XML_File=file
##TOA_Reflectance=boolean True
##Output_raster=output raster

"""
WV2RadiometricCorrection
========================

This module reads parameters from a DigitalGlobe supplied xml file and applies
radiometric correction to WorldView-2 imagery according to the instructions
found here::

DigitalGlobe, 2010. Radiometric Use of WorldView-2 Imagery.
http://www.digitalglobe.com/downloads/Radiometric_Use_of_WorldView-2_Imagery.pdf

This code is also include in the OpticalRS library: http://jkibele.github.io/OpticalRS/

Notes
=====

This module gives results that are very close to, but not the same as, the Orfeo
Toolbox [Optical Calibration
module](https://www.orfeo-toolbox.org/CookBook/CookBooksu35.html). I think this
may be because the OTB module is using general parameters for WV-2 imagery while
this module is using specific parameters from the image's xml file. So, I am
reasonably certain this module is working as intended. On the other hand, I
really haven't investigated the issue. If you look into it, please let me know
what you find out. At some point I need to do some more testing and add in much
more thorough automated tests.
"""

from xml.etree.ElementTree import ElementTree as ET
from collections import OrderedDict
from datetime import datetime
from osgeo import gdal
from osgeo.gdalconst import *
import os, sys
import numpy as np

from OpticalRS.WV2RadiometricCorrection import open_raster, bandarr_from_ds, \
                                        toa_radiance_multiband, toa_reflectance_multiband, \
                                        output_gtif_like_img

img = open_raster(Multispectral_image)
bandarr = bandarr_from_ds(img)

if TOA_Reflectance:
    bandarr = toa_radiance_multiband(bandarr,XML_File)
else:
    bandarr = toa_reflectance_multiband(bandarr,XML_File)

output_gtif_like_img(img,bandarr,Output_raster)
