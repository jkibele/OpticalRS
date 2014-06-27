# -*- coding: utf-8 -*-
"""
The RasterDS object will provide some utilities for getting raster data sets into
and out of numpy array formats. There'll be some other methods as well.

Created on Fri Jun 27 14:58:13 2014

@author: jkibele
"""

import os
from osgeo import gdal, ogr
from osgeo.gdalconst import *
import numpy as np

class RasterDS(object):
    """
    You can pass in a file path to any file that gdal can open or you can pass
    in a QGIS raster layer instead. Either way, you will get back a raster 
    dataset object.
    """
    def __init__(self, rlayer, overwrite=False):
        self.rlayer = rlayer
        self.overwrite = overwrite
        try:
            self.file_path = str( rlayer.publicSource() )
        except AttributeError:
            self.file_path = rlayer
        self.gdal_ds = self.__open_gdal_ds()
        # store the text portion of the file extension in case we need the file type
        self.file_type = os.path.splitext(self.file_path)[-1].split(os.path.extsep)[-1]
        
    def __open_gdal_ds(self):
        """return a gdal datasource object"""
        # register all of the GDAL drivers
        gdal.AllRegister()
    
        # open the image
        img = gdal.Open(self.file_path, GA_ReadOnly)
        if img is None:
            print 'Could not open %s. This file does not seem to be one that gdal can open.' % self.file_path
            return None
        else:
            return img
        
    @property
    def band_names(self):
        """
        Return a list of strings representing names for the bands. For now, this
        will simply be 'band1','band2','band2',... etc. At some point perhaps 
        these names will relate to the color of the bands.
        """
        return [ 'band'+str(i) for i in range(1,self.gdal_ds.RasterCount + 1) ]
        
    @property
    def projection_wkt(self):
        """
        Return the well known text (WKT) representation of the raster's projection.
        """
        return self.gdal_ds.GetProjection()
        
    @property
    def output_file_path(self):
        """
        Return a file path for output. Assume that we'll output same file extension.
        """
        if self.overwrite:
            return self.file_path
        else:
            f_ext = self.file_type
            fname = self.file_path
            add_num = 0
            while os.path.exists(fname):
                add_num += 1
                if add_num==1:
                    fname = fname.replace( os.path.extsep + f_ext, '_%i' % add_num + os.path.extsep + f_ext )
                else:
                    old = '_%i.%s' % ( add_num - 1, f_ext )
                    new = '_%i.%s' % ( add_num, f_ext )
                    fname = fname.replace( old, new )
            return fname
            
    @property
    def band_array(self):
        """
        Return the image data in a numpy array of shape (Rows, Columns, Bands).
        """
        barr = self.gdal_ds.ReadAsArray()
        return barr.T.swapaxes(0,1)