# -*- coding: utf-8 -*-
"""
The RasterDS object will provide some utilities for getting raster data sets into
and out of numpy array formats. There'll be some other methods as well.

Created on Fri Jun 27 14:58:13 2014

@author: jkibele
"""

import os
from osgeo import gdal, ogr, osr
from osgeo.gdalconst import GDT_Unknown,GDT_Byte,GDT_UInt16,GDT_Int16,GDT_UInt32,GDT_Int32,GDT_Float32, GDT_Float64,GDT_CInt16,GDT_CInt32,GDT_CFloat32,GDT_CFloat64
from osgeo.gdal_array import NumericTypeCodeToGDALTypeCode
import numpy as np
from RasterSubset import masked_subset

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
            if os.path.exists(self.file_path):
                print 'Could not open %s. This file does not seem to be one that gdal can open.' % self.file_path
            else:
                print 'Could not open %s. It seems that this file does not exist.' % self.file_path
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
    def epsg(self):
        srs = osr.SpatialReference(wkt=self.projection_wkt)
        return int( srs.GetAttrValue('AUTHORITY',1) )
        
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
        If the image has a "no data value" return a masked array.
        """
        return self.band_array_subset()
        
    def band_array_subset(self,xoff=0, yoff=0, win_xsize=None, win_ysize=None):
        """
        Return the image data in a numpy array of shape (Rows, Columns, Bands).
        If the image has a "no data value" return a masked array. Take a subset
        if subset values are given.
        """
        #barr = self.gdal_ds.ReadAsArray()
        #ourarr = barr.T.swapaxes(0,1)
        blist = []
        bmlist = []
        for b in range( self.gdal_ds.RasterCount ):
            b += 1 #Bands are 1 based indexed
            band = self.gdal_ds.GetRasterBand( b )
            barr = band.ReadAsArray(xoff=xoff,yoff=yoff,win_xsize=win_xsize,win_ysize=win_ysize)
            blist.append( barr )
            nodat = band.GetNoDataValue()
            if nodat is not None:
                bmlist.append( barr==nodat )
            else:
                bmlist.append( np.full_like( barr, False, dtype=np.bool ) )
        allbands = np.ma.dstack( blist )
        allbands.mask = np.dstack( bmlist )
        if nodat is not None:
            try:
                allbands.set_fill_value( nodat )
            except ValueError:
                # This means band.GetNoDataValue() is returning nan for integer layer
                # I just won't set the fill value
                pass
        # make sure that a single band raster will return (Rows,Columns,1(Band))
        if allbands.ndim==2:
            np.expand_dims(allbands,2)
        return allbands
        
    def geometry_subset(self, geom):
        """
        Return a subset of rds band array where the extent is the bounding box 
        of geom and all cells outside of geom are masked.
                    
        geom: shapely geometry
            The polygon bounding the area of interest.
            
        Returns:
            A numpy masked array of shape (Rows,Columns,Bands). Cells not within
            geom will be masked as will any values that were masked in rds.
        """
        return masked_subset(self, geom)
        
    def new_image_from_array(self,bandarr,outfilename=None,dtype=None,no_data_value=None):
        """
        Save an image like self from a band array.
        """
        if dtype==None:
            # try to translate the dtype
            dtype = NumericTypeCodeToGDALTypeCode( bandarr.dtype )
            if dtype==None:
                # if that didn't work out, just make it float32
                dtype = GDT_Float32
        if no_data_value==None:
            # try to figure it if it's a masked array
            if np.ma.is_masked( bandarr ):
                # gdal does not like the numpy dtypes
                no_data_value = np.asscalar( bandarr.fill_value )
                bandarr = bandarr.filled()
            else:
                # just make it -99 and hope for the best
                no_data_value = -99
        else: # a no_data_value has been supplied by the user
            if np.ma.is_masked( bandarr ):
                # set the array's fill value to no_data_value
                bandarr.fill_value = no_data_value
                bandarr = bandarr.filled()
        bandarr = np.rollaxis(bandarr,2,0)
        if not outfilename:
            outfilename = self.output_file_path()
        output_gtif_like_img(self.gdal_ds, bandarr, outfilename, no_data_value=no_data_value, dtype=dtype)
        return RasterDS(outfilename)
        
def output_gtif(bandarr, cols, rows, outfilename, geotransform, projection, no_data_value=-99, driver_name='GTiff', dtype=GDT_Float32):
    """Create a geotiff with gdal that will contain all the bands represented
    by arrays within bandarr which is itself array of arrays. Expecting bandarr
    to be of shape (Bands,Rows,Columns)"""
    # make sure bandarr is a proper band array
    if bandarr.ndim==2:
        bandarr = np.array([ bandarr ])
    driver = gdal.GetDriverByName(driver_name)
    # The compress and predictor options below just reduced a geotiff
    # from 216MB to 87MB. Predictor 2 is horizontal differencing.
    outDs = driver.Create(outfilename, cols, rows, len(bandarr), dtype,  options=[ 'COMPRESS=LZW','PREDICTOR=2' ])
    if outDs is None:
        print "Could not create %s" % outfilename
        sys.exit(1)
    for bandnum in range(1,len(bandarr) + 1):  # bandarr is zero based index while GetRasterBand is 1 based index
        outBand = outDs.GetRasterBand(bandnum)
        outBand.WriteArray(bandarr[bandnum - 1])
        outBand.FlushCache()
        outBand.SetNoDataValue(no_data_value)
        
    # georeference the image and set the projection
    outDs.SetGeoTransform(geotransform)
    outDs.SetProjection(projection)

    # build pyramids
    gdal.SetConfigOption('HFA_USE_RRD', 'YES')
    outDs.BuildOverviews(overviewlist=[2,4,8,16,32,64,128])
    
def output_gtif_like_img(img, bandarr, outfilename, no_data_value=-99, dtype=GDT_Float32):
    """Create a geotiff with attributes like the one passed in but make the
    values and number of bands as in bandarr."""
    cols = img.RasterXSize
    rows = img.RasterYSize
    geotransform = img.GetGeoTransform()
    projection = img.GetProjection()
    output_gtif(bandarr, cols, rows, outfilename, geotransform, projection, no_data_value, driver_name='GTiff', dtype=dtype)