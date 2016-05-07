# -*- coding: utf-8 -*-
"""
RasterDS
========

The `RasterDS` object will provide some utilities for getting raster data sets
into and out of numpy array formats. The main feature is the simplification of
reading and writing to GeoTiffs from `numpy.array` format.
"""

import os,sys
from osgeo import gdal, osr, ogr
from osgeo.gdalconst import *
from osgeo.gdal_array import NumericTypeCodeToGDALTypeCode
import numpy as np
from scipy.stats import mode as scipymode
from RasterSubset import masked_subset
from shapely.geometry import Polygon as shpPoly
import shapely as shpl

class RasterDS(object):
    """
    You can pass in a file path to any file that gdal can open or you can pass
    in a QGIS raster layer instead. Either way, you will get back a raster
    dataset object.

    Parameters
    ----------
    rlayer : string or QGIS raster layer
        If string, it should be a file path to a file (e.g. a GeoTiff) that can
        be read by `GDAL <http://gdal.org>`_. See the GDAL documentation for the
        full list of compatible formats. A `RasterDS` can also be created from
        a QGIS raster layer. This is useful when building `OpticalRS` based QGIS
        processing tools and plugins or when using the Python command line in
        QGIS.
    overwrite : boolean, optional
        Whether to allow overwriting of the data source file. This feature is
        incomplete in implementation and testing so don't rely on it. Default is
        `False`.
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
        """
        Return a gdal datasource object. In theory, this will only be called by
        other `RasterDS` methods. There shouldn't be a need for a user to
        interact directly with the GDAL data source object.
        """
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
        Return the well known text (WKT) representation of the raster's
        projection.
        """
        return self.gdal_ds.GetProjection()

    @property
    def epsg(self):
        """
        Return the EPSG code for the raster's projection.
        """
        srs = osr.SpatialReference(wkt=self.projection_wkt)
        return int( srs.GetAttrValue('AUTHORITY',1) )

    @property
    def raster_extent_list(self):
        """
        Get the extent of the raster as a list of coordinates.
        """
        gds = self.gdal_ds
        gt = gds.GetGeoTransform()
        cols = gds.RasterXSize
        rows = gds.RasterYSize
        return get_extent(gt, cols, rows)

    @property
    def raster_extent(self):
        """
        Get a shapely polygon representation of the raster extent.
        """
        return shpPoly(self.raster_extent_list)

    @property
    def output_file_path(self):
        """
        Return a file path for output. Assume that we'll output same file
        extension.
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
        if subset values are given. With default values, the whole image array
        will be returned.

        Parameters
        ----------
        xoff : int, optional
            The start position in the x-axis.
        yoff : int, optional
            The start position in the y-axis.
        win_xsize : int, optional
            The number of pixels to read in the x direction.
        win_ysize : int, optional
            The number of pixels to read in the y direction.

        Returns
        -------
        np.ma.MaskedArray
            The whole image array (default) or a subset of the image array. The
            shape of the returned array will be (Rows, Columns, Bands).
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

    def geometry_subset(self, geom, all_touched=False):
        """
        Return a subset of rds band array where the extent is the bounding box
        of geom and all cells outside of geom are masked.

        Parameters
        ----------
        geom : shapely geometry
            The polygon bounding the area of interest.

        Returns
        -------
        numpy.ma.MaskedArray
            A numpy masked array of shape (Rows,Columns,Bands). Cells not within
            geom will be masked as will any values that were masked in rds.
        """
        return masked_subset(self, geom, all_touched=all_touched)

    def index_at_point(self,point):
        """
        Return the matrix index for a point.

        Parameters
        ----------
        point : ogr.Geometry or shapely.geometry.point.Point
            Specifically, this is expected to be a point geometry.

        Returns
        -------
        tuple
            yOffset and xOffset of the point on the gdal image array.
        """
        if type(point)==ogr.Geometry:
            x = point.GetX()
            y = point.GetY()
        else: # We'll assume the point is a shapely.geometry.point.Point via geopandas
            x = point.x
            y = point.y

        trans = transform_dict(self.gdal_ds)

        xOffset = int( (x - trans['originX']) / trans['pixWidth'] )
        yOffset = int( (y - trans['originY']) / trans['pixHeight'] )

        return yOffset, xOffset

    def spectrum_at_point(self, point):
        """
        Return the spectra at a given point. The spectra will be returned as a
        numpy array, 1 row x N columns where N is the number of bands in the
        image.

        Parameters
        ----------
        point : ogr.Geometry or shapely.geometry.point.Point
            Specifically, this is expected to be a point geometry.

        Returns
        -------
        numpy array
            The shape depends on the number of bands in the raster.
        """
        yOffset, xOffset = self.index_at_point( point )

        band_values = []
        for bnum in range( 1, self.gdal_ds.RasterCount + 1 ):
            band = self.gdal_ds.GetRasterBand( bnum )
            data = band.ReadAsArray(xOffset, yOffset, 1, 1)[0,0]
            band_values.append( data )
        return np.array( band_values )

    def value_at_point(self, point, band_index=0):
        """
        Return the value at a point for a given band. The default is the first
        image band.

        Parameters
        ----------
        point : ogr.Geometry or shapely.geometry.point.Point
            Specifically, this is expected to be a point geometry.
        band_index : int
            Index of the image band to sample. Zero indexed (band 1 = 0). For
            single band rasters, this should be left at the default value (0).

        Returns
        -------
        scalar value
            The data type of the return depends on the data type of the raster
            band being sampled.
        """
        return self.spectrum_at_point(point)[band_index]

    def radiused_point_check(self, point, search_value=None, radius=0,
                             band_index=0, out_of_bounds=np.nan):
        """
        If `search_value` is found within `radius` of `point`, return that
        value. If not, return the mode (most common value) within radius. If
        `point` is not within `self.raster_extent`, return `out_of_bounds`. This
        function is primarily intended for checking ground truth point
        shapefiles against a thematic map raster. It is a more generous form of
        accuracy assessment that allows for some positional mismatch between
        ground truth points and the map. When `radius` is 0, this  function is
        equivalent to `RasterDS.value_at_point`.

        Parameters
        ----------
        point : ogr.Geometry or shapely.geometry.point.Point
            Specifically, this is expected to be a point geometry.
        search_value : numeric (int or float) or None
            If `None` (default), the mode of raster values within `radius` of
            `point` will be returned. If `search_value` is numeric,
            `search_value` will be returned if found within `radius`, otherwise
            the mode will be returned.
        radius : float
            The radius with which to buffer `point`. The units of this value
            depend on the projection being used.
        band_index : int
            Index of the image band to sample. Zero indexed (band 1 = 0). For
            single band rasters, this should be left at the default value (0).
        out_of_bounds : float, int, or nan (default)
            If `point` is not within `self.raster_extent`, `out_of_bounds` will
            be returned.

        Returns
        -------
        scalar value
            The data type of the return depends on the data type of the raster
            band being sampled.
        """
        if type(point) is ogr.Geometry:
            # if this is an ogr geom, make it into a shapely geom
            point = shpl.geometry.base.geom_from_wkb(point.ExportToWkb())
        if point.within(self.raster_extent):
            if radius == 0:
                retval = self.value_at_point(point, band_index=band_index)
            else:
                clsarr = self.geometry_subset(point.buffer(radius),
                                              all_touched=True)[...,band_index]
                if search_value is not None \
                and search_value in clsarr.compressed():
                    retval = search_value
                elif len(clsarr.compressed()) == 0:
                    # This means that all of the returned values are masked and
                    # therefore outside of the useful image bounds.
                    retval = out_of_bounds
                else:
                    retval = scipymode(clsarr.compressed()).mode.item()
        else:
            retval = out_of_bounds
        return retval

    def new_image_from_array(self,bandarr,outfilename=None,dtype=None,no_data_value=None):
        """
        Save an GeoTiff like `self` with data from  `bandarray`.

        Notes
        -----
        It should, in theory, be pretty easy to modify this method so that it
        can save in any format for which GDAL supports creation and writing. So
        far, I've been happy with just using GeoTiff but let me know if you want
        it to support some other format.

        Parameters
        ----------
        bandarr : np.array
            Image array of shape (Rows,Cols,Bands)
        outfilename : string, optional
            The path to the output file. If `None` (the default), then the
            `RasterDS.output_file_path` method will be used to come up with one.
            What it comes up with is dependent on the `RasterDS.overwrite`
            property.
        dtype : int, optional
            If unspecified, an attempt will be made to find a GDAL datatype
            compatible with `bandarr.dtype`. This doesn't always work. These are
            the GDAL data types::
                GDT_Unknown = 0, GDT_Byte = 1, GDT_UInt16 = 2, GDT_Int16 = 3,
                GDT_UInt32 = 4, GDT_Int32 = 5, GDT_Float32 = 6, GDT_Float64 = 7,
                GDT_CInt16 = 8, GDT_CInt32 = 9, GDT_CFloat32 = 10,
                GDT_CFloat64 = 11, GDT_TypeCount = 12
        no_data_value : int or float, optional
            The `no_data_value` to use in the output. If `None` or not specified
            an attempt will be made to use the `fill_value` of `bandarr`. If
            `bandarr` does not have a `fill_value`, the arbitrary value of -99
            will be used.

        Returns
        -------
        RasterDS
            The new GeoTiff will be written to disk and a `RasterDS` object will
            be returned for that GeoTiff.

        See Also
        --------
        output_gtif_like_img, output_gtif
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
        bandarr = np.ma.atleast_3d(bandarr)
        bandarr = np.rollaxis(bandarr,2,0)
        if not outfilename:
            outfilename = self.output_file_path()
        output_gtif_like_img(self.gdal_ds, bandarr, outfilename, no_data_value=no_data_value, dtype=dtype)
        return RasterDS(outfilename)

def transform_dict(img):
    """
    Take a raster data source and return a dictionary with geotranform values
    and keys that make sense.

    Parameters
    ----------
    img : gdal.datasource
        The image datasource from which the GeoTransform will be retrieved.

    Returns
    -------
    dict
        A dict with the geotransform values labeled.
    """
    geotrans = img.GetGeoTransform()
    ret_dict = {
            'originX':   geotrans[0],
            'pixWidth':  geotrans[1],
            'rotation1': geotrans[2],
            'originY':   geotrans[3],
            'rotation2': geotrans[4],
            'pixHeight': geotrans[5],
        }
    return ret_dict

def get_extent(gt,cols,rows):
    """
    Return list of corner coordinates from a geotransform. This code was taken
    from: http://gis.stackexchange.com/questions/57834/how-to-get-raster-corner-coordinates-using-python-gdal-bindings

    Parameters
    ----------
    gt : tuple or list
        geotransform
    cols : int
        number of columns in the dataset
    rows : int
        number of rows in the dataset

    Returns
    -------
    list of floats
        Coordinates of each corner
    """
    ext=[]
    xarr=[0,cols]
    yarr=[0,rows]

    for px in xarr:
        for py in yarr:
            x=gt[0]+(px*gt[1])+(py*gt[2])
            y=gt[3]+(px*gt[4])+(py*gt[5])
            ext.append([x,y])
#            print x,y
        yarr.reverse()
    return ext

def output_gtif(bandarr, cols, rows, outfilename, geotransform, projection, no_data_value=-99, driver_name='GTiff', dtype=GDT_Float32):
    """
    Create a geotiff with gdal that will contain all the bands represented
    by arrays within bandarr which is itself array of arrays. Expecting bandarr
    to be of shape (Bands,Rows,Columns).

    Parameters
    ----------
    bandarr : np.array
        Image array of shape (Rows,Cols,Bands)
    cols : int
        The number of columns measure in pixels. I may be able to do away with
        this parameter by just using the shape of `bandarr` to determine this
        value.
    rows : int
        The number of rows measure in pixels. I may be able to do away with this
        parameter by just using the shape of `bandarr` to determine this value.
    outfilename : string, optional
        The path to the output file. If `None` (the default), then the
        `RasterDS.output_file_path` method will be used to come up with one.
        What it comes up with is dependent on the `RasterDS.overwrite`
        property.
    geotransform : tuple or list
        The geotransform will determine how the elements of `bandarr` are
        spatially destributed. The elements of the geotransform are as follows::
            adfGeoTransform[0] /* top left x */
            adfGeoTransform[1] /* w-e pixel resolution */
            adfGeoTransform[2] /* rotation, 0 if image is "north up" */
            adfGeoTransform[3] /* top left y */
            adfGeoTransform[4] /* rotation, 0 if image is "north up" */
            adfGeoTransform[5] /* n-s pixel resolution */
    projection : string
        The string should be a projection in OGC WKT or PROJ.4 format.
    no_data_value : int or float, optional
        The `no_data_value` to use in the output. If `None` or not specified
        an attempt will be made to use the `fill_value` of `bandarr`. If
        `bandarr` does not have a `fill_value`, the arbitrary value of -99
        will be used.
    driver_name : string, optional
        The name of the GDAL driver to use. This will determine the format of
        the output. For GeoTiff output, use the default value ('GTiff').
    dtype : int, optional
        If unspecified, an attempt will be made to find a GDAL datatype
        compatible with `bandarr.dtype`. This doesn't always work. These are
        the GDAL data types::
            GDT_Unknown = 0, GDT_Byte = 1, GDT_UInt16 = 2, GDT_Int16 = 3,
            GDT_UInt32 = 4, GDT_Int32 = 5, GDT_Float32 = 6, GDT_Float64 = 7,
            GDT_CInt16 = 8, GDT_CInt32 = 9, GDT_CFloat32 = 10,
            GDT_CFloat64 = 11, GDT_TypeCount = 12

    Returns
    -------
    Nothing
        This method just writes a file. It has no return.
    """
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
    """
    Create a geotiff with attributes like the one passed in but make the
    values and number of bands as in bandarr.

    Parameters
    ----------
    img : GDAL data source
        This is a image to use as a template for the new GeoTiff. The new image
        will use the extent, projection, and geotransform from `img`.
    bandarr : np.array
        Image array of shape (Rows,Cols,Bands)
    outfilename : string, optional
        The path to the output file. If `None` (the default), then the
        `RasterDS.output_file_path` method will be used to come up with one.
        What it comes up with is dependent on the `RasterDS.overwrite`
        property.
    no_data_value : int or float, optional
        The `no_data_value` to use in the output. If `None` or not specified
        an attempt will be made to use the `fill_value` of `bandarr`. If
        `bandarr` does not have a `fill_value`, the arbitrary value of -99
        will be used.
    dtype : int, optional
        If unspecified, an attempt will be made to find a GDAL datatype
        compatible with `bandarr.dtype`. This doesn't always work. These are
        the GDAL data types::
            GDT_Unknown = 0, GDT_Byte = 1, GDT_UInt16 = 2, GDT_Int16 = 3,
            GDT_UInt32 = 4, GDT_Int32 = 5, GDT_Float32 = 6, GDT_Float64 = 7,
            GDT_CInt16 = 8, GDT_CInt32 = 9, GDT_CFloat32 = 10,
            GDT_CFloat64 = 11, GDT_TypeCount = 12

    Returns
    -------
    Nothing
        This method just writes a file. It has no return.
    """
    cols = img.RasterXSize
    rows = img.RasterYSize
    geotransform = img.GetGeoTransform()
    projection = img.GetProjection()
    output_gtif(bandarr, cols, rows, outfilename, geotransform, projection, no_data_value, driver_name='GTiff', dtype=dtype)
