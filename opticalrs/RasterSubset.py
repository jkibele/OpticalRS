# -*- coding: utf-8 -*-
"""
RasterSubset
============

These are methods for subsetting a raster to get just the cells within a vector
geometry.  Much of this code is derived from the python `rasterstats package
<https://github.com/perrygeo/python-raster-stats>`_ and is dependent on some
utilities from that package. I think this code could be rewritten to remove the
dependency on having rasterstats installed but I'm not sure when I'll get around
to that.
"""
#import numpy as np
from osgeo import gdal, ogr, osr
from rasterstats.utils import bbox_to_pixel_offsets, shapely_to_ogr_type

def mask_from_geom( geom, rds, band_num=1, epsg=32760, nodata_value=None,
                    all_touched=False, full_extent=False ):
    """
    Return a binary mask to mask off everything outside of geom.

    Parameters
    ----------
    geom : shapely.geometry
        Cells inside this geometry will be `False`. Cells outside will be `True`
    rds : RasterDS
        The raster dataset to create a mask for.
    full_extent : bool
        If `True`, return a mask that's the full extent of `rds`. If `False`
        (default), return a mask that's the extent of `geom`.

    Returns
    -------
    numpy boolean array
        An array with `True` outside `geom` and `False` inside.
    """
    spatial_ref = osr.SpatialReference()
    spatial_ref.ImportFromEPSG(epsg)
    mem_drv = ogr.GetDriverByName('Memory')
    driver = gdal.GetDriverByName('MEM')
    rb = rds.gdal_ds.GetRasterBand(band_num)
    rgt = rds.gdal_ds.GetGeoTransform()
    rsize = (rds.gdal_ds.RasterXSize, rds.gdal_ds.RasterYSize)

    if nodata_value is not None:
        nodata_value = float(nodata_value)
        rb.SetNoDataValue(nodata_value)
    else:
        nodata_value = rb.GetNoDataValue()
    # Point and MultiPoint don't play well with GDALRasterize
    # convert them into box polygons the size of a raster cell
#    buff = rgt[1] / 2.0
#    if geom.type == "MultiPoint":
#        geom = MultiPolygon([box(*(pt.buffer(buff).bounds))
#                            for pt in geom.geoms])
#    elif geom.type == 'Point':
#        geom = box(*(geom.buffer(buff).bounds))

    ogr_geom_type = shapely_to_ogr_type(geom.type)

    if full_extent:
        geom_bounds = list(rds.raster_extent.bounds)
    else:
        geom_bounds = list(geom.bounds)

    # calculate new pixel coordinates of the feature subset
    src_offset = bbox_to_pixel_offsets(rgt, geom_bounds, rsize)

    new_gt = (
        (rgt[0] + (src_offset[0] * rgt[1])),
        rgt[1],
        0.0,
        (rgt[3] + (src_offset[1] * rgt[5])),
        0.0,
        rgt[5]
    )

    if src_offset[2] <= 0 or src_offset[3] <= 0:
        # we're off the raster completely, no overlap at all
        # so there's no need to even bother trying to calculate
        return None
    else:
        # use feature's source extent and read directly from source
        # fastest option when you have fast disks and well-indexed raster
        # advantage: each feature uses the smallest raster chunk
        # disadvantage: lots of disk reads on the source raster
        #src_array = rb.ReadAsArray(*src_offset)

        # Create a temporary vector layer in memory
        mem_ds = mem_drv.CreateDataSource('out')
        mem_layer = mem_ds.CreateLayer('out', spatial_ref, ogr_geom_type)
        ogr_feature = ogr.Feature(feature_def=mem_layer.GetLayerDefn())
        ogr_geom = ogr.CreateGeometryFromWkt(geom.wkt)
        ogr_feature.SetGeometryDirectly(ogr_geom)
        mem_layer.CreateFeature(ogr_feature)

        # Rasterize it
        rvds = driver.Create('rvds', src_offset[2], src_offset[3], 1, gdal.GDT_Byte)
        rvds.SetGeoTransform(new_gt)

        if all_touched:
            gdal.RasterizeLayer(rvds, [1], mem_layer, None, None, burn_values=[1], options = ['ALL_TOUCHED=True'])
        else:
            gdal.RasterizeLayer(rvds, [1], mem_layer, None, None, burn_values=[1], options = ['ALL_TOUCHED=False'])
        rv_array = rvds.ReadAsArray()

        # Mask the source data array with our current feature
        # we take the logical_not to flip 0<->1 to get the correct mask effect
        # we also mask out nodata values explictly
#         masked = np.ma.MaskedArray(
#             src_array,
#             mask=np.logical_or(
#                 src_array == nodata_value,
#                 np.logical_not(rv_array)
#             )
#         )

#         return masked
#         return np.logical_or( src_array == nodata_value, np.logical_not( rv_array ) )
        return ~rv_array.astype('bool')

def masked_subset( rds, geom, all_touched=False ):
    """
    Return a subset of rds where the extent is the bounding box of geom and all
    cells outside of geom are masked.

    Parameters
    ----------
    rds : OpticalRS.RasterDS
        The raster dataset that you want to subset
    geom : shapely geometry
        The polygon bounding the area of interest.

    Returns
    -------
    numpy.ma.MaskedArray
        A numpy masked array of shape (Rows,Columns,Bands). Cells not within
        geom will be masked as will any values that were masked in rds.
    """
    # calculate new pixel coordinates of the feature subset
    src_offset = bbox_to_pixel_offsets(rds.gdal_ds.GetGeoTransform(),\
                                       list(geom.bounds), \
                                       (rds.gdal_ds.RasterXSize, rds.gdal_ds.RasterYSize) )
    #return src_offset
    barr = rds.band_array_subset( *src_offset )
    nbands = barr.shape[-1]
    band_num = 1
    epsg = rds.epsg
    rb = rds.gdal_ds.GetRasterBand(1)
    nodata_value = rb.GetNoDataValue()

    geom_mask = mask_from_geom( geom, rds, band_num=band_num, epsg=epsg, \
                                nodata_value=nodata_value, all_touched=all_touched )

    for bn in range( nbands ):
        barr[:,:,bn].mask = barr[:,:,bn].mask | geom_mask

    return barr
