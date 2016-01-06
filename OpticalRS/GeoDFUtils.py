# -*- coding: utf-8 -*-
"""
GeoDFUtils
==========

Methods to work with GeoDataFrames from GeoPandas.
"""

import geopandas as gpd
from osgeo import osr
import numpy as np

def point_sample_raster(gdf, rds, win_radius=0, stat_func=np.mean, col_names=None):
    """
    Sample all bands of a raster data set `rds`, and attribute each point of
    a geopandas GeoDataFrame `gdf`. A radius value can be supplied to sample
    the mean of a square window rather than individual pixels.

    Parameters
    ----------
    gdf : GeoPandas.GeoDataFrame
        A point GeoDataFrame. Point locations will be used to sample the raster.
    rds : OpticalRS.RasterDS.RasterDS
        The raster dataset to be sampled.
    win_radius : int
        The numper of pixels to sample around each point in `gdf`. If, for
        instance, `win_radius=2` then a 5x5 pixel window centered on each point
        will be sampled. The default value of 0 means that just a single pixel
        will be sampled.
    stat_fun : function
        This can be any function that takes an array and an axis and returns
        a numeric value. In practice, this will probably be numpy statistical
        functions like `np.mean`, `np.median`, `np.std`, `np.var`, etc.
    col_names : list or array-like
        Names for the columns to be created in the output GeoDataFrame. The
        length must be the same as the number of bands in `rds`. If `None`, then
        `rds.band_names` will be used. This will generally mean that the columns
        will be called 'band1', 'band2', etc.

    Returns
    -------
    GeoDataFrame
        A copy of `gdf` with additional columns populated with values from `rds`.
    """
    if col_names==None:
        col_names = rds.band_names
    nbands = rds.gdal_ds.RasterCount
    winsize = 1 + win_radius * 2
    rast_poly = rds.raster_extent
    gt = rds.gdal_ds.GetGeoTransform()
    outdf = gdf.copy()
    pointstatlist = []
    for i in outdf.index:
        geom = outdf.ix[i].geometry
        if rast_poly.contains(geom):
            px, py = map_to_pix(geom.x, geom.y, gt)
            pointarr = rds.band_array_subset(px-win_radius,py-win_radius,winsize,winsize).reshape(-1,nbands)
            pointstat = stat_func(pointarr,axis=0)
            pointstatlist.append(pointstat)
    banddf = gpd.GeoDataFrame(np.array(pointstatlist), columns=col_names)
    return outdf.join(banddf)

def reproject_coords(coords,src_srs,tgt_srs):
    """
    Reproject a list of x,y coordinates. Code borrowed from:
    http://gis.stackexchange.com/questions/57834/how-to-get-raster-corner-coordinates-using-python-gdal-bindings

    Parameters
    ----------
    geom : tuple or list
        List of [[x,y],...[x,y]] coordinates
    src_srs : osr.SpatialReference
        OSR SpatialReference object of source
    tgt_srs : osr.SpatialReference
        OSR SpatialReference object target

    Returns
    -------
    list
        Transformed [[x,y],...[x,y]] coordinates

    Notes
    -----
    Usage:
        src_srs=osr.SpatialReference()
        src_srs.ImportFromWkt(ds.GetProjection())
        tgt_srs = src_srs.CloneGeogCS()

        geo_ext=ReprojectCoords(ext,src_srs,tgt_srs)
    """
    trans_coords=[]
    transform = osr.CoordinateTransformation( src_srs, tgt_srs)
    for x,y in coords:
        x,y,z = transform.TransformPoint(x,y)
        trans_coords.append([x,y])
    return trans_coords

def map_to_pix(x, y, gt):
    """
    Convert from map to pixel coordinates. Works for geotransforms
    with no rotation.

    Parameters
    ----------
    x : float
        x coordinate in map units
    y : float
        y coordinate in map units
    gt : gdal geotransform (list)
        See http://www.gdal.org/gdal_datamodel.html

    Returns
    -------
    px : int
        x coordinate in pixel index units
    py : int
        y coordinate in pixel index units
    """
    px = int((x - gt[0]) / gt[1])
    py = int((y - gt[3]) / gt[5])
    return px, py
