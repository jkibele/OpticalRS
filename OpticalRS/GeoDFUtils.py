# -*- coding: utf-8 -*-
"""
GeoDFUtils
==========

Methods to work with GeoDataFrames from GeoPandas.
"""

import geopandas as gpd
from osgeo import osr
import numpy as np
from RasterDS import RasterDS

class RasterShape(object):
    """
    Provide a object to make it more convenient to deal with a raster and a
    shapefile.

    Parameters
    ----------
    rds : RasterDS or string or QGIS raster layer
        If `rds` is a RasterDS, then yay. If not, try to make a RasterDS out of
        whatever `rds` is. RasterDS can take a filepath to a GDAL compatible
        raster (like a GeoTiff) or a QGIS raster layer.
    shp : GeoPandas.GeoDataFrame or string (filepath to a shapefile)
        If `shp` is a GeoDataFrame, that will be used. Otherwise `shp` will be
        assumed to be a filepath string and will be handed to
        GeoPandas.read_file().
    gdf_query : string or None
        A string for the pandas query method:
        http://pandas.pydata.org/pandas-docs/version/0.17.0/generated/pandas.DataFrame.query.html
        Where a single geometry is used in a method: If `None` is passed, the
        first geometry in the GeoDataFrame will be used. If a query is passed,
        the first geometry in the query results will be used.
    """
    def __init__(self, rds, shp, gdf_query=None):
        if type(rds).__name__ == 'RasterDS':
            self.rds = rds
        else:
            self.rds = RasterDS(rds)

        if type(shp).__name__ == 'GeoDataFrame':
            self.gdf = shp
        else:
            self.gdf = gpd.read_file(shp)

    def geometry_subset(self, gdf_query=None, all_touched=False):
        if gdf_query == None:
            geom = self.gdf.ix[0].geometry
        else:
            geom = gdf.query(gdf_query).ix[0].geometry

        return self.rds.geometry_subset(geom, all_touched=all_touched)

def compare_raster(gdf, column, rds, radius=0, generous=False,
                   band_index=0, out_of_bounds=np.nan):
    """
    Compare habitat codes in `gdf` with codes in corresponding locations of
    a raster habitat map (`rds`). This can be an exact point to point
    comparison (when `radius`=0) or can be more forgiving. When `radius`>0
    and `generous` is `False`, the mode (most common) value within `radius`
    of each point will be returned. When `radius`>0 and `generous` is True,
    ground truth habitat codes will be returned if found within `radius` of
    each point, and the mode will be returned if not.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        A geopandas geo data frame of a point shapefile. Projection much match
        that of `rds`.
    column : string
        The name of the column in `gdf` that contains the habitat codes. These
        must be numeric codes that match the codes in `rds`.
    rds : OpticalRS.RasterDS
        The habitat map (or whatever raster) you want to compare to the
        `GroundTruthShapefile` (self). The projection of this raster must
        match the projection of the `GroundTruthShapefile`. If it doesn't
        match, you might get results but they'll be wrong.
    radius : float
        The radius with which to buffer `point`. The units of this value
        depend on the projection being used.
    generous : boolean
        If False (default), mode will be returned. If True, habitat code will be
        returned if within `radius`. See function description for more info.
    band_index : int
        Index of the image band to sample. Zero indexed (band 1 = 0). For
        single band rasters, this should be left at the default value (0).
    out_of_bounds : float, int, or nan (default)
        If `point` is not within `self.raster_extent`, `out_of_bounds` will
        be returned.

    Returns
    -------
    pandas Series
        The values from `rds` that correspond to each point in `gdf`.
    """
    if generous:
        rcheck = lambda row: rds.radiused_point_check(row.geometry,
                                                      radius=radius,
                                                      search_value=row[column],
                                                      band_index=band_index,
                                                      out_of_bounds=out_of_bounds)
    else:
        rcheck = lambda row: rds.radiused_point_check(row.geometry,
                                                      radius=radius,
                                                      search_value=None,
                                                      band_index=band_index,
                                                      out_of_bounds=out_of_bounds)
    return gdf.apply(rcheck, axis=1)


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
    col_names : list, array-like, or string
        Names for the columns to be created in the output GeoDataFrame. The
        length must be the same as the number of bands in `rds`. If `None`, then
        `rds.band_names` will be used. This will generally mean that the columns
        will be called 'band1', 'band2', etc. A string can be used with `rds`
        is a single band raster.

    Returns
    -------
    GeoDataFrame
        A copy of `gdf` with additional columns populated with values from `rds`.
    """
    if col_names==None:
        col_names = rds.band_names
    elif type(col_names) == str:
        col_names = [col_names]
    nbands = rds.gdal_ds.RasterCount
    winsize = 1 + win_radius * 2
    rast_poly = rds.raster_extent
    gt = rds.gdal_ds.GetGeoTransform()
    subst = gdf[gdf.within(rast_poly)]
    outdf = subst.copy()
    pointstatlist = []
    if winsize == 1:
        xs = outdf.geometry.map(lambda g: map_to_pix(g.x, g.y, gt)[0]).as_matrix()
        ys = outdf.geometry.map(lambda g: map_to_pix(g.x, g.y, gt)[1]).as_matrix()
        bandarr = rds.band_array
        outarr = bandarr[ys, xs]
        banddf = gpd.GeoDataFrame(outarr, columns=col_names, index=outdf.index)
    else:
        for i in outdf.index:
            geom = outdf.ix[i].geometry
            if rast_poly.contains(geom):
                px, py = map_to_pix(geom.x, geom.y, gt)
                print px, py
                pointarr = rds.band_array_subset(px-win_radius,py-win_radius,winsize,winsize).reshape(-1,nbands)
                pointstat = stat_func(pointarr,axis=0)
                pointstatlist.append(pointstat)
        banddf = gpd.GeoDataFrame(np.array(pointstatlist), columns=col_names, index=outdf.index)
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
    px = np.int((x - gt[0]) / gt[1])
    py = np.int((y - gt[3]) / gt[5])
    return px, py
