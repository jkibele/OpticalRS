# -*- coding: utf-8 -*-
"""
RasterAT
========

The `RAT` object will subclass `RasterDS` and add methods for handling Raster
Attribute Tables. The idea is to read and write RATs using GDAL but to represent
and manipulate them using pandas.
"""

from RasterDS import RasterDS
from GeoDFUtils import point_sample_raster
from GroundTruthShp import GroundTruthShapefile
import geopandas as gpd
from osgeo import gdal
import pandas as pd
import numpy as np
from scipy.ndimage import measurements

f_names = ['Name', 'PixelCount', 'ClassNumber', 'Red', 'Blue', 'Green', 'Alpha']
f_use = [gdal.GFU_Name, gdal.GFU_PixelCount, gdal.GFU_MinMax, gdal.GFU_Red,
         gdal.GFU_Blue, gdal.GFU_Green, gdal.GFU_Alpha]
f_type = [gdal.GFT_String, gdal.GFT_Integer, gdal.GFT_Integer, gdal.GFT_Integer,
          gdal.GFT_Integer, gdal.GFT_Integer, gdal.GFT_Integer]
f_use_d = dict(zip(f_names, f_use))
f_type_d = dict(zip(f_names, f_type))

class RAT(RasterDS):
    def __init__(self, rlayer, overwrite=True):
        RasterDS.__init__(self, rlayer, overwrite=overwrite)
        self.ratdf = self.__get_or_create_rat()

    def __open_gdal_ds(self):
        self._RasterDS__open_gdal_ds()

    def _erase_rat(self):
        """
        Are you sure you want to do this? This doesn't fully work. It leaves
        behind the *.tif.aux.xml file that, apparently, contains the actual RAT.
        """
        band = self.gdal_ds.GetRasterBand(1)
        band.SetDefaultRAT(None)
        self.gdal_ds = None
        self.gdal_ds = self._RasterDS__open_gdal_ds()

    @property
    def unique_values(self):
        return np.unique(self.band_array.compressed())

    def save_rat(self, df=None):
        """
        Write the RAT to the GDAL file. For now, we're just assuming a single
        band.
        """
        if df is None:
            df = self.ratdf
        else:
            self.ratdf = df

        if not self.overwrite:
            raise ValueError("RasterAT object is not set to allow overwriting of its file.")
        band = self.gdal_ds.GetRasterBand(1)
        grat = df_to_gdal_rat(df)
        ret = band.SetDefaultRAT(grat)
        if ret == gdal.CE_None:
            self.gdal_ds = None
            self.gdal_ds = self._RasterDS__open_gdal_ds()
            return True
        else:
            return False

    def __create_rat(self):
        """
        Create a default pandas RAT for a raster that doesn't already have one.
        """
        # columns and index
        cols = list(f_names)
        cols.remove('ClassNumber') #This is going to be the index
        df = pd.DataFrame(columns=cols, index=self.unique_values)
        # PixelCount
        bins = np.append(self.unique_values, self.unique_values.max() +1)
        pcnt = np.histogram(self.band_array.compressed(), bins=bins)[0]
        df.PixelCount = pcnt
        # Colors
        df[['Red','Green','Blue','Alpha']] = np.random.randint(0, 255, (len(df), 4))
        df.index.name = 'ClassNumber'
        return df

    def __read_rat(self):
        """
        Read gdal rat if there is one and return it as a pandas dataframe.
        Return `None` if there is no rat.
        """
        band = self.gdal_ds.GetRasterBand(1)
        grat = band.GetDefaultRAT()
        if grat is not None:
            return gdal_rat_to_df(grat)
        else:
            return None

    def __get_or_create_rat(self):
        readrat = self.__read_rat()
        if readrat is None:
            return self.__create_rat()
        else:
            return readrat

    def properties_df(self, img, func=np.mean, prefix=None, postfix='_b',
                      colnames=None):
        """
        Sample values from `img` for each segment (a.k.a. class or class number)
        in the RAT. `func` is used on the `img` pixels to produce a single value
        for each segment.
        """
        if isinstance(img, np.ndarray):
            img = img.copy()
        elif isinstance(img, RasterDS):
            img = img.band_array
        else:
            img = RasterDS(img).band_array
        img = np.atleast_3d(img)
        labels = self.band_array.squeeze()
        nbands = img.shape[-1]
        if colnames == None:
            if prefix == None:
                try:
                    prefix = func.__name__
                except AttributeError:
                    prefix = ''
            colnames = [prefix+postfix+str(i+1) for i in range(nbands)]
        ddict = {}
        for bnum in range(nbands):
            band = img[...,bnum]
            coln = colnames[bnum]
            ind = self.ratdf.index.to_series().as_matrix()
            ddict[coln] = band_label_properties(labels, band, ind, func)
        newdf = pd.DataFrame(ddict, columns=colnames, index=self.ratdf.index)
        return newdf

    def column_array(self, cols, df=None, out_mask=None, default=0):
        """
        Produce an image array from values in the RAT. Instead of the integer
        `ClassNumber` values from the segments, the image array will contain
        values from the specified column. When multiple columns (`cols`) are
        passed, the result will be a multi-band image array.

        Parameters
        ----------
        cols : string or interable of strings
            The dataframe columns to pull values from. Each string needs to
            correspond to a column in the RAT or, if `df` has been passed,
            if must be a column in `df`.
        df : pandas dataframe
            The dataframe from which raster values will be pulled. If `None`
            (default), the dataframe representation of the RAT will be used.

        Returns
        -------
        numpy array
            An image array of size (Rows, Columns, nBands) where nBands =
            `len(cols)`.
        """
        if type(cols) == str:
            cols = [cols]
        if type(df) == type(None):
            df = self.ratdf
        lblarr = self.band_array.squeeze()
        noutbands = len(cols)
        outshape = lblarr.shape + (noutbands,)
        outarr = np.ma.zeros(outshape)
        idxset = set(df.index)

        for i, col in enumerate(cols):
            # outarr[...,i] = np.vectorize(df[col].to_dict().get)(lblarr)
            bnd = outarr[:,:,i]
            for idx in idxset:
                bnd[lblarr==idx] = df.loc[idx, col]

        if out_mask is not None:
            out_mask = np.repeat(np.atleast_3d(out_mask.astype('bool')), noutbands, axis=2)
            outarr.mask = out_mask


        # for i, col in enumerate(cols):
        #     bnd = outarr[...,i] # view, not copy so changes to bnd will
        #                         # be reflected in outarr, ...I think.
        #     for cn in df.index.astype('int'):
        #         idx = np.index_exp[bndarr==cn]
        #         bnd[idx] = df.loc[cn, col]
        #     if np.ma.is_masked(bndarr):
        #         bnd.mask = bndarr.mask

        # if np.ma.is_masked(bndarr):
        #     outmask = np.repeat(np.atleast_3d(bndarr), noutbands, axis=2)
        #     outarr.mask = outmask

        ## This worked but it was very slow.
        # outarr = np.repeat(self.band_array.astype(float), len(cols), axis=2)
        # for i, col in enumerate(cols):
        #     def class_map(classnum):
        #         if classnum in df.index:
        #             return df.loc[int(classnum), col]
        #         else:
        #             return np.nan
        #     vclass_map = np.vectorize(class_map)
        #     outarr[...,i] = vclass_map(outarr[...,i])
        return outarr

    def bps_habitat_df(self, bps_shp, hab_cols=None):
        """
        Build a data frame with aggregated habitat values from a BPS shapefile.
        If `hab_cols` is not specified, all columns of `bps_shp` that start with
        a capital letter will be assumed to represent habitats.

        Parameters
        ----------
        bps_shp : string (file path), GeoDataFrame, or GroundTruthShapefile
            Whichever type is passed, will be used to attempt to creat a
            geodataframe. This needs to be a point shapefile. Typically, one
            created by [Benthic Photo Survey](http://openresearchsoftware.metajnl.com/articles/10.5334/jors.104/)
        hab_cols : list of strings
            These are the names of the habitat columns in the BPS file. If left
            as `None`, all captiolized columns with numeric values will be
            assumed to be habitats.

        Returns
        -------
        pandas DataFram
            A data frame with habitat columns and a point count column. The
            point count column contains the number of BPS points in each
            segment (ClassNumber). The habitat columns contain the average of
            the habitat values for the BPS points in the corresponding segment.
        """
        if type(bps_shp) == gpd.geodataframe.GeoDataFrame:
            bpsdf = bps_shp
        elif type(bps_shp) == GroundTruthShapefile:
            bpsdf = bps_shp.geo_data_frame
        else:
            bpsdf = gpd.read_file(bps_shp)

        if hab_cols is None:
            # This will create a list of numeric columns with names that
            # are capitolized. This works because I tend to set up my BPS
            # habitat names with capitolized names. ...but it won't work
            # right if you haven't set up BPS that way.
            hab_cols = [ c for c in bpsdf.select_dtypes(include=[np.number]).columns if c[0].isupper() ]
        # if habitats haven't been assigned, we don't want to use those rows.
        bpsdf.dropna('rows', subset=hab_cols, inplace=True)
        # get the segment number for each point
        bpsdf = point_sample_raster(bpsdf, self, col_names=['ClassNumber'])
        # group the BPS point by the segment they're in
        grp = bpsdf[hab_cols + ['ClassNumber']].groupby('ClassNumber')
        # count how many points are in each segment
        pntcnt = grp.count().ix[:,0]
        pntcnt.name = 'point_count'
        # average the hab columns for each segment
        habmeans = grp.mean()
        seghabs = habmeans.join(pntcnt)
        return seghabs


def band_label_properties(labels, band, ind=None, func=np.mean,
                          outdtype=np.float, default=0.0):
    if type(ind) == type(None):
        ind = np.unique(labels.compressed())
    proparr = measurements.labeled_comprehension(band, labels, ind, func, outdtype, default)
    return pd.Series(proparr, index=ind)

def dtype_map(typ):
    if type(typ) == np.dtype:
        # map numpy to GFT
        if typ.kind in ['i', 'u']:
            return gdal.GFT_Integer
        elif typ.kind in ['f', 'c']:
            return gdal.GFT_Real
        else:
            return gdal.GFT_String
    else:
        #map GFT to numpy
        if typ == gdal.GFT_Integer:
            return np.dtype('int32')
        elif typ == gdal.GFT_Real:
            return np.dtype('float32')
        else:
            # I'm using object instead of string so I can represent np.NaN
            return np.dtype('O')

def df_to_gdal_rat(df):
    df = df.copy()
    if 'ClassNumber' not in df.columns:
        df['ClassNumber'] = df.index

    rat = gdal.RasterAttributeTable()
    rat.SetRowCount(len(df))
    for num, col in enumerate(df.columns):
        gftype = dtype_map(df[col].dtype)
        if col in f_names:
            usetype = f_use_d[col]
        else:
            usetype = gdal.GFU_Generic
        # have to call str(col) because CreateColumn can't take unicode
        rat.CreateColumn(str(col), gftype, usetype)
        rat.WriteArray(df[col].tolist(), num)
    return rat

def gdal_rat_to_df(grat):
    dfdict = {}
    idx = None
    for colnum in range(grat.GetColumnCount()):
        colname = grat.GetNameOfCol(colnum)
        coldtype = dtype_map(grat.GetTypeOfCol(colnum))
        coluse = grat.GetUsageOfCol(colnum)
        # print "{}: {}".format(colname, coldtype)
        if coluse == gdal.GFU_MinMax:
            idx = grat.ReadAsArray(colnum)
        elif coldtype == np.dtype('O'):
            # fix 'nan' strings
            arr = grat.ReadAsArray(colnum).astype('O', copy=False)
            arr[arr=='nan'] = np.NaN
            dfdict[colname] = arr
        else:
            dfdict[colname] = grat.ReadAsArray(colnum)
    # I want to order the columns in a sensible way
    stdcols = list(f_names)
    stdcols.remove('ClassNumber')
    customcols = [c for c in dfdict.keys() if c not in stdcols]
    colord = customcols
    colord.extend(stdcols)
    df = pd.DataFrame(dfdict, index=idx)[colord]
    df.index.name = "ClassNumber"
    return df
