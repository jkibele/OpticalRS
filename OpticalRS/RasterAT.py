# -*- coding: utf-8 -*-
"""
RasterAT
========

The `RAT` object will subclass `RasterDS` and add methods for handling Raster
Attribute Tables. The idea is to read and write RATs using GDAL but to represent
and manipulate them using pandas.
"""

from RasterDS import RasterDS
from osgeo import gdal
import pandas as pd
import numpy as np

f_names = ['Name', 'PixelCount', 'ClassNumber', 'Red', 'Blue', 'Green', 'Alpha']
f_use = [gdal.GFU_Name, gdal.GFU_PixelCount, gdal.GFU_MinMax, gdal.GFU_Red,
         gdal.GFU_Blue, gdal.GFU_Green, gdal.GFU_Alpha]
f_type = [gdal.GFT_String, gdal.GFT_Integer, gdal.GFT_Integer, gdal.GFT_Integer,
          gdal.GFT_Integer, gdal.GFT_Integer, gdal.GFT_Integer]
f_use_d = dict(zip(f_names, f_use))
f_type_d = dict(zip(f_names, f_type))

class RAT(RasterDS):
    def __init__(self, rlayer, overwrite=False):
        RasterDS.__init__(self, rlayer, overwrite=overwrite)
        self.ratdf = self.__read_rat()
        if self.ratdf == None:
            self.ratdf = self.__create_rat()

    @property
    def unique_values(self):
        return np.unique(self.band_array.compressed())

    def __create_rat(self):
        """
        Create a default pandas RAT for a raster that doesn't already have one.
        """
        # columns and index
        cols = f_type_d.keys()
        cols.remove('ClassNumber') #This is going to be the index
        df = pd.DataFrame(columns=cols, index=self.unique_values)
        # PixelCount
        bins = np.append(self.unique_values, self.unique_values.max() +1)
        pcnt = np.histogram(self.band_array.compressed(), bins=bins)[0]
        df.PixelCount = pcnt

        return df

    def __read_rat(self):
        """
        Read gdal rat if there is one and return it as a pandas dataframe.
        Return `None` if there is no rat.
        """
        return None

def dtype_map(typ):
    if type(typ) == np.dtype:
        # map numpy to GFT
        pass
    else:
        #map GFT to numpy
        if typ == gdal.GFT_Integer:
            return np.dtype('int32')
        elif typ == gdal.GFT_Real:
            return np.dtype('float32')
        else:
            return np.dtype('string')

def df_to_gdal_rat(df):
    if 'ClassNumber' not in df.columns:
        df['ClassNumber'] = df.index
    rat = gdal.RasterAttributeTable()
    rat.SetRowCount(len(df))
    colnum = {}
    for num, col in enumerate(f_names):
        rat.CreateColumn(col, f_type_d[col], f_use_d[col])
        rat.WriteArray(df[col], num)

    return rat
