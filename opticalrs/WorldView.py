"""
WorldView
=========

This module provides an interface for the other (more general) functions in OpticalRS that is tailored for WorldView
8 band imagery (both WorldView 2 and 3). When using OpticalRS with WorldView imagery, you should use this interface.
Unlike earlier versions of OpticalRS, this module uses Rasterio to deal with file operations. This should be a big
improvement over the old gdal based methods.

"""


import rasterio as rio
import numpy as np
from opticalrs.LandMasking import mask_land
from opticalrs.MSExposure import equalize_adapthist


class WorldViewObj(object):
    def __init__(self, img_path):
        self.rio_ds = rio.open(img_path)
        self.band_array = self._read_band_array()

    def _read_band_array(self):
        """
        Return a numpy array in the (Rows, Columns, Bands) shape used by OpticalRS.

        Returns
        -------
        band_arr : numpy.ndarray
            The image array in (Rows, Columns, Bands) shape
        """
        rio_arr = self.rio_ds.read()
        band_arr = np.moveaxis(rio_arr, 0, -1)
        return band_arr

    def mask_land(self):
        mask = mask_land(self.band_array, nir_threshold=100, conn_threshold=1000, structure=None)
        return mask

    def ocean_equalized_rgb(self):
        """
        Return an RGB image that's based off of adaptive histogram equalization of the first 3 bands.

        Returns
        -------

        """
        masked = self.mask_land()
        equalized = equalize_adapthist(masked[..., [3, 2, 1]], clip_limit=0.02)
        return equalized

