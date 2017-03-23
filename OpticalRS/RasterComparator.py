# -*- coding: utf-8 -*-
"""
RasterComparator
========

The `RasterComparator` will offer ways to easily compare to single band rasters
of continuous variables. The specific use case for which it is designed is the
comparison of depth rasters.
"""

# from GeoDFUtils import RasterShape
from RasterDS import RasterDS
from ArrayUtils import equalize_array_masks
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

class RasterComparator(object):
    """
    An object to easily compare to single band rasters of continuous variables.
    The specific use case for which it is designed is the comparison of depth
    rasters. Input rasters must be of same size and resolution, but can have
    unequal masks. Only pixels unmasked in both images will be compared.

    Parameters
    ----------
    pred_rds : OpticalRS.RasterDS (or acceptable input for RasterDS)
        The predicted or estimated values you'd like to validate against true
        or reference values.
    true_rds : OpticalRS.RasterDS (or acceptable input for RasterDS)
        The values you want to use as reference/true values
    pred_range : tuple
        The depth range to which predition values should be limited
    true_range : tuple
        The depth range to which true values should be limited

    Returns
    -------
    RasterComparator
        An object with RMSE and regression methods.
    """
    def __init__(self, pred_rds, true_rds, pred_range=None, true_range=None,
                 pred_name='Predicted', true_name='True'):
        self.pred_range = pred_range
        self.true_range = true_range
        if type(pred_rds).__name__ == 'RasterDS':
            self.pred_rds = pred_rds
        else:
            self.pred_rds = RasterDS(pred_rds)

        if type(true_rds).__name__ == 'RasterDS':
            self.true_rds = true_rds
        else:
            self.true_rds = RasterDS(true_rds)
        self.pred_name = pred_name
        self.true_name = true_name
        self._set_arrays()

    def copy(self, pred_range="unchanged", true_range="unchanged"):
        if pred_range is "unchanged":
            pred_range = self.pred_range
        if true_range is "unchanged":
            true_range = self.true_range
        return RasterComparator(self.pred_rds, self.true_rds, pred_range, true_range)

    def set_pred_range(self, pred_range):
        self.pred_range = pred_range
        self._set_arrays()

    def set_true_range(self, true_range):
        self.true_range = true_range
        self._set_arrays()

    def _set_arrays(self):
        # get prediction and true arrays
        parr, tarr = self.pred_rds.band_array.squeeze(), self.true_rds.band_array.squeeze()
        if type(self.pred_range).__name__ != 'NoneType':
            parr = np.ma.masked_outside(parr, *self.pred_range)
        if type(self.true_range).__name__ != 'NoneType':
            tarr = np.ma.masked_outside(tarr, *self.true_range)
        parr, tarr = equalize_array_masks(parr, tarr)
        self.pred_arr = parr
        self.true_arr = tarr
        return True

    @property
    def dataframe(self):
        dct = {
        'pred' : self.pred_arr.compressed(),
        'true' : self.true_arr.compressed()
        }
        return pd.DataFrame.from_dict(dct)

    @property
    def rmse(self):
        df = self.dataframe
        errs = (df.pred - df.true)
        return np.sqrt(np.square(errs).sum() / float(errs.count()))

    @property
    def rsquared(self):
        x,y = self.pred_arr.compressed(), self.true_arr.compressed()
        return stats.pearsonr(x,y)[0] ** 2

    def seaborn_jointplot(self):
        import seaborn as sns
        def r2(x,y):
            return stats.pearsonr(x,y)[0] ** 2
        g = sns.jointplot('true', 'pred', data=self.dataframe, kind='reg', stat_func=r2)

    def hexbin_plot(self, colorbar=True):
        df = self.dataframe
        fig,ax = plt.subplots(1,1)
        mapa = ax.hexbin(df.true,df.pred,mincnt=1,bins=None,gridsize=500,\
                             cmap=plt.cm.jet)
        ax.set_ylabel(self.pred_name)
        ax.set_xlabel(self.true_name)
        ax.set_aspect('equal')
        dmin = df.pred.min()
        dmax = df.pred.max()
        ax.plot([dmin,dmax],[dmin,dmax],c='white',alpha=0.6)
        ax.set_title(r"RMSE: {:.2f}, $R^2$: {:.2f}".format(self.rmse, self.rsquared))
        if colorbar:
            fig.colorbar(mapa)
        return fig

    def error_array(self):
        return (self.pred_arr - self.true_arr).squeeze()

    def same_resolution(self, print_res=False):
        """
        Check if the gdal geotransforms match for the rasters. If they match,
        the resolutions are the same.
        """
        gt1 = np.array(self.pred_rds.gdal_ds.GetGeoTransform())[[1,5]]
        gt2 = np.array(self.true_rds.gdal_ds.GetGeoTransform())[[1,5]]
        if print_res:
            print gt1, gt2
        return np.allclose(gt1, gt2)
