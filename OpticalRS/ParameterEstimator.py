

from GeoDFUtils import RasterShape
from RasterDS import RasterDS
from ArrayUtils import equalize_array_masks
from AlbedoIndex import albedo_parameter_plots, est_curve_params, param_df
from WV2RadiometricCorrection import get_xmlroot, meanSunEl, meanOffNadirViewAngle
from Lyzenga2006 import dark_pixel_array
from Lyzenga1978 import regression_plot, regressions
from Const import wv2_center_wavelength, jerlov_Kd
from pylab import subplots
import numpy as np
import geopandas as gpd
import pandas as pd

class ParameterEstimator(RasterShape):
    def __init__(self, img_rds, depth_rds, sand_shp, gdf_query=None, depth_range=None):
        self.depth_range = depth_range
        if type(img_rds).__name__ == 'RasterDS':
            self.img_rds = img_rds
        else:
            self.img_rds = RasterDS(img_rds)

        if type(depth_rds).__name__ == 'RasterDS':
            self.depth_rds = depth_rds
        else:
            self.depth_rds = RasterDS(depth_rds)

        if type(sand_shp).__name__ == 'GeoDataFrame':
            self.gdf = sand_shp
        else:
            self.gdf = gpd.read_file(sand_shp)

        self.gdf_query = gdf_query
        self.full_image_array = self.img_rds.band_array

        self._set_arrays()

    def copy(self, gdf_query="unchanged", depth_range="unchanged"):
        if gdf_query is "unchanged":
            gdf_query = self.gdf_query
        if depth_range is "unchanged":
            depth_range = self.depth_range
        return ParameterEstimator(self.img_rds, self.depth_rds, self.gdf,
                                  gdf_query, depth_range)

    @property
    def _unequal_image_subset(self):
        """
        The image array masked outside of the geometry. The mask on this array
        may not match the mask on the depth array.
        """
        return self.img_rds.geometry_subset(self.geometry)

    @property
    def _unequal_depth_subset(self):
        """
        The depth array masked outside of the geometry. The mask on this array
        may not match the mask on the image array.
        """
        darr = self.depth_rds.geometry_subset(self.geometry).squeeze()
        if type(self.depth_range).__name__ != 'NoneType':
            darr = np.ma.masked_outside(darr, *self.depth_range)
        return darr

    def set_depth_range(self, depth_range):
        self.depth_range = depth_range
        self._set_arrays()

    def _set_arrays(self):
        imarr, darr = equalize_array_masks(self._unequal_image_subset, self._unequal_depth_subset)
        self.image_subset_array = imarr
        self.depth_subset_array = darr.squeeze()
        return True

    def same_resolution(self, print_res=False):
        """
        Check if the gdal geotransforms match for the rasters. If they match,
        the resolutions are the same.
        """
        gt1 = np.array(self.img_rds.gdal_ds.GetGeoTransform())[[1,5]]
        gt2 = np.array(self.depth_rds.gdal_ds.GetGeoTransform())[[1,5]]
        if print_res:
            print gt1, gt2
        return np.allclose(gt1, gt2)

    @property
    def geometry(self):
        """
        Return a single geometry from `self.gdf` (the GeoDataFrame representation
        of `sand_shp`). If `gdf_query` has not been set, the geometry returned
        will just be the first geometry in `sand_shp`. If `gdf_query` has been
        set, the returned geometry will be the first one returned by that query.

        Returns
        -------
        shapely.geometry
            A geometry shapely (https://pypi.python.org/pypi/Shapely) geometry
            object.

        """
        if self.gdf_query == None:
            geom = self.gdf.ix[0].geometry
        else:
            geom = gdf.query(self.gdf_query).ix[0].geometry
        return geom

    def deep_water_means(self, p=10, win_size=3, win_percentage=50):
        dpa = dark_pixel_array(self.full_image_array, p=p, win_size=win_size,
                               win_percentage=win_percentage)
        deep_water_means = dpa.reshape(-1,dpa.shape[-1]).mean(0)
        return deep_water_means.data

    def linear_parameters(self, deep_water_means=None, geometric_factor=2.0):
        if type(deep_water_means).__name__ == 'NoneType':
            dwm = self.deep_water_means()
        else:
            dwm = deep_water_means
        X = np.ma.log(self.image_subset_array - dwm)
        X, Xdepth = equalize_array_masks(X, self.depth_subset_array)
        params = regressions(Xdepth, X)
        Kg_arr = -1 * params[0]
        nbands = np.atleast_3d(X).shape[-1]
        pardf = pd.DataFrame(Kg_arr, columns=["Kg"],
                             index=wv2_center_wavelength[:nbands])
        pardf['K'] = pardf.Kg / geometric_factor
        return pardf

    def linear_fit_plot(self, deep_water_means=None):
        if type(deep_water_means).__name__ == 'NoneType':
            dwm = self.deep_water_means()
        else:
            dwm = deep_water_means
        X = np.ma.log(self.image_subset_array - dwm)
        X, Xdepth = equalize_array_masks(X, self.depth_subset_array)
        fig = regression_plot(Xdepth, X)
        return fig

    def curve_fit_parameters(self, geometric_factor=2.0):
        paramdf = param_df(self.depth_subset_array, self.image_subset_array, geometric_factor=geometric_factor)
        return paramdf

    def curve_fit_plots(self, params=None):
        return albedo_parameter_plots(self.image_subset_array, self.depth_subset_array, params=params)

    def K_comparison_plot(self, paramdf, columns='K', figure_title="$K$ Estimates vs. $K$ Values from Jerlov"):
        return jerlov_Kd_plot(paramdf, columns, figure_title)

## Geometric Factor #######################################################
#     Methods for finding the geometric factor 'g'. Not to be confused with
#     what I'm calling 'g' in AlbedoIndex. That's really 'Kg'. Sorry.

def angle_of_refraction(ang_incidence, n_h2o=1.34):
    """
    Calculate the angle of refraction underwater given the angle of incidence
    in air using Snell's Law.
    """
    aor = np.arcsin( np.sin(np.deg2rad(ang_incidence)) / n_h2o )
    return np.rad2deg(aor)

def geometric_factor(sun_elevation, view_angle):
    """
    Calculate the geometric factor for path length bellow the water's
    surface based on the sun elevation and satellite off nadir view
    angle. This will account for refraction at the water's surface
    both downwelling and upwelling. Basically, you'll get a result of
    2 or a bit more than 2 in most situations. For an example of the us
    of the geometric factor see Sagawa et al. 2010 equation 2. Also see
    Lyzenga et al. 2006 equation 16 for something similar.

    Parameters
    ----------
    sun_elevation : float
        The sun elevation in degrees. For WorldView-2 imagery you can
        find this in the image metadata as MEANSUNEL.
    view_angle : float
        The off-nadir view angle from the satellite. For WorldView-2
        imagery this is in the image metadata as MEANOFFNADIRVIEWANGLE.

    Returns
    -------
    g : float
        The geometric factor for path length.
    """
    Ts = angle_of_refraction(90 - sun_elevation)
    Tv = angle_of_refraction(view_angle)
    secant = lambda t: np.cos(np.deg2rad(t))**-1.0
    g = secant(Ts) + secant(Tv)
    return g

def geometric_factor_from_imd(imd_path):
    """
    Calculate the geometric factor for path length bellow the water's
    surface based on the sun elevation and satellite off nadir view
    angle. This will account for refraction at the water's surface
    both downwelling and upwelling. Basically, you'll get a result of
    2 or a bit more than 2 in most situations. For an example of the us
    of the geometric factor see Sagawa et al. 2010 equation 2. Also see
    Lyzenga et al. 2006 equation 16 for something similar.

    Parameters
    ----------
    imd_path : string
        The path to a WorldView-2 image metadata xml file.

    Returns
    -------
    g : float
        The geometric factor for path length.
    """
    xmlroot = get_xmlroot(imd_path)
    sun_el = meanSunEl(xmlroot)
    off_nad = meanOffNadirViewAngle(xmlroot)
    return geometric_factor(sun_el, off_nad)

## Visualization ###########################################################

def jerlov_Kd_plot(paramdf, columns='K', figure_title="$K$ Estimates vs. $K$ Values from Jerlov"):
    from matplotlib.cm import summer_r
    from matplotlib import style
    style.use('ggplot')
    jerlov_df = jerlov_Kd()
    fig, ax = subplots(1,1, figsize=(8,6))
    jerlov_df.plot(linestyle='--', cmap=summer_r, ax=ax)
    if paramdf is not None:
        paramdf[columns].plot(ax=ax, marker='o')
        maxval = paramdf[columns].as_matrix().max()
        ax.set_ylim(0,maxval + 0.5 * maxval)
    blah = ax.set_title(figure_title)
    return fig
