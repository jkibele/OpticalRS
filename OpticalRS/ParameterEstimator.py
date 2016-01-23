

from GeoDFUtils import RasterShape
from RasterDS import RasterDS
from ArrayUtils import equalize_array_masks
from AlbedoIndex import albedo_parameter_plots, est_curve_params, param_df, jerlov_Kd_plot
from WV2RadiometricCorrection import get_xmlroot, meanSunEl, meanOffNadirViewAngle
import numpy as np
import geopandas as gpd
import pandas as pd

class ParameterEstimator(RasterShape):
    def __init__(self, img_rds, depth_rds, sand_shp, gdf_query=None):
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
        imarr, darr = equalize_array_masks(self._full_image_array, self._full_depth_array)
        self.image_array = imarr
        self.depth_array = darr.squeeze()

    @property
    def _full_image_array(self):
        """
        The image array masked outside of the geometry. The mask on this array
        may not match the mask on the depth array.
        """
        return self.img_rds.geometry_subset(self.geometry)

    @property
    def _full_depth_array(self):
        """
        The depth array masked outside of the geometry. The mask on this array
        may not match the mask on the image array.
        """
        return self.depth_rds.geometry_subset(self.geometry).squeeze()

    def same_geotransform(self):
        """
        Check if the gdal geotransforms match for the rasters. If they match,
        the resolutions are the same.
        """
        gt1 = img_rds.gdal_ds.GetGeoTransform()
        gt2 = depth_rds.gdal_ds.GetGeoTransform()
        return np.allclose(gt1, gt2)

    @property
    def geometry(self):
        if self.gdf_query == None:
            geom = self.gdf.ix[0].geometry
        else:
            geom = gdf.query(self.gdf_query).ix[0].geometry
        return geom

    def curve_fit_parameters(self, geometric_factor=2.0):
        paramdf = param_df(self.depth_array, self.image_array, geometric_factor=geometric_factor)
        return paramdf

    def curve_fit_plots(self, params=None):
        return albedo_parameter_plots(self.image_array, self.depth_array, params=params)

    def K_comparison_plot(self, paramdf):
        return jerlov_Kd_plot(paramdf)

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
