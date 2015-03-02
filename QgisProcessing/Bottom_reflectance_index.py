##Multispectral_raster=raster
##Depth_raster=raster
##Sand_shape=vector
##Output_raster=output raster

from OpticalRS import Sagawa2010, RasterDS, RasterSubset
import geopandas as gpd
from processing.core.GeoAlgorithmExecutionException import GeoAlgorithmExecutionException

imrds = RasterDS(Multispectral_raster)
depthrds = RasterDS(Depth_raster)
ss = gpd.read_file(Sand_shape)

# just take the first geometry
geom = ss.iloc[0].geometry

# get masked subsets of the rasters to calculate geometric/attenuation
# coefficients.
imarr = RasterSubset.masked_subset( imrds, geom )
deptharr = RasterSubset.masked_subset( depthrds, geom )

# get coefficient for each band
negKgs = Sagawa2010.negKg_array( imarr, deptharr.squeeze() )

# calculate reflectance index
sag_ri = Sagawa2010.reflectance_index( imrds.band_array, depthrds.band_array.squeeze(), negKgs )

# save the output
outrds = imrds.new_image_from_array(sag_ri,Output_raster)

#raise GeoAlgorithmExecutionException("I hate you.")

Output_raster = outrds.file_path