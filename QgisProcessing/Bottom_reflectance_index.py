##OpticalRS=group
##Multispectral_raster=raster
##Depth_raster=raster
##Sand_shape=vector
##Output_raster=output raster

from OpticalRS import Sagawa2010, RasterDS, RasterSubset
from OpticalRS.ArrayUtils import equalize_array_masks
import geopandas as gpd
from processing.core.GeoAlgorithmExecutionException import GeoAlgorithmExecutionException

imrds = RasterDS(Multispectral_raster)
depthrds = RasterDS(Depth_raster)
ss = gpd.read_file(Sand_shape)

# just take the first geometry
geom = ss.iloc[0].geometry

# get masked subsets of the rasters to calculate geometric/attenuation
# coefficients.
sandim = RasterSubset.masked_subset( imrds, geom )
sanddep = RasterSubset.masked_subset( depthrds, geom ).squeeze()

# make sure the masks match
sandim, sanddep = equalize_array_masks(sandim, sanddep)

# get coefficient for each band
negKgs = Sagawa2010.negKg_array( sandim, sanddep )

# calculate reflectance index
imarr, deptharr = equalize_array_masks(imrds.band_array, depthrds.band_array.squeeze() )
sag_ri = Sagawa2010.reflectance_index( imarr, deptharr, negKgs )

# save the output
outrds = imrds.new_image_from_array(sag_ri,Output_raster)

#raise GeoAlgorithmExecutionException("I hate you.")

Output_raster = outrds.file_path
