##OpticalRS=group
##Multispectra_image=raster
##Glint_shape=vector
##NIR_band=number 8
##Output_raster=output raster

from OpticalRS import RasterDS, RasterSubset, Lyzenga2006
import geopandas as gpd

imrds = RasterDS( Multispectra_image )
ss = gpd.read_file( Glint_shape )

# just take the first geometry
geom = ss.iloc[0].geometry

# get masked array subset of the raster
glint_arr = RasterSubset.masked_subset( imrds, geom )

glint_corrected = Lyzenga2006.glint_correct_image( imrds.band_array, glint_arr, nir_band=NIR_band - 1 )

outrds = imrds.new_image_from_array( glint_corrected, Output_raster )