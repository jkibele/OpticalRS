##OpticalRS=group
##Raster=raster
##Shape=vector
##Output_file=output file

from OpticalRS import RasterDS, RasterSubset
import geopandas as gpd

imrds = RasterDS( Raster )
ss = gdp.read_file( Shape )

# just take the first geometry
geom = ss.iloc[0].geometry

# get masked array subset of the raster
imarr = RasterSubset.masked_subset( imrds, geom )

imarr.dump( Output_file )