##OpticalRS=group
##Denoise=name
##Multispectral_image=raster
##Output_raster=output raster

from OpticalRS import RasterDS, ArrayUtils
from processing.core.GeoAlgorithmExecutionException import \
        GeoAlgorithmExecutionException
try:
    from skimage.restoration import denoise_bilateral
    
except:
    raise GeoAlgorithmExecutionException("Scikit-image isn't insalled")

imrds = RasterDS( Multispectral_image )

# get the image array
imarr = imrds.band_array


progress.setText("Masking the Land")
# create the masked version of the input array
outarr = LandMasking.mask_land( imarr, NIR_threshold, Connectivity_threshold )

# make a raster image from the output array
imrds.new_image_from_array( outarr, Output_raster )
