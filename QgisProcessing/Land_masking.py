##OpticalRS=group
##Multispectral_image=raster
##NIR_threshold=number 100
##Connectivity_threshold=number 1000
##Output_raster=output raster

from OpticalRS import RasterDS, LandMasking

imrds = RasterDS( Multispectral_image )

# get the image array
imarr = imrds.band_array

# create the masked version of the input array
outarr = LandMasking.mask_land( imarr, NIR_threshold, Connectivity_threshold )

# make a raster image from the output array
imrds.new_image_from_array( outarr, Output_raster )