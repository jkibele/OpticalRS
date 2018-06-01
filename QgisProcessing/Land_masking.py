##OpticalRS=group
##Land Masking=name
##Multispectral_image=raster
##Estimate_Threshold=boolean True
##NIR_threshold=number 100
##Connectivity_threshold=number 1000
##Output_raster=output raster

from OpticalRS import RasterDS, LandMasking

imrds = RasterDS( Multispectral_image )

# get the image array
imarr = imrds.band_array

if Estimate_Threshold:
    progress.setText("Estimating NIR Threshold (this may take a while)")
    NIR_threshold = LandMasking.auto_water_threshold(imarr)
    est_text = "Threshold estimated to be {}".format(NIR_threshold)
    progress.setText(est_text)

progress.setText("Masking the Land")
# create the masked version of the input array
outarr = LandMasking.mask_land( imarr, NIR_threshold, Connectivity_threshold )

# make a raster image from the output array
imrds.new_image_from_array( outarr, Output_raster )
