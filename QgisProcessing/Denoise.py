##OpticalRS=group
##Denoise=name
##Multispectral_image=raster
##Output_raster=output raster

from OpticalRS import RasterDS, ArrayUtils
from osgeo.gdal import GDT_Float32
import numpy as np
from processing.core.GeoAlgorithmExecutionException import \
        GeoAlgorithmExecutionException
# from qgis.core import QgsMessageLog
try:
    from skimage.restoration import denoise_bilateral

except:
    raise GeoAlgorithmExecutionException("Scikit-image isn't insalled")

imrds = RasterDS( Multispectral_image )

# get the image array
imarr = imrds.band_array.astype('float32')

progress.setText("Array dtype: {}".format(imarr.dtype))

progress.setText("Denoising")
denoised = ArrayUtils.each_band(np.clip(imarr, 0, 1), denoise_bilateral,
                                multichannel=False)

# make a raster image from the output array
imrds.new_image_from_array( denoised, Output_raster, dtype=GDT_Float32,
                            no_data_value=0.0)
