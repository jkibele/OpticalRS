# -*- coding: utf-8 -*-
"""
DepthEstimator
==============

Code for handling required data and producing depth estimates from multispectral
satellite imagery.
"""

from OpticalRS.RasterDS import RasterDS

class DepthEstimator(object):
    """
    I want to be able to chuck in the image and the known depths in a number of
    different formats and do depth prediction stuff with it.
    """
    def __init__(self,img,known_depths,k=5,weights='uniform'):
        self.img_original = img
        self.rds = None
        try:
            self.level = img.ndim
        except AttributeError:
            if type(img).__name__=='RasterDS':
                self.level = 4
                self.rds = img
            else:
                # next line should raise exception if `img` can't make RasterDS
                self.rds = RasterDS(img)
                self.level = 4
        self.known_original = known_depths
        self.k = k
        self.weights = weights
        self.imarr = self.__imarr()
        if self.imarr:
            self.nbands = self.imarr.shape[-1]
        else:
            self.nbands = img.shape[-1]
        
        
    def __imarr(self):
        """
        Return 3D (R,C,nBands) image array if possible. If only 2D 
        (pixels,nBands) array is available, return `None`.
        """
        if self.imarr:
            pass # we'll return self.imarr
        elif self.level==4:
            self.imarr = self.rds.band_array
        elif self.level==3:
            self.imarr = self.img_original
        else: # level 2
            return None
        return self.imarr
            
    @property
    def flat_imarr(self):
        """
        Return all the image pixels in (pixels,bands) shape.
        """
        if self.level>2:
            return self.imarr.reshape(-1,self.nbands)
        else:
            return self.img_original
            
    