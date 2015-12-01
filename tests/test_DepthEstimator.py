#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_DepthEstimator
-------------------

pytest Tests for `OpticalRS.DepthEstimator` module. To run these tests, install
pytest and run `py.test` in this test directory.
"""

from OpticalRS.RasterDS import RasterDS
from OpticalRS.DepthEstimator import DepthEstimator
import pytest
import numpy as np

rds = RasterDS('data/eReefWV2.tif')
drds = RasterDS('data/eReefDepth.tif')
imarr = rds.band_array
darr = drds.band_array.squeeze()
imflat = imarr.reshape(-1,imarr.shape[-1])
dflat = darr.ravel()

@pytest.fixture(params=[rds,imarr,imflat], 
                ids=['imgRasterDS','imgArray','imgFlat'])
def image(request):
    return request.param
    
@pytest.fixture(params=[drds,darr,dflat], 
                ids=['depRasterDS','depArray','depFlat'])
def depth(request):
    return request.param

@pytest.fixture
def dep_est(image,depth):
    print "Img Type: {}, Depth Type: {}".format(type(image).__name__,type(depth).__name__)
    return DepthEstimator(image,depth)


class TestDepthEstimator:
        
    def test_nbands(self,dep_est):
        assert dep_est.nbands == 8
        
    def test_image_masking(self,dep_est):
        if np.ma.isMA(dep_est.imarr) and np.ma.isMA(dep_est.known_depth_arr):
            imgmask = dep_est.known_imarr.mask[...,0]
            depmask = dep_est.known_depth_arr.mask
            assert np.array_equiv(imgmask,depmask)
            
        imgmask = dep_est.known_imarr_flat.mask[...,0]
        depmask = dep_est.known_depth_arr_flat.mask
        assert np.array_equiv(imgmask,depmask)


