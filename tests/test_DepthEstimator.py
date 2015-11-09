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

rds = RasterDS('data/eReefWV2.tif')
drds = RasterDS('data/eReefDepth.tif')
imarr = rds.band_array
darr = drds.band_array.squeeze()

#imflat = imarr.reshape(-1,imarr.shape[-1])
#dflat = darr.ravel()

@pytest.fixture
def dep_est(scope='class', image=rds, depth=darr):
#    print image
    print "Img Type: {}".format(type(image).__name__)
    return DepthEstimator(image,depth)

#@pytest.mark.parametrize("image", [rds,imarr])
#@pytest.mark.parametrize("depth", [drds,darr])
class TestDepthEstimatorWith2RasterDS:
        
    def test_image_level(self,dep_est):
        assert dep_est.imlevel == 4