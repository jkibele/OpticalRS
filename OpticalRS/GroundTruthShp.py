"""
I'm copying this from some of my much older work. It's probably going to be
buggy. Yep, it is. This needs to be refactored but there's some useful stuff.
Maybe combine with `GeoDFUtils`?

`buffer` and `rasterize` are working pretty well. Still need tests and whatnot
but those methods are the main use for this module at the moment.
"""

#from error_matrix import *
from RasterDS import RasterDS
from scipy.stats.stats import mode
import numpy as np
import geopandas as gpd
import os
from osgeo import ogr, gdal, osr

class GroundTruthShapefile(object):
    """
    I'm trying to make this as general as possible but we'll have to assume a
    few things. I'm going to assume that there's a field called habitat that
    contains a text description of the habitat class for each point.
    """
    def __init__(self, file_path, habfield='habitat'):
        self.habfield = habfield
        self.file_path = file_path
        self.ds = open_shapefile(self.file_path)
        self.hab_dict = self.__setup_hab_dict()
        self.legit_habs = sorted( [ h for h in self.habitats if h ] ) # Exclude None as a habitat value
        self.habitat_codes = self.__setup_hab_codes() # dict( zip( legit_habs, range( 1, len(legit_habs) + 1 ) ) )

    def __setup_hab_dict(self):
        """
        The hab_dict is a dictionary that contains a list of ogr features for
        each habitat key.
        """
        hab_dict = {}
        for hab in self.habitats:
            hab_dict[hab] = [f for f in self.features if f.habitat==hab]
        return hab_dict

    def __setup_hab_codes(self):
        """
        There should be habitat codes in the shapefile in a field called
        hab_num. We need to get them and set up the matching names. This only
        works for BPS shapefiles with a hab_num field set up to match the
        habitat field. If `self.habfield` is set to something else, we'll just
        generate integer codes.
        """
        # Exclude None from list of habitats
        hcd = {}
        if self.habfield == 'habitat':
            for hab in self.legit_habs:
                feat = self.hab_dict[hab][0]
                hcd[hab] = feat.hab_num
        else:
            for i, hab in enumerate(self.legit_habs):
                hcd[hab] = i+1 # +1 to make it not zero indexed
        return hcd

    @property
    def features(self):
        fts = [f for f in self.ds.GetLayer()]
        self.ds.GetLayer().ResetReading()
        return fts

    @property
    def habitats(self):
        habs = sorted( set([f.__getattr__(self.habfield) for f in self.features]))
        return habs

    @property
    def geo_data_frame(self):
        """
        Return a GeoPandas GeoDataFrame object.
        """
        return gpd.GeoDataFrame.from_file(self.file_path)

    @property
    def spatial_reference(self):
        """
        Return the OGR spatial reference object for the shapefile.
        """
        return self.ds.GetLayer().GetSpatialRef()

    @property
    def projection_wkt(self):
        """
        Return the well known text (WKT) representation of the shapefile's projection.
        """
        return self.spatial_reference.ExportToWkt()

    @property
    def projcs(self):
        """
        Return the PROJCS value from the shapefile's spatial reference. This is
        basically the name of the projection. ...I think.
        """
        return self.spatial_reference.GetAttrValue('PROJCS')

    @property
    def geometry_type(self):
        """
        Just return whether it's a type of point, line, or polygon.
        """
        type_name = ogr.GeometryTypeToName( self.ds.GetLayer().GetGeomType() ).lower()
        if type_name.find('point') <> -1:
            return 'point'
        elif type_name.find('line') <> -1:
            return 'line'
        elif type_name.find('polygon') <> -1:
            return 'polygon'
        else:
            return None

    @property
    def hab_colors(self):
        """
        return a dictionary with hab codes as keys and hab colors as values.
        """
        legit_habs = sorted( [ h for h in self.habitats if h ] )
        hcd = {}
        for hab in legit_habs:
            feat = self.hab_dict[hab][0]
            hcd[hab] = feat.hab_color
        return hcd

    @property
    def codes_habitat(self):
        """
        Return a dictionary just like habitat_codes only backwards.
        """
        chd = {}
        for k,v in self.habitat_codes.items():
            chd[v] = k
        return chd

    @property
    def qgis_vector(self):
        qvl = QgsVectorLayer(self.file_path,'grnd_truth','ogr')
        if qvl.isValid():
            return qvl
        else:
            raise Exception("Failed to create a QGis Vector Layer. QGis provider path problems, perhaps?")

    def buffer(self, radius=1.0, file_path=None):
        """
        Buffer the geometries in `self` and return a new `ogr` datasource. If
        `file_path` is `None`, just create the datasource in memory. If a file
        path is given, write out a shapefile. All fields and values (aside from
        geometry) are cloned.
        """
        if file_path == None:
            drvname = 'Memory'
        else:
            drvname = 'ESRI Shapefile'
        srcds = self.ds
        # get projection
        lyr = srcds.GetLayer(0)
        sptrf = lyr.GetSpatialRef()
        proj = osr.SpatialReference()
        proj.ImportFromWkt(sptrf.ExportToWkt())
        drv = ogr.GetDriverByName(drvname)
        if file_path == None:
            dst_ds = drv.CreateDataSource('out')
        elif os.path.exists(file_path):
            raise Exception("{} already exists!".format(file_path))
        else:
            dst_ds = drv.CreateDataSource(file_path)
        dst_lyr = dst_ds.CreateLayer('', srs=proj, geom_type=ogr.wkbPolygon)
        # copy all the fields to the destination ds
        featr = lyr.GetFeature(0)
        nfields = featr.GetFieldCount()
        for i in range(nfields):
            fld = featr.GetFieldDefnRef(i)
            dst_lyr.CreateField(fld)
        feat_defn = dst_lyr.GetLayerDefn()
        # reset the feature counter
        lyr.ResetReading()
        # buffer the geometries and copy the fields
        for i in range(lyr.GetFeatureCount()):
            # get the feature and geometry
            feat = lyr.GetFeature(i)
            geom = feat.GetGeometryRef()
            # create a new feature
            newfeat = feat.Clone()
            # get the buffered geometry
            bufgeom = geom.Buffer(radius)
            # set the new geometry to the buffered geom
            newfeat.SetGeometry(bufgeom)
            # add the new feature to the destination layer
            dst_lyr.CreateFeature(newfeat)
            # clean up
            newfeat.Destroy()
            feat.Destroy()
        # ensure the new features are written
        dst_lyr.SyncToDisk()

        return dst_ds


    def rasterize(self, buffer_radius=None, raster_template=None,
                  pixel_size=1.99976, value_field='hab_num', float_values=False,
                  array_only=False, out_file_path=None):
        """
        Return a raster that can be used for classification training.

        buffer_radius: A float value in projection units to buffer the
        geometries by. If buffer_radius is left None then only pixels right
        under points will be classified.

        raster_template: A RasterDS object. If supplied, the resulting
        rasterized image will have the same extent and geotransform as the
        template. Also, if a raster_template is provided, the pixel_size keyword
        value will be ignored and pixel size will come from the template.

        pixel_size: A float value representing pixel size in projection units.
        This value will be ignored if a raster_template is supplied.

        value_field: A string representing the name of the field in the
        shapefile that holds the numeric code that will be burned into the
        raster output as the pixel value.

        float_values: Boolean. If `True`, the output raster will contain floats.
        If `False`, the output will be integers. Default is `False`.

        array_only: A boolean. If true we'll try to just write the raster to
        memory and not to disk. If you don't need to keep the raster, this will
        just keep you from having to clean up useless files later. Then we'll
        just return an array instead of GroundTruthRaster object.

        out_file_path: String. Path to the raster file output. If `None`
        (default) and `array_only=False`, a file name based on the
        `GroundTruthShapefile` file name will be created. If `array_only=True`,
        `out_file_path` is ignored.
        """
        if float_values:
            datatype = gdal.GDT_Float32
        else:
            datatype = gdal.GDT_Byte
        # Make a copy of the layer's data source because we'll need to
        # modify its attributes table
        if buffer_radius:
            source_ds = ogr.GetDriverByName("Memory").CopyDataSource( self.buffer(radius=buffer_radius), "" )
        else:
            source_ds = ogr.GetDriverByName("Memory").CopyDataSource( self.ds, "")
        source_layer = source_ds.GetLayer(0)
        source_srs = source_layer.GetSpatialRef()

        if raster_template:
            gTrans = raster_template.gdal_ds.GetGeoTransform()
            pixsizeX = gTrans[1]
            pixsizeY = gTrans[5]
            x_res = raster_template.gdal_ds.RasterXSize
            y_res = raster_template.gdal_ds.RasterYSize
            rdsarr = raster_template.band_array
            # if np.ma.is_masked(rdsarr):
            #     mask = rdsarr[...,0].mask
            # else:
            #     mask = None
        else:
            x_min, x_max, y_min, y_max = source_layer.GetExtent()
            # Create the destination data source
            x_res = int((x_max - x_min) / pixel_size)
            y_res = int((y_max - y_min) / pixel_size)

        if out_file_path:
            targ_fn = out_file_path
        else:
            # make a target ds with filename based on source filename
            targ_fn = self.file_path.rsplit(os.path.extsep, 1)[0] + '_rast' + os.path.extsep + 'tif'
        # print "x_res: %i, y_res: %i" % (x_res,y_res)
        target_ds = gdal.GetDriverByName('GTiff').Create(targ_fn, x_res, y_res, 1, datatype)

        if raster_template:
            # Use the raster template supplied so that we get the same extent as the raster
            # we're trying to classify
            target_ds.SetGeoTransform( gTrans )
        else:
            # None supplied so use the pixel_size value and the extent of the shapefile
            target_ds.SetGeoTransform(( x_min, pixel_size, 0, y_max, 0, -pixel_size, ))
        if raster_template:
            target_ds.SetProjection( raster_template.gdal_ds.GetProjection() )
        elif source_srs:
            # Make the target raster have the same projection as the source
            target_ds.SetProjection(source_srs.ExportToWkt())
        else:
            # Source has no projection (needs GDAL >= 1.7.0 to work)
            target_ds.SetProjection('LOCAL_CS["arbitrary"]')
        # Rasterize
        err = gdal.RasterizeLayer(target_ds, [1], source_layer,
                burn_values=[0],
                options=["ATTRIBUTE=%s" % value_field])
        if err != 0:
            raise Exception("error rasterizing layer: %s" % err)
        # clean up
        source_layer = None
        source_srs = None
        source_ds = None

        if array_only:
            out_array = target_ds.ReadAsArray()
            target_ds = None
            os.remove( targ_fn )
            return out_array
        else:
            target_ds = None
            return RasterDS(targ_fn)

    def error_matrix(self, classification_ds):
        """
        Take a RasterDS (classification_ds) and create a user / producer accuracy table. Return
        as an array so it can be displayed in multiple ways.
        """
        errmat = np.zeros((len(self.legit_habs),len(self.legit_habs)),int)
        for hab,code in self.habitat_codes.items():
            for feature in self.hab_dict[hab]:
                ref_val = code
                cls_val = classification_ds.value_at_point( feature.geometry() )
                errmat[ cls_val - 1 ][ ref_val - 1 ] += 1

        return errmat.view( ErrorMatrix )

    @property
    def hab_dict_counts(self):
        ret_dict = {}
        for hab in self.habitats:
            ret_dict[hab] = len( self.hab_dict[hab] )
        return ret_dict

    def add_raster_values(self, raster_ds):
        """
        The raster data source here is assumed to be a classified image. The raster
        values should correspond to classes.
        """
        trans = transform_dict(raster_ds)
        band = raster_ds.GetRasterBand(1)
        self.features = [ add_raster_value(f,trans,band) for f in self.ds.GetLayer() ]
        self.ds.GetLayer().ResetReading()
        self.hab_dict = self.__setup_hab_dict()

    @property
    def unsupervised_habitat_class_dict(self):
        """
        For each habitat, give a list of raster values that correspond to the ground truth
        points of that habitat type. This will be used with unsupervised classifications to
        figure out which, if any, of the classes correspond to particular habitat types.
        """
        try:
            hcd = {}
            for hab in self.habitats:
                hcd[hab] = [ f.raster_value for f in self.hab_dict[hab] ]
        except AttributeError:
            raise AttributeError("Features need to be assigned raster values before you can create a habitat class dictionary.")
        return hcd

    @property
    def unsupervised_habitat_class_modes(self):
        hcm = {}
        for hab in self.habitats:
            md, cn = mode( self.unsupervised_habitat_class_dict[hab] )
            if len( md )==1:
                hcm[hab] = md[0]
            else:
                hcm[hab] = None
        return hcm

    def __output_training_LAN(self,img,buffer_radius=3.5,driver_str='LAN'):
        """
        DEPRICATED! -> This only works for points. I think I can use the rasterize method instead. I need to
        verify and then get rid of this method.

        Create a raster input for supervised classifications. img is the image that we
        want to classify (in the form of a gdal datasource). Spectral can't use tifs so we will create LAN file.

        A buffer radius of 3.5 meters gives us 3 x 3 sets of pixels with our point feature in the center.
        This, of course, assumes that we're dealing with WV2 imagery and a projection with meters as the
        units. This works for me on my project but might now work for others.
        """
        if driver_str=='LAN':
            f_ext = 'lan'
        elif driver_str=='GTiff':
            f_ext = 'tif'
        else:
            raise ValueError("At this point, the output_training_LAN method only knows how to deal with LAN and GTiff file types. Sorry.")

        lyr = self.ds.GetLayer()
        lyr.ResetReading()
        trans = transform_dict(img)
        driver = gdal.GetDriverByName(driver_str)
        rows = img.RasterYSize
        cols = img.RasterXSize
        fname = img.GetDescription().rsplit(os.path.extsep)[0] + '_train' + os.path.extsep + f_ext
        add_num = 0
        while os.path.exists(fname):
            add_num += 1
            if add_num==1:
                fname = fname.replace( os.path.extsep + f_ext, '_%i' % add_num + os.path.extsep + f_ext )
            else:
                old = '_%i.%s' % ( add_num - 1, f_ext )
                new = '_%i.%s' % ( add_num, f_ext )
                fname = fname.replace( old, new )
        outDs = driver.Create(fname, cols, rows, 1, GDT_Int16)
        if outDs is None:
            print 'Could not create %s' % fname
            sys.exit(1)

        outBand = outDs.GetRasterBand(1)

        pixel_count = 0
        hab_pix_count = dict( zip( [h for h in self.habitats if h], np.zeros( len([h for h in self.habitats if h]), dtype=np.int ) ) )
        for feat in lyr:
            if not feat.habitat:
                continue
            if self.hab_dict_counts[feat.habitat] < 24:
                continue
            if buffer_radius:
                geom = feat.geometry().Buffer(buffer_radius)
                elp = envelope_dict(geom)
                xtop = elp['xLeft']
                ytop = elp['yTop']
                xOffset = int( (xtop - trans['originX']) / trans['pixWidth'] )
                yOffset = int( (ytop - trans['originY']) / trans['pixHeight'] )
                xdist = elp['xRight'] - elp['xLeft']
                ydist = elp['yBottom'] - elp['yTop']
                cols = int( xdist / trans['pixWidth'] )
                rows = int( ydist / trans['pixHeight'] )
                pixarr = int( self.habitat_codes[feat.habitat] ) * np.ones((rows,cols), dtype=np.int16)
            else:
                geom = feat.geometry()
                xOffset = int( (geom.GetX() - trans['originX']) / trans['pixWidth'] )
                yOffset = int( (geom.GetY() - trans['originY']) / trans['pixHeight'] )
                pixarr = np.array( [[ self.habitat_codes[feat.habitat] ]] )

            outBand.WriteArray(pixarr,xOffset,yOffset)
            pixel_count += pixarr.size
            hab_pix_count[feat.habitat] += pixarr.size

        outBand.FlushCache()
        outBand.SetNoDataValue(0)
        # georeference the image and set the projection
        outDs.SetGeoTransform(img.GetGeoTransform())
        outDs.SetProjection(img.GetProjection())

        # build pyramids
        gdal.SetConfigOption('HFA_USE_RRD', 'YES')
        outDs.BuildOverviews(overviewlist=[2,4,8,16,32,64,128])

        print "%i pixels total" % pixel_count
        for hab in self.habitats:
            if hab:
                print "%i pixels for %s" % ( hab_pix_count[hab], hab )

        return GroundTruthRaster( outDs.GetDescription() )

    def training_classes(self, rds, buffer_radius=None,calc_stats=0):
        """
        I think I should move some of this functionality over to the GroundTruthRaster class
        in common.py.  I'm generating classes okay from what I can tell but I get a singular
        matrix error when I try to run the Gaussian Classifier. I have no idea why. Baffled,
        I am.
        """
        grnd_truth = self.rasterize(buffer_radius=buffer_radius,raster_template=rds,array_only=True)
        sp_img = rds.spy_image.load()
        return sp.create_training_classes(sp_img, grnd_truth,calc_stats=calc_stats)


def add_raster_value(feature, trans, band ):
    geom = feature.geometry()
    x = geom.GetX()
    y = geom.GetY()

    xOffset = int( (x - trans['originX']) / trans['pixWidth'] )
    yOffset = int( (y - trans['originY']) / trans['pixHeight'] )

    data = band.ReadAsArray(xOffset, yOffset, 1, 1)
    feature.raster_value = data[0,0]
    return feature

def open_shapefile(filename):
    """Take a file path string and return an ogr shape"""
    # open the shapefile and get the layer
    driver = ogr.GetDriverByName('ESRI Shapefile')
    shp = driver.Open(filename)
    if shp is None:
        print 'Could not open %s' % filename
        sys.exit(1)
    return shp
